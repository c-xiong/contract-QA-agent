"""Citation parsing, including malformed input.

SPEC 13.3 requires this parser be deterministic code with unit tests, explicitly
including tests for malformed citations.
"""

from __future__ import annotations

import pytest

from app.evidence.citation_parser import extract_citations, parse_citation, strip_citations


class TestWellFormed:
    def test_full_form(self) -> None:
        citation = parse_citation("[doc-014, p. 12, §8.1]")
        assert citation is not None
        assert (citation.document_id, citation.page_number, citation.section_id) == (
            "doc-014",
            12,
            "8.1",
        )

    def test_page_only_form(self) -> None:
        """The chunker does not always detect numbering, so a page-only citation is valid."""
        citation = parse_citation("[doc-014, p. 12]")
        assert citation is not None
        assert citation.page_number == 12
        assert citation.section_id is None

    def test_section_only_form(self) -> None:
        """ContractNLI ships unpaginated text (SPEC 6.3); a page is genuinely unavailable."""
        citation = parse_citation("[doc-051, §12]")
        assert citation is not None
        assert citation.page_number is None
        assert citation.section_id == "12"

    @pytest.mark.parametrize(
        "text",
        [
            "[doc-014, page 12, §8.1]",
            "[doc-014, p 12, §8.1]",
            "[doc-014,p.12,§8.1]",
            "[ doc-014 , p. 12 , §8.1 ]",
            "[doc-014, p. 12, Section 8.1]",
            "[DOC-014, P. 12, §8.1]",
        ],
    )
    def test_near_misses_parse_then_fail_verification_instead_of_parsing(self, text: str) -> None:
        """Permissive on spacing and markers on purpose.

        A near-miss should reach the verifier and fail with a specific error, not die
        here with a generic one. The two failures need different remedies.
        """
        citation = parse_citation(text)
        assert citation is not None
        assert citation.document_id == "doc-014"

    def test_multi_level_and_subsection_identifiers(self) -> None:
        assert parse_citation("[doc-014, p. 1, §8.1.2]").section_id == "8.1.2"  # type: ignore[union-attr]
        assert parse_citation("[doc-014, p. 1, §(a)]").section_id == "(a)"  # type: ignore[union-attr]

    def test_raw_text_is_preserved_for_error_messages(self) -> None:
        citation = parse_citation("[doc-014, p. 12, §8.1]")
        assert citation is not None and citation.raw == "[doc-014, p. 12, §8.1]"

    def test_render_round_trips(self) -> None:
        citation = parse_citation("[doc-014, p. 12, §8.1]")
        assert citation is not None and citation.render() == "[doc-014, p. 12, §8.1]"


class TestMalformed:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "doc-014, p. 12",  # no brackets
            "[doc-14, p. 12]",  # too few digits
            "[document-014, p. 12]",  # wrong prefix
            "[doc-014, p. twelve]",  # non-numeric page
            "[doc-014, p. -3]",  # negative page
            "[p. 12, §8.1]",  # no document
            "[doc-014",  # unclosed
        ],
    )
    def test_rejected(self, text: str) -> None:
        assert parse_citation(text) is None


class TestExtraction:
    def test_finds_multiple_citations_in_prose(self) -> None:
        text = (
            "The cap is $1M [doc-014, p. 12, §8.1], subject to carve-outs [doc-014, p. 13, §8.3]."
        )
        citations, malformed = extract_citations(text)
        assert [c.section_id for c in citations] == ["8.1", "8.3"]
        assert malformed == []

    def test_reports_broken_attempts_separately_from_valid_ones(self) -> None:
        """An uncited claim and a claim with a broken citation are different bugs."""
        citations, malformed = extract_citations("Real [doc-014, p. 12]. Broken [doc-14, p. 99].")
        assert len(citations) == 1
        assert malformed == ["[doc-14, p. 99]"]

    def test_redaction_marks_are_not_broken_citations(self) -> None:
        """CUAD contracts are full of [***]; flagging those would drown real errors."""
        _, malformed = extract_citations("The fee is [***] per unit.")
        assert malformed == []

    def test_ordinary_bracketed_prose_is_not_a_broken_citation(self) -> None:
        _, malformed = extract_citations("The party [sic] shall indemnify.")
        assert malformed == []

    def test_no_citations_yields_two_empty_lists(self) -> None:
        assert extract_citations("An answer with no citation at all.") == ([], [])


class TestStripCitations:
    def test_removes_citations_leaving_prose(self) -> None:
        stripped = strip_citations("The cap is $1M [doc-014, p. 12, §8.1] per year.")
        assert "doc-014" not in stripped
        assert "The cap is $1M" in stripped
