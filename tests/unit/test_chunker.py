"""Section detection, cross-reference extraction, and splitting.

Author-owned module (CLAUDE.md rule 2); these tests pin the behavior each
`# DECISION:` comment claims, so changing a decision breaks a test with a name that
says what changed.
"""

from __future__ import annotations

from app.ingestion.chunker import _match_heading, chunk_document, extract_references
from app.ingestion.pdf_parser import ParsedDocument, ParsedPage


def doc(*pages: str) -> ParsedDocument:
    from pathlib import Path

    return ParsedDocument(
        source_path=Path("test.pdf"),
        pages=[ParsedPage(page_number=i, text=t) for i, t in enumerate(pages, start=1)],
    )


def chunks_of(*pages: str, max_tokens: int = 512, overlap: int = 0):
    return chunk_document(
        doc(*pages),
        document_id="doc-900",
        document_title="Test",
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        chars_per_token=4.0,
    )


class TestHeadingDetection:
    def test_article_heading_with_roman_numeral(self) -> None:
        assert _match_heading("ARTICLE V", []) == (["5"], None)

    def test_section_heading_builds_a_nested_path(self) -> None:
        """8.1 implies membership in 8, derived by splitting rather than by a stack."""
        assert _match_heading("8.1 Limitation of Liability", []) == (
            ["8", "8.1"],
            "Limitation of Liability",
        )

    def test_three_level_section(self) -> None:
        path, _ = _match_heading("8.1.2 Carve-outs", []) or ([], None)
        assert path == ["8", "8.1", "8.1.2"]

    def test_long_dotted_clause_is_a_boundary_with_no_title(self) -> None:
        """455 of 787 numbered lines in the first 5 contracts look like this.

        Treating them as body text loses most section granularity, and with it most of
        what cross-reference resolution has to resolve to.
        """
        line = (
            '1.2 "Acquired Party Family" means in the case of a Change of Control of a Party '
            "or its Affiliate, such Party or such Affiliate existing immediately prior thereto."
        )
        result = _match_heading(line, [])
        assert result is not None
        path, title = result
        assert path == ["1", "1.2"]
        assert title is None, "a 200-char sentence is a clause body, not a section title"

    def test_long_bare_integer_line_is_prose_not_a_heading(self) -> None:
        """The guard against false positives. Observed cases are all prose."""
        line = (
            "1 Clinical Trials, and Phase 2 Clinical Trials that are not Registrational "
            "Clinical Trials, and (b) Sanofi for all other purposes under this Agreement."
        )
        assert _match_heading(line, []) is None

    def test_section_reference_in_prose_is_not_a_heading(self) -> None:
        line = (
            "Section 306 of the FFDCA or analogous provisions of Applicable Law outside "
            "the United States, or that is the subject of a pending proceeding."
        )
        assert _match_heading(line, []) is None

    def test_subsection_extends_the_parent_path_instead_of_replacing_it(self) -> None:
        """An earlier version returned ["(a)"], making §(a) a citable location. It is not."""
        assert _match_heading("(a) Definitions", ["8", "8.3"]) == (
            ["8", "8.3", "(a)"],
            "Definitions",
        )

    def test_long_subsection_paragraph_stays_with_its_parent_section(self) -> None:
        """(a), (b), (c) under a liability cap ARE its carve-outs.

        Splitting them into separate chunks severs the cap from its exceptions, which
        is the exact failure cross-reference resolution exists to prevent.
        """
        line = (
            "(a) any liability arising from gross negligence or wilful misconduct of "
            "either party or its agents, which shall not be subject to the cap set out above."
        )
        assert _match_heading(line, ["8", "8.1"]) is None

    def test_blank_line_is_not_a_heading(self) -> None:
        assert _match_heading("   ", []) is None


class TestReferenceExtraction:
    def test_extracts_section_references(self) -> None:
        assert extract_references("subject to Section 8.3 below") == ["8.3"]

    def test_deduplicates_preserving_order(self) -> None:
        text = "See Section 8.3, and Section 2.1, and again Section 8.3."
        assert extract_references(text) == ["8.3", "2.1"]

    def test_normalizes_roman_article_references(self) -> None:
        assert extract_references("as defined in Article IV") == ["4"]

    def test_no_references_yields_empty_list_not_none(self) -> None:
        """The field is always populated so 'none found' differs from 'field dropped'."""
        assert extract_references("plain text with no references") == []


class TestChunking:
    def test_provenance_survives_on_every_chunk(self) -> None:
        result = chunks_of("8.1 Liability\nThe cap is $1,000,000.")
        assert result
        for chunk in result:
            assert chunk.document_id == "doc-900"
            assert chunk.page_number >= 1
            assert chunk.chunk_id

    def test_chunks_never_span_a_page_boundary(self) -> None:
        """Preserving page boundaries is what makes a single page_number honest."""
        result = chunks_of("8.1 Liability\nFirst page text.", "8.2 Term\nSecond page text.")
        pages = {c.page_number for c in result}
        assert pages == {1, 2}
        for chunk in result:
            source = "First page" if chunk.page_number == 1 else "Second page"
            other = "Second page" if chunk.page_number == 1 else "First page"
            if source in chunk.text:
                assert other not in chunk.text

    def test_a_section_continuing_onto_a_new_page_inherits_its_path(self) -> None:
        result = chunks_of("8.1 Liability\nThe cap is", "continued text on the next page.")
        second = [c for c in result if c.page_number == 2]
        assert second and second[0].section_path == ["8", "8.1"]

    def test_page_with_no_headings_still_produces_a_citable_chunk(self) -> None:
        """A silently unretrievable document is the worst outcome."""
        result = chunks_of("Just some unnumbered prose with no section markers at all.")
        assert len(result) == 1
        assert result[0].page_number == 1
        assert result[0].section_path == []

    def test_empty_page_produces_no_chunk(self) -> None:
        assert chunks_of("   \n\n  ") == []

    def test_long_section_is_split_within_the_token_budget(self) -> None:
        body = "\n\n".join(f"Paragraph {i} with some filler text." * 3 for i in range(40))
        result = chunks_of(f"8.1 Long Section\n{body}", max_tokens=64)
        assert len(result) > 1
        for chunk in result:
            assert chunk.token_count <= 64 * 1.5, "hard cut should bound chunk size"
        assert all(c.section_path == ["8", "8.1"] for c in result)

    def test_outbound_references_are_populated_on_chunks(self) -> None:
        result = chunks_of("8.1 Liability\nExcept as set out in Section 8.3, the cap applies.")
        assert any("8.3" in c.outbound_references for c in result)

    def test_chunk_ids_are_unique_and_ordered(self) -> None:
        result = chunks_of("8.1 A\ntext one", "8.2 B\ntext two", "8.3 C\ntext three")
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)
