"""Section detection, cross-reference extraction, and splitting.

Author-owned module (CLAUDE.md rule 2); these tests pin the behavior each
`# DECISION:` comment claims, so changing a decision breaks a test with a name that
says what changed.
"""

from __future__ import annotations

from app.ingestion.chunker import (
    _logical_lengths,
    _match_heading,
    chunk_document,
    extract_references,
)
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


class TestHardWrappedText:
    """The corpus is hard-wrapped at ~76 columns, so a physical line is not a paragraph.

    Measuring "short" on the wrap column made the first line of every long subsection
    look like a heading, which severed carve-outs from the caps they qualify.
    """

    # Verbatim from doc-019 p.25, the case that surfaced this: a liability cap whose
    # exceptions were split into three chunks of their own.
    WRAPPED_CAP = (
        "6.2      LIMIT OF LIABILITY\n"
        "(a)      FOR ANY BREACH OR DEFAULT BY CHANGEPOINT OF ANY OF THE PROVISIONS OF\n"
        "THIS AGREEMENT, OR WITH RESPECT TO ANY CLAIM ARISING HEREFROM OR RELATED HERETO,\n"
        "EXCEPT FOR ANY CLAIM FOR BREACH OF SECTION 5.2 (UNAUTHORIZED DISCLOSURE OF\n"
        "CONFIDENTIAL INFORMATION), CHANGEPOINT SHALL NOT BE LIABLE TO CUSTOMER FOR AN\n"
        "AMOUNT EXCEEDING THE LICENSE FEES PAID UNDER THIS AGREEMENT."
    )

    def test_wrapped_subsection_is_measured_by_paragraph_not_by_wrap_column(self) -> None:
        """The 77-char first line opens a 300-char paragraph. It is body, not a heading."""
        lines = self.WRAPPED_CAP.splitlines()
        logical = _logical_lengths(lines)
        assert len(lines[1]) <= 90, "the physical line is short enough to look like a heading"
        assert logical[1] > 90, "the paragraph it opens is not"
        assert _match_heading(lines[1], ["6", "6.2"], logical_length=logical[1]) is None

    def test_a_heading_line_stays_a_heading_when_the_next_line_opens_its_own_paragraph(
        self,
    ) -> None:
        """Continuation stops at a marker, so "6.2 LIMIT OF LIABILITY" measures 27, not 400."""
        lines = self.WRAPPED_CAP.splitlines()
        logical = _logical_lengths(lines)
        assert _match_heading(lines[0], [], logical_length=logical[0]) == (
            ["6", "6.2"],
            "LIMIT OF LIABILITY",
        )

    def test_cap_and_its_carve_out_land_in_one_chunk(self) -> None:
        """The failure that started this: the cap was citable, the exception was not."""
        result = chunks_of(self.WRAPPED_CAP)
        assert len(result) == 1
        assert "LIMIT OF LIABILITY" in result[0].text
        assert "EXCEPT FOR ANY CLAIM" in result[0].text
        assert result[0].section_path == ["6", "6.2"]

    def test_wrapped_clause_does_not_store_a_truncated_prose_fragment_as_its_title(self) -> None:
        """section_title is a label. A wrap-truncated sentence in it is not a label."""
        page = (
            '1.2      "Acquired Party Family" means, in the case of a Change of Control\n'
            "of a Party or its Affiliate, such Party or such Affiliate existing immediately\n"
            "prior to the closing of such Change of Control."
        )
        result = chunks_of(page)
        assert all(c.section_title is None for c in result)


class TestBareHeadings:
    def test_a_heading_severed_from_its_body_is_folded_into_it(self) -> None:
        """A heading-only chunk is a strong lexical match for text it does not contain."""
        result = chunks_of("12.2 Termination.\n(a) Terminations by Sanofi. Sanofi may terminate.")
        assert len(result) == 1
        assert result[0].text.startswith("12.2 Termination.")
        assert "Sanofi may terminate" in result[0].text

    def test_the_folded_chunk_keeps_the_descendant_path_so_both_ids_resolve(self) -> None:
        """find_section matches on containment, so the ancestor id still finds the chunk."""
        result = chunks_of("12.2 Termination.\n(a) Terminations by Sanofi. Sanofi may terminate.")
        assert result[0].section_path == ["12", "12.2", "(a)"]
        assert "12.2" in result[0].section_path

    def test_folding_chains_through_consecutive_bare_headings(self) -> None:
        result = chunks_of("8. LIABILITY\n8.1 CAP\n(a) The cap is one million dollars.")
        assert len(result) == 1
        assert result[0].text.splitlines() == [
            "8. LIABILITY",
            "8.1 CAP",
            "(a) The cap is one million dollars.",
        ]

    def test_a_short_but_complete_section_is_not_folded_into_its_sibling(self) -> None:
        """'1.65 "Territory" means Japan.' is 7 tokens and complete. Siblings stay apart."""
        result = chunks_of('1.64 "Term" means ten years.\n1.65 "Territory" means Japan.')
        assert len(result) == 2
        assert result[0].section_path == ["1", "1.64"]
        assert result[1].section_path == ["1", "1.65"]

    def test_a_carried_continuation_line_is_not_mistaken_for_a_bare_heading(self) -> None:
        """The first block on a page has no heading of its own; its one line is body text."""
        result = chunks_of("8.1 Liability\nThe cap is", "one million dollars.\n(a) Except:")
        second = [c for c in result if c.page_number == 2]
        assert second[0].text.startswith("one million dollars.")


class TestDecimalArticleNumbering:
    """Three documents number articles "1.0", "2.0". A trailing zero is not a subsection."""

    def test_decimal_article_number_collapses_to_the_article(self) -> None:
        assert _match_heading("9.0 Limitation of Liability", []) == (
            ["9"],
            "Limitation of Liability",
        )

    def test_a_real_subsection_keeps_its_full_path(self) -> None:
        assert _match_heading("9.1 IBM's Limitation of Liability", []) == (
            ["9", "9.1"],
            "IBM's Limitation of Liability",
        )

    def test_ten_point_zero_is_article_ten_not_article_one(self) -> None:
        """The strip is on components, not on characters."""
        assert _match_heading("10.0 Notices", []) == (["10"], "Notices")

    def test_references_are_normalized_the_same_way_as_paths(self) -> None:
        """A reference only resolves if it is spelled the way the target is labeled."""
        assert extract_references("as provided in Section 9.0 above") == ["9"]

    def test_the_heading_and_its_first_clause_now_land_in_one_chunk(self) -> None:
        """9.0/9.1 were siblings, so the bare-heading fold could not reunite them."""
        result = chunks_of("9.0   Limitation of Liability\n9.1   Circumstances may arise where.")
        assert len(result) == 1
        assert result[0].text.startswith("9.0   Limitation of Liability")
        assert result[0].section_path == ["9", "9.1"]


class TestSplitRemainders:
    def test_a_trailing_fragment_is_merged_back_not_emitted(self) -> None:
        """64 chunks in the corpus were 'ion.', 'es.', 'r.' -- unretrievable by anyone."""
        body = "This sentence is filler repeated to overflow the budget. " * 12
        result = chunks_of(f"8.1 Long Section\n{body}ion.", max_tokens=64)
        assert len(result) > 1
        assert all(c.token_count >= 20 for c in result[1:]), [c.text for c in result]
        assert result[-1].text.endswith("ion.")

    def test_a_short_but_complete_section_is_not_subject_to_the_floor(self) -> None:
        """The floor applies to splitter tails, never to a section that is simply short."""
        result = chunks_of('1.65 "Territory" means Japan.')
        assert len(result) == 1
        assert result[0].token_count < 20
