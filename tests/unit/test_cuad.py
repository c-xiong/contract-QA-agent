"""The CUAD adapter. Every case here was observed in the actual v1 release."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.cuad import (
    CATEGORY_COLUMNS,
    CuadFormatError,
    index_contract_pdfs,
    normalize_filename,
    parse_clause_spans,
    resolve_pdf,
)


class TestCategoryColumns:
    def test_the_irregular_column_is_hardcoded_not_derived(self) -> None:
        """The one trap in the release: a space after the hyphen.

        Deriving this name as f"{category}-Answer" silently yields an empty column,
        dropping the numeric category SPEC 6.2 specifically wants for
        query-refinement tasks. This test is the guard.
        """
        _, answer_column = CATEGORY_COLUMNS["Notice Period To Terminate Renewal"]
        assert answer_column == "Notice Period To Terminate Renewal- Answer"
        assert answer_column != "Notice Period To Terminate Renewal-Answer"

    def test_every_category_has_a_distinct_clause_and_answer_column(self) -> None:
        for category, (clause, answer) in CATEGORY_COLUMNS.items():
            assert clause != answer, category
            assert clause == category, f"{category}: clause column should equal the name"


class TestParseClauseSpans:
    def test_parses_a_python_list_literal(self) -> None:
        """Clause cells are list literals stored as text, not plain strings."""
        assert parse_clause_spans("['first span', 'second span']") == [
            "first span",
            "second span",
        ]

    def test_absent_annotation_yields_no_spans(self) -> None:
        assert parse_clause_spans("") == []
        assert parse_clause_spans("   ") == []
        assert parse_clause_spans("[]") == []

    def test_malformed_cell_is_treated_as_absent_not_raised(self) -> None:
        """Indistinguishable from absence for retrieval; must not crash ingestion."""
        assert parse_clause_spans("['unterminated") == []

    def test_quoted_string_cell_is_accepted(self) -> None:
        """Defensive: a quoted scalar parses to a str, not a list."""
        assert parse_clause_spans("'a single clause'") == ["a single clause"]

    def test_unquoted_text_is_treated_as_absent(self) -> None:
        """Measured: all 4080 clause cells in the 8 categories in scope are list
        literals. Unquoted text does not occur, so treating it as absent cannot lose
        real annotation; if a future release changes format, the coverage counts drop
        visibly rather than ingestion crashing."""
        assert parse_clause_spans("just a clause") == []

    def test_embedded_apostrophes_survive(self) -> None:
        spans = parse_clause_spans('["Company\'s liability shall not exceed"]')
        assert spans == ["Company's liability shall not exceed"]

    def test_blank_entries_are_dropped(self) -> None:
        assert parse_clause_spans("['real span', '', '   ']") == ["real span"]


class TestNormalizeFilename:
    def test_case_and_extension_differences_collapse(self) -> None:
        assert normalize_filename("Agreement.PDF") == normalize_filename("agreement.pdf")

    def test_ampersand_and_apostrophe_map_to_underscore(self) -> None:
        """The CSV keeps them; the packaged PDFs replaced them. 7 of 510 rows."""
        assert normalize_filename("MOELIS&CO.PDF") == normalize_filename("MOELIS_CO.PDF")
        assert normalize_filename("MACY'S,INC.PDF") == normalize_filename("MACY_S,INC.PDF")

    def test_trailing_quote_is_stripped(self) -> None:
        """One row in the release has a stray apostrophe after the extension."""
        assert normalize_filename("KALLOINC.PDF'") == normalize_filename("KALLOINC.PDF")

    def test_space_before_extension_is_trimmed(self) -> None:
        assert normalize_filename("Monsanto .PDF") == normalize_filename("Monsanto.PDF")

    def test_unicode_normalizes(self) -> None:
        """macOS stores filenames decomposed; the CSV is composed."""
        assert normalize_filename("LECLANCHÉ.PDF") == normalize_filename("LECLANCHÉ.PDF")

    def test_distinct_names_stay_distinct(self) -> None:
        assert normalize_filename("alpha.pdf") != normalize_filename("beta.pdf")


class TestResolvePdf:
    def test_override_handles_the_one_genuine_rename(self) -> None:
        """One CSV row names a file the release renamed, adding a suffix.

        Listed explicitly rather than fuzzy-matched: guessing which contract an expert
        annotation belongs to would mislabel ground truth undetectably.
        """
        csv_name = (
            "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18"
            "_Development Agreement.PDF"
        )
        disk_name = (
            "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18"
            "_Development Agreement_Option Agreement.pdf"
        )
        index = {normalize_filename(disk_name): Path("/corpus") / disk_name}
        assert resolve_pdf(csv_name, index) == Path("/corpus") / disk_name

    def test_unresolvable_name_returns_none_rather_than_guessing(self) -> None:
        assert resolve_pdf("nonexistent.pdf", {}) is None


class TestIndexContractPdfs:
    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CuadFormatError, match="No PDFs"):
            index_contract_pdfs(tmp_path)

    def test_collision_raises_rather_than_silently_overwriting(self, tmp_path: Path) -> None:
        """Two files normalizing to one key means normalization is too aggressive."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "Deal&Co.PDF").write_bytes(b"%PDF-")
        (tmp_path / "b" / "Deal_Co.pdf").write_bytes(b"%PDF-")
        with pytest.raises(CuadFormatError, match="normalize to the same key"):
            index_contract_pdfs(tmp_path)
