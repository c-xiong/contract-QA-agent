"""Human-readable titles derived from CUAD's irregular source filenames."""

from __future__ import annotations

from datetime import date

import pytest

from app.ingestion.manifest import ManifestEntry
from app.ingestion.pipeline import title_from_entry


def entry(source_filename: str) -> ManifestEntry:
    return ManifestEntry(
        document_id="doc-900",
        source_filename=source_filename,
        relative_path=f"CUAD_v1/full_contract_pdf/{source_filename}",
        corpus_source="cuad",
        added=date(2026, 8, 25),
    )


@pytest.mark.parametrize(
    ("source_filename", "expected"),
    [
        (
            "RevolutionMedicinesInc_20200117_S-1_EX-10.1_11948417_"
            "EX-10.1_Development Agreement.pdf",
            "Development Agreement",
        ),
        (
            "INKTOMICORP_06_08_1998-EX-10.14-SOFTWARE HOSTING AGREEMENT.PDF",
            "SOFTWARE HOSTING AGREEMENT",
        ),
        (
            "NEONSYSTEMSINC_03_01_1999-EX-10.5-DISTRIBUTOR AGREEMENT_New.pdf",
            "DISTRIBUTOR AGREEMENT",
        ),
        (
            "VirtuosoSurgicalInc_20191227_1-A_EX1A-6 MAT CTRCT_11933379_"
            "EX1A-6 MAT CTRCT_License Agreement.pdf",
            "License Agreement",
        ),
        (
            "2ThemartComInc_19990826_10-12G_EX-10.10_6700288_"
            "EX-10.10_Co-Branding Agreement_ Agency Agreement.pdf",
            "Co-Branding Agreement_ Agency Agreement",
        ),
        (
            "BEYONDCOMCORP_08_03_2000-EX-10.2-CO-HOSTING AGREEMENT.PDF",
            "CO-HOSTING AGREEMENT",
        ),
        ("unstructured-contract-name.pdf", "unstructured-contract-name"),
    ],
)
def test_title_from_cuad_filename(source_filename: str, expected: str) -> None:
    assert title_from_entry(entry(source_filename)) == expected
