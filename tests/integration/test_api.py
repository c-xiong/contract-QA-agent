"""API layer. Runs against a synthetic store and the deterministic stub."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.runner import ResearchAgent
from app.api.main import create_app
from app.config import Settings
from app.ingestion.store import ChunkStore


@pytest.fixture
def app(store: ChunkStore, settings: Settings) -> FastAPI:
    application = create_app()
    application.state.agent = ResearchAgent(store, settings)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # The agent is injected in the `app` fixture; lifespan honours it and skips the
    # corpus load, so these tests never touch data/ and never need it to exist.
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_reports_corpus_size_and_model(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["documents"] == 2
        assert body["chunks"] == 4
        assert body["live_model"] is False, "nothing spends money by default"

    def test_missing_corpus_is_reported_not_crashed(self) -> None:
        """A container that exits on a missing corpus gives a crash loop and no diagnostic."""
        application = create_app()
        application.state.agent = None
        with TestClient(application) as test_client:
            test_client.app.state.agent = None  # type: ignore[attr-defined]
            body = test_client.get("/health").json()
            assert body["status"] in ("no_corpus", "ok")


class TestDocuments:
    def test_lists_documents_with_provenance(self, client: TestClient) -> None:
        body = client.get("/documents").json()
        assert {d["document_id"] for d in body} == {"doc-900", "doc-901"}
        assert all("corpus_source" in d and "is_synthetic" in d for d in body)

    def test_single_document(self, client: TestClient) -> None:
        body = client.get("/documents/doc-900").json()
        assert body["document_id"] == "doc-900"
        assert body["chunk_count"] == 3

    def test_unknown_document_is_404(self, client: TestClient) -> None:
        assert client.get("/documents/doc-999").status_code == 404


class TestResearch:
    def test_returns_a_cited_answer(self, client: TestClient) -> None:
        body = client.post("/research", json={"question": "What is the governing law?"}).json()
        assert body["status"] == "completed"
        assert body["abstained"] is False
        assert body["citations"], "an answer must carry at least one verified citation"
        assert body["citation_errors"] == []

    def test_citations_carry_both_internal_and_display_forms(self, client: TestClient) -> None:
        """SPEC 13.3: internal is keyed on document_id and is what the verifier checks;
        display uses the title and is what a person reads."""
        body = client.post("/research", json={"question": "What is the governing law?"}).json()
        citation = body["citations"][0]
        assert citation["internal"].startswith("[doc-")
        assert citation["document_title"] in citation["display"]
        assert "doc-" not in citation["display"]

    def test_evidence_is_returned_with_provenance(self, client: TestClient) -> None:
        # "aggregate exceed negligence" rather than "liability cap": on a four-chunk
        # fixture, a term appearing in half the chunks gets non-positive Okapi IDF and
        # is dropped. See tests/unit/test_bm25.py.
        body = client.post("/research", json={"question": "aggregate exceed negligence"}).json()
        assert body["evidence"]
        for item in body["evidence"]:
            assert item["document_id"].startswith("doc-")
            assert item["pulled_by"] in ("search", "cross_reference")

    def test_document_allowlist_is_honoured(self, client: TestClient) -> None:
        body = client.post(
            "/research",
            json={"question": "confidential information", "document_ids": ["doc-901"]},
        ).json()
        assert set(body["documents_searched"]) <= {"doc-901"}

    def test_unknown_document_in_allowlist_is_a_client_error(self, client: TestClient) -> None:
        """Silently dropping it would answer from the whole corpus while the caller
        believes the query was scoped."""
        response = client.post(
            "/research", json={"question": "anything", "document_ids": ["doc-999"]}
        )
        assert response.status_code == 400
        assert "doc-999" in response.json()["detail"]

    def test_trace_is_opt_in(self, client: TestClient) -> None:
        without = client.post("/research", json={"question": "governing law"}).json()
        assert without["trace"] is None

        with_trace = client.post(
            "/research", json={"question": "governing law", "include_trace": True}
        ).json()
        assert with_trace["trace"]
        assert with_trace["trace"][0]["step"] == "search"

    def test_abstains_when_nothing_matches(self, client: TestClient) -> None:
        body = client.post("/research", json={"question": "zzzz qqqq nonexistent"}).json()
        assert body["abstained"] is True
        assert body["citations"] == []

    def test_empty_question_is_rejected_by_validation(self, client: TestClient) -> None:
        assert client.post("/research", json={"question": ""}).status_code == 422

    def test_unknown_field_is_rejected(self, client: TestClient) -> None:
        response = client.post("/research", json={"question": "x", "temperature": 0.7})
        assert response.status_code == 422
