"""Experiment C: citation gate ablation. See SPEC 16.3.

What does the deterministic citation gate actually buy? Two conditions:

  gate_off  the writer's answer is returned as written. Citations are parsed for
            reporting but never checked, and nothing is ever repaired or abstained
            on citation grounds. This is what a system without a gate does.
  gate_on   the full layer-1 gate: document exists, page exists, section exists,
            document is within the allowlist, and the citation points at evidence
            that was actually retrieved. One repair attempt, then abstention.

The number this experiment exists to produce is the rate of citations that would have
reached a reader unverified.

    uv run python experiments/citation_gate_ablation.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

from _harness import ConditionResult, print_report, record, write_artifact

from app.agent.graph import build_graph
from app.agent.llm import ModelResponse, StubClient
from app.agent.runner import ResearchAgent
from app.config import get_settings
from app.evidence.citation_parser import extract_citations
from app.evidence.citation_verifier import CitationVerifier, VerificationResult
from app.evidence.claim_support import extract_cited_claims
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import build_retriever
from app.schemas.evidence import Evidence
from app.schemas.retrieval import RetrievedChunk
from evals.citation_metrics import CitationAblationObservation, summarize_citation_ablation
from evals.loader import DatasetError, load_suite


class FaultInjectingClient(StubClient):
    """A writer that miscites on a fixed fraction of answers.

    DECISION: the gate is measured under fault injection, not only under the honest stub.
      The honest stub cites the evidence it was handed, so it produces zero bad citations
      and the ablation measures nothing -- a 0% unverified rate says the writer behaved,
      not that the gate works. The question the experiment must answer is conditional:
      WHEN the writer miscites, does the gate catch it?
      This client answers that by citing a real, correctly-formatted, in-corpus location
      that was never retrieved for the question -- the single failure mode that passes
      every check except grounding, and the one a naive verifier misses.
      This is fault injection and is labelled as such everywhere it is reported. It is
      NOT a claim about how often a real model miscites; only a live run measures that.
    """

    def __init__(
        self,
        evidence: list[Evidence] | None = None,
        *,
        every: int = 3,
        counter: list[int] | None = None,
    ) -> None:
        super().__init__(evidence)
        self._every = every
        # Shared mutable counter. The graph calls `with_evidence` before every write,
        # which returns a NEW client; a plain int attribute would reset to zero on each
        # call and no fault would ever fire. This bit me once -- the injected condition
        # reported a 0% unverified rate, which looked like the gate finding nothing.
        self._counter = counter if counter is not None else [0]

    def with_evidence(self, evidence: list[Evidence]) -> FaultInjectingClient:
        return FaultInjectingClient(evidence, every=self._every, counter=self._counter)

    async def complete(self, system: str, user: str) -> ModelResponse:
        self._counter[0] += 1
        response = await super().complete(system, user)
        if self._counter[0] % self._every != 0:
            return response
        # A location that exists in the corpus but was not retrieved for this question.
        return ModelResponse(
            text=(
                "The agreement addresses this point. [doc-002, p. 7, §1.1]\n\n"
                "(Fault-injected citation: real location, never retrieved.)"
            ),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model_id="stub+fault",
        )


class UngatedVerifier(CitationVerifier):
    """Parses citations and approves all of them.

    DECISION: the ablation removes the CHECKS, not the parsing.
      Keeping the parser means both conditions produce the same `citations` list, so
      `citation_validity` measures the same thing in both and the difference is purely
      the gate. Removing parsing too would leave nothing to count in the off condition.
    """

    def verify(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        allowed_document_ids: list[str] | None = None,
    ) -> VerificationResult:
        citations, _ = extract_citations(answer)
        return VerificationResult(citations=citations, errors=[], uncited=False)


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        tasks = load_suite(args.suite)
        store = ChunkStore.load(settings.processed_dir)
    except (DatasetError, StoreError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    retriever = build_retriever(args.arm, store, settings.index_dir)
    generated = tasks[0].dataset_version.startswith("provisional-generated")

    # DECISION: fault injection is OFF by default and is only meaningful against the stub.
    #   The injecting client subclasses StubClient, so switching it on replaces the real
    #   model entirely. An earlier version installed it unconditionally, which meant a
    #   CRA_LIVE_MODEL=1 run silently measured the injection rate instead of the model --
    #   the number looked like a live measurement and was not one.
    #   With a live model, no injection: the rate reported is then what the model
    #   actually does, which is the number worth having.
    inject = args.fault_every > 0
    if inject and settings.live_model:
        print(
            "REFUSING: --fault-every replaces the live model with a stub, so the result "
            "would not be a live measurement. Drop the flag, or unset CRA_LIVE_MODEL.",
            file=sys.stderr,
        )
        return 1
    if not inject and not settings.live_model:
        print(
            "NOTE: against the stub with no injection the writer always cites correctly, "
            "so the gate has nothing to catch. Pass --fault-every N to exercise it.",
            file=sys.stderr,
        )

    gated = ResearchAgent(store, settings, retriever=retriever)
    ungated = ResearchAgent(store, settings, retriever=retriever)
    if inject:
        gated.client = FaultInjectingClient(every=args.fault_every)
        ungated.client = FaultInjectingClient(every=args.fault_every)
        gated.graph = build_graph(gated.retriever, gated.verifier, gated.client, store)
    ungated.verifier = UngatedVerifier(store)
    ungated.graph = build_graph(ungated.retriever, ungated.verifier, ungated.client, store)

    # The honest gate is the real verifier, run over the ungated condition's output as a
    # measurement instrument. It never influences that condition's behaviour; it only
    # counts what would have reached a reader.
    audit = CitationVerifier(store)

    print("=" * 78)
    print("EXPERIMENT C -- CITATION GATE ABLATION")
    print("=" * 78)
    print(f"corpus    : {len(store.documents)} documents, {len(store)} chunks")
    print(f"retriever : {retriever.name}")
    print(f"model     : {gated.client.model_id}")
    print(f"suite     : {args.suite} ({tasks[0].dataset_version}), n={len(tasks)}")
    if inject:
        print(f"fault     : injected, 1 in {args.fault_every} answers")
        print()
        print("!! FAULT INJECTION. Every N-th answer is deliberately miscited, to ask the")
        print("!! conditional question: WHEN the writer errs, does the gate catch it? The")
        print("!! rate is a TEST PARAMETER and says nothing about how often a real model")
        print("!! miscites.")
    else:
        print("fault     : none -- measuring what the model actually does")
    print()

    baseline = ConditionResult(name="gate_off")
    variant = ConditionResult(name="gate_on")

    invalid_by_code: Counter[str] = Counter()
    affected_tasks: list[str] = []
    ungated_audits: dict[str, VerificationResult] = {}
    gated_audits: dict[str, VerificationResult] = {}
    gated_answers: dict[str, str] = {}
    gated_delivered: dict[str, bool] = {}
    repair_triggered: dict[str, bool] = {}

    for label, agent, condition in (("gate_off", ungated, baseline), ("gate_on", gated, variant)):
        print(f"running {label}...", flush=True)
        for task in tasks:
            result = await agent.research(
                task.question, allowed_document_ids=task.allowed_document_ids
            )
            record(
                condition,
                task,
                result,
                settings.retrieval_top_k,
                excluded_metrics=(
                    frozenset({"post_gate_citation_validity"})
                    if label == "gate_off"
                    else frozenset()
                ),
            )

            if label == "gate_off":
                audited = audit.verify(
                    result.answer,
                    result.retrieved,
                    allowed_document_ids=task.allowed_document_ids,
                )
                ungated_audits[task.task_id] = audited
                for error in audited.errors:
                    invalid_by_code[error.code] += 1
                claims = extract_cited_claims(result.answer) if not result.abstained else []
                if audited.errors or any(not claim.citations for claim in claims):
                    affected_tasks.append(task.task_id)
            else:
                gated_audits[task.task_id] = audit.verify(
                    result.answer,
                    result.retrieved,
                    allowed_document_ids=task.allowed_document_ids,
                )
                gated_answers[task.task_id] = result.answer
                gated_delivered[task.task_id] = result.status == "completed"
                repair_triggered[task.task_id] = any(
                    event.step == "repair" for event in result.trace
                )

    observations = [
        CitationAblationObservation(
            ungated_answer=baseline.answers[task.task_id],
            ungated_audit=ungated_audits[task.task_id],
            gated_answer=gated_answers[task.task_id],
            gated_audit=gated_audits[task.task_id],
            gated_delivered=gated_delivered[task.task_id],
            repair_triggered=repair_triggered[task.task_id],
        )
        for task in tasks
    ]
    citation_metrics = summarize_citation_ablation(observations)

    print("\n" + "=" * 78)
    print("WHAT THE GATE CAUGHT")
    print("=" * 78)
    print(f"raw citation attempts               : {citation_metrics['raw_citation_attempts']}")
    print(f"raw invalid citations               : {citation_metrics['raw_invalid_citations']}")
    print(
        f"raw invalid-citation rate          : {citation_metrics['raw_invalid_citation_rate']:.1%}"
    )
    print(
        "unsafe answer exposure rate        : "
        f"{citation_metrics['unsafe_answer_exposure_rate']:.1%}"
    )
    post_gate = citation_metrics["post_gate_citation_validity"]
    post_gate_display = f"{post_gate:.1%}" if post_gate is not None else "not applicable"
    print(f"post-gate citation validity         : {post_gate_display}")
    print(f"tasks affected                      : {len(affected_tasks)}/{len(tasks)}")
    if invalid_by_code:
        print("\nby failure code:")
        for code, count in invalid_by_code.most_common():
            print(f"  {code:24s} {count}")
    if affected_tasks:
        print(f"\naffected: {', '.join(affected_tasks[:10])}")

    print_report(
        "EXPERIMENT C",
        baseline,
        variant,
        tasks,
        k=settings.retrieval_top_k,
        generated=generated,
        notes=[
            "Both conditions share retriever, writer, corpus, and budget. Only citation",
            "  verification differs.",
            "The gate is layer 1 only: it proves a citation points somewhere real that",
            "  was retrieved. It cannot prove the text supports the claim -- that is",
            "  layer 2 (claim support), and it is not exercised here.",
            (
                f"Faults injected 1-in-{args.fault_every}: a test parameter, not an "
                "estimate of how often a model miscites."
                if inject
                else "No injection: the rate below is what this model actually produced."
            ),
        ],
    )

    path = write_artifact(
        args.out,
        "experiment-c",
        baseline,
        variant,
        tasks,
        {
            "retriever": retriever.name,
            "model_id": gated.client.model_id,
            **citation_metrics,
            "failure_codes": dict(invalid_by_code),
            "affected_tasks": affected_tasks,
            "fault_injection_every": args.fault_every,
        },
    )
    print(f"\nArtifact: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--arm", default="rrf_hybrid_rerank")
    parser.add_argument(
        "--fault-every",
        type=int,
        default=0,
        help="inject a deliberately bad citation every Nth answer. Stub-only, and off by "
        "default: with a live model the point is to measure the model, not the injector.",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
