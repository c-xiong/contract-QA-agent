"""Experiment C: citation gate ablation. See SPEC 16.3.

What does the deterministic citation gate actually buy? Three conditions:

  gate_off  the writer's answer is returned as written. Citations are parsed for
            reporting but never checked, and nothing is ever repaired or abstained
            on citation grounds. This is what a system without a gate does.
  verify_only  verify and abstain on failure, with no repair.
  verify_repair  the full layer-1 gate: document exists, page exists, section exists,
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
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from app.agent.graph import build_graph
from app.agent.llm import ModelClient, ModelResponse, StubClient
from app.agent.runner import ResearchAgent
from app.agent.state import Budget
from app.config import get_settings
from app.evidence.citation_parser import extract_citations
from app.evidence.citation_verifier import CitationVerifier, VerificationResult
from app.ingestion.store import ChunkStore, StoreError
from app.retrieval.factory import build_retriever
from app.schemas.evidence import Evidence
from app.schemas.retrieval import RetrievedChunk
from evals.citation_metrics import CitationAblationObservation, summarize_citation_ablation
from evals.loader import DatasetError, load_suite
from evals.provenance import capture_provenance
from experiments._harness import (
    ConditionResult,
    limited_tasks,
    paired_summary,
    record,
    write_conditions_artifact,
)


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


class RecordingWriter(StubClient):
    """Observe raw drafts without changing the delegated model's responses."""

    def __init__(self, delegate: ModelClient, drafts: list[str]) -> None:
        super().__init__()
        self.delegate = delegate
        self.drafts = drafts
        self.model_id = delegate.model_id

    def with_evidence(self, evidence: list[Evidence]) -> RecordingWriter:
        active = (
            self.delegate.with_evidence(evidence)
            if isinstance(self.delegate, StubClient)
            else self.delegate
        )
        return RecordingWriter(active, self.drafts)

    async def complete(self, system: str, user: str) -> ModelResponse:
        response = await self.delegate.complete(system, user)
        self.drafts.append(response.text)
        return response


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        tasks = limited_tasks(load_suite(args.suite), args.limit)
        store = ChunkStore.load(settings.processed_dir)
    except (DatasetError, StoreError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if args.fault_every < 0 or (args.fault_every and settings.live_model):
        print("REFUSING: fault injection must be non-negative and is stub-only.", file=sys.stderr)
        return 1
    retriever = build_retriever(args.arm, store, settings.index_dir)
    base = Budget(max_chunks_per_query=settings.retrieval_top_k)
    budgets = {
        "ungated": replace(base, max_repair_attempts=0),
        "verify_only": replace(base, max_repair_attempts=0),
        "verify_repair": base,
    }
    agents = {
        name: ResearchAgent(store, settings, retriever=retriever, answerability_gate=False)
        for name in budgets
    }
    model_id = agents["ungated"].client.model_id
    provenance = capture_provenance(
        store,
        dataset_name=args.suite,
        dataset_version=tasks[0].dataset_version,
        retrieval_arm=retriever.name,
        top_k=settings.retrieval_top_k,
        model_id=model_id,
    )
    print(f"Experiment C: 3 arms x {len(tasks)} tasks, 1 trial; model={model_id}")
    print(f"Projected research calls: {3 * len(tasks)}; writer calls <= {4 * len(tasks)}")
    print(f"Fault injection: {args.fault_every or 'off'}; answerability gate: off")
    conditions = {name: ConditionResult(name=name) for name in budgets}
    audit = CitationVerifier(store)
    final_results = {}
    raw_answers = {}
    raw_audits = {}
    final_audits = {}
    raw_details = {}

    async def run_condition(name: str, agent: ResearchAgent) -> None:
        print(f"running {name}...", flush=True)
        for task_index, task in enumerate(tasks):
            # DECISION: reset injection by task, so a repair call cannot shift the
            # fault schedule for every subsequent question in just one arm.
            delegate = agent.client
            if args.fault_every:
                delegate = FaultInjectingClient(every=args.fault_every, counter=[task_index])
            drafts: list[str] = []
            writer = RecordingWriter(delegate, drafts)
            verifier = UngatedVerifier(store) if name == "ungated" else agent.verifier
            agent.graph = build_graph(retriever, verifier, writer, store, answerability_gate=False)
            start = perf_counter()
            result = await agent.research(
                task.question, allowed_document_ids=task.allowed_document_ids, budget=budgets[name]
            )
            elapsed = perf_counter() - start
            record(
                conditions[name],
                task,
                result,
                settings.retrieval_top_k,
                excluded_metrics=frozenset({"post_gate_citation_validity"})
                if name == "ungated"
                else frozenset(),
                elapsed_seconds=elapsed,
            )
            key = (name, task.task_id)
            final_results[key] = result
            raw = drafts[0] if drafts else ""
            raw_answers[key] = raw
            raw_audits[key] = audit.verify(
                raw, result.retrieved, allowed_document_ids=task.allowed_document_ids
            )
            final_audits[key] = audit.verify(
                result.answer, result.retrieved, allowed_document_ids=task.allowed_document_ids
            )
            raw_details[f"{name}/{task.task_id}"] = {
                "raw_answer": raw,
                "draft_count": len(drafts),
                "raw_failure_codes": [e.code for e in raw_audits[key].errors],
                "delivered": result.status == "completed",
            }

    await asyncio.gather(*(run_condition(name, agent) for name, agent in agents.items()))
    comparisons = {}
    for name in ("verify_only", "verify_repair"):
        observations = []
        for task in tasks:
            key, off = (name, task.task_id), ("ungated", task.task_id)
            result = final_results[key]
            observations.append(
                CitationAblationObservation(
                    ungated_answer=raw_answers[off],
                    ungated_audit=raw_audits[off],
                    gated_answer=result.answer,
                    gated_audit=final_audits[key],
                    gated_delivered=result.status == "completed",
                    repair_triggered=any(event.step == "repair" for event in result.trace),
                )
            )
        comparisons[name] = summarize_citation_ablation(observations)
        print(f"{name}: {comparisons[name]}")
    raw_by_arm = {}
    for name in budgets:
        observations = []
        for task in tasks:
            key = (name, task.task_id)
            result = final_results[key]
            observations.append(
                CitationAblationObservation(
                    ungated_answer=raw_answers[key],
                    ungated_audit=raw_audits[key],
                    gated_answer=result.answer,
                    gated_audit=final_audits[key],
                    gated_delivered=result.status == "completed",
                    repair_triggered=any(event.step == "repair" for event in result.trace),
                )
            )
        raw_by_arm[name] = summarize_citation_ablation(observations)
    failures = Counter(
        e.code
        for (name, _), audit_result in raw_audits.items()
        if name == "ungated"
        for e in audit_result.errors
    )
    path = write_conditions_artifact(
        args.out,
        "experiment-c",
        list(conditions.values()),
        tasks,
        {
            "retriever": retriever.name,
            "model_id": model_id,
            "answerability_gate": False,
            "citation_auditor_version": "citation_ablation@2",
            "citation_comparisons": comparisons,
            "raw_and_final_by_arm": raw_by_arm,
            "raw_drafts": raw_details,
            "failure_codes": dict(failures),
            "fault_injection_every": args.fault_every,
            "paired_contrasts": {
                "verify_vs_ungated": paired_summary(
                    conditions["ungated"], conditions["verify_only"]
                ),
                "repair_vs_verify": paired_summary(
                    conditions["verify_only"], conditions["verify_repair"]
                ),
            },
            "initial_drafts_shared": False,
            "limitations": [
                "One trial; independently sampled initial drafts can differ across arms.",
                "Citation validity measures location and format, not semantic claim support.",
            ],
        },
        provenance=provenance,
        output=args.output,
    )
    print(f"Artifact: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="hand")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, help="exact output filename")
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
