"""Run an eval suite and produce a result artifact.

Every run writes a JSON artifact recording the task set, the grader versions, the
retriever, the model, and per-task scores. A number quoted in the README without a
corresponding artifact is not reportable (CLAUDE.md rule 6).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.agent.runner import ResearchAgent, ResearchResult
from app.config import Settings
from evals.graders.answer import grade_answer
from evals.graders.claim_support import claim_support
from evals.graders.retrieval import GradeResult, grade_retrieval
from evals.schema import EvalTask


def _counter(data: dict[str, object], key: str) -> int:
    """Read an integer counter out of a grader's untyped `data` payload.

    Grade payloads are `dict[str, object]` because each grader records different
    detail. Aggregation reads only the counters it knows about, and a missing or
    non-integer value contributes nothing rather than raising: an aggregate is a
    summary, and one malformed grade must not take down a whole run's report.
    """
    value = data.get(key)
    return value if isinstance(value, int) else 0


@dataclass
class TaskRun:
    """One task's outcome, with enough detail to re-grade without re-running."""

    task_id: str
    category: str
    question: str
    expected_behavior: str
    status: str
    abstained: bool
    answer: str
    retrieved_document_ids: list[str]
    retrieved_chunk_ids: list[str]
    citations: list[str]
    citation_errors: list[str]
    grades: list[GradeResult]
    input_tokens: int
    output_tokens: int
    behavior_matched: bool

    @property
    def scores(self) -> dict[str, float]:
        return {g.name: g.score for g in self.grades}


@dataclass
class SuiteReport:
    """Aggregate result for one suite run."""

    suite: str
    dataset_version: str
    retriever: str
    model_id: str
    top_k: int
    started_at: str
    runs: list[TaskRun] = field(default_factory=list)
    grader_versions: dict[str, str] = field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.runs)

    def mean(self, grader: str) -> float:
        values = [r.scores[grader] for r in self.runs if grader in r.scores]
        return sum(values) / len(values) if values else 0.0

    def claim_support_aggregate(self) -> dict[str, float | int] | None:
        """Return citation-scoped macro and micro support for current-version grades."""
        grades = [
            grade
            for run in self.runs
            for grade in run.grades
            if grade.name == "claim_support" and grade.version == "claim_support@2"
        ]
        if not grades:
            return None
        supported = sum(_counter(grade.data, "supported") for grade in grades)
        evaluated = sum(_counter(grade.data, "evaluated_claims") for grade in grades)
        return {
            "macro": sum(grade.score for grade in grades) / len(grades),
            "micro": supported / evaluated if evaluated else 1.0,
            "supported_claims": supported,
            "evaluated_claims": evaluated,
        }

    def citation_coverage_aggregate(self) -> dict[str, float | int] | None:
        """Return macro task coverage and micro factual-claim coverage."""
        grades = [
            grade for run in self.runs for grade in run.grades if grade.name == "citation_coverage"
        ]
        if not grades:
            return None
        cited = sum(_counter(grade.data, "cited_claims") for grade in grades)
        claims = sum(_counter(grade.data, "claims") for grade in grades)
        return {
            "macro": sum(grade.score for grade in grades) / len(grades),
            "micro": cited / claims if claims else 1.0,
            "cited_claims": cited,
            "factual_claims": claims,
        }

    def summary(self) -> str:
        if not self.runs:
            return "no tasks run"

        lines = []
        if self.dataset_version.startswith("provisional-generated"):
            lines += [
                "!! QUESTIONS IN THIS SUITE ARE MODEL-GENERATED. Evidence labels are expert",
                "!! annotation, but the questions were written by the same model family the",
                "!! system uses. These figures are a pipeline check, not a capability",
                "!! measurement. See CLAUDE.md rule 1.",
                "",
            ]
        lines += [
            f"suite            : {self.suite}  (dataset {self.dataset_version})",
            f"tasks            : {self.task_count}",
            f"retriever / model: {self.retriever} / {self.model_id}",
            f"top_k            : {self.top_k}",
            "",
        ]
        graders = sorted({g.name for r in self.runs for g in r.grades})
        for name in graders:
            lines.append(f"{name:24s} mean {self.mean(name):.3f}")
        if support := self.claim_support_aggregate():
            lines.append(f"{'claim_support micro':24s} {support['micro']:.3f}")
        if coverage := self.citation_coverage_aggregate():
            lines.append(f"{'citation_coverage micro':24s} {coverage['micro']:.3f}")

        matched = sum(r.behavior_matched for r in self.runs)
        lines.append(f"{'expected_behavior match':24s} {matched}/{self.task_count}")

        # SPEC 15.7: never report a bare aggregate. With n this small, per-task detail
        # is the finding; the mean is a convenience.
        lines.append("\nper task:")
        for run in self.runs:
            scores = "  ".join(f"{k}={v:.2f}" for k, v in sorted(run.scores.items()))
            flag = "ok " if run.behavior_matched else "BEH"
            lines.append(f"  [{flag}] {run.task_id:20s} {run.status:10s} {scores}")

        lines.append(
            f"\nn={self.task_count}. Too small for a significance claim; report these as "
            "directional only (SPEC 15.7)."
        )
        return "\n".join(lines)

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "suite": self.suite,
            "dataset_version": self.dataset_version,
            "retriever": self.retriever,
            "model_id": self.model_id,
            "top_k": self.top_k,
            "started_at": self.started_at,
            "task_count": self.task_count,
            "grader_versions": self.grader_versions,
            "means": {
                name: self.mean(name)
                for name in sorted({g.name for r in self.runs for g in r.grades})
            },
            "runs": [asdict(run) for run in self.runs],
        }
        if support := self.claim_support_aggregate():
            payload["claim_support_aggregate"] = support
        if coverage := self.citation_coverage_aggregate():
            payload["citation_coverage_aggregate"] = coverage
        return payload


def behavior_matches(task: EvalTask, result: ResearchResult) -> bool:
    """Did the system do the kind of thing the task expected?

    Deliberately coarse at Sprint 0: answered-vs-abstained only. Grading "partial"
    properly needs the coverage grader from SPEC 13.4 layer 3, which does not exist
    yet. Reporting a crude match is honest; inventing a partial-credit rule here
    would be a threshold nobody chose.
    """
    if task.expected_behavior == "abstain":
        return result.abstained
    return not result.abstained and result.status == "completed"


async def run_suite(
    suite: str,
    tasks: list[EvalTask],
    agent: ResearchAgent,
    settings: Settings,
    *,
    with_claim_support: bool = False,
) -> SuiteReport:
    report = SuiteReport(
        suite=suite,
        dataset_version=tasks[0].dataset_version if tasks else "unknown",
        retriever=agent.retriever.name,
        model_id=agent.client.model_id,
        top_k=settings.retrieval_top_k,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    for task in tasks:
        result = await agent.research(task.question, allowed_document_ids=task.allowed_document_ids)
        grades = grade_retrieval(task, result.retrieved, settings.retrieval_top_k)
        grades += grade_answer(task, result)
        if with_claim_support:
            # One model call per factual claim, so it is opt-in: it roughly doubles the
            # cost of a suite run.
            grades.append(await claim_support(task, result, agent.client, live=settings.live_model))
        for grade in grades:
            report.grader_versions[grade.name] = grade.version

        report.runs.append(
            TaskRun(
                task_id=task.task_id,
                category=task.category,
                question=task.question,
                expected_behavior=task.expected_behavior,
                status=result.status,
                abstained=result.abstained,
                answer=result.answer,
                retrieved_document_ids=result.retrieved_document_ids,
                retrieved_chunk_ids=[i.chunk.chunk_id for i in result.retrieved],
                citations=[c.render() for c in result.citations],
                citation_errors=[f"{e.code}: {e.raw}" for e in result.citation_errors],
                grades=grades,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                behavior_matched=behavior_matches(task, result),
            )
        )

    return report


def write_report(report: SuiteReport, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.replace(":", "").replace("-", "")
    path = runs_dir / f"{report.suite}-{stamp}.json"
    path.write_text(json.dumps(report.to_json(), indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path
