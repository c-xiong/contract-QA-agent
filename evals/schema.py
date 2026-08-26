"""Eval task schema. See docs/SPEC.md section 15.2.

This module defines the *shape* of a task. It contains no task content, and it must
not: a dataset authored by the model family under evaluation is circular and voids
every number in the README. See .claude/rules/evals.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceSource = Literal["cuad", "contractnli", "manual", "synthetic"]
ExpectedBehavior = Literal["answer", "partial", "abstain"]

# A task whose question still reads as the template placeholder is not a task. The
# loader rejects it, so an unfilled scaffold cannot silently score as a pass.
PLACEHOLDER_MARKERS = ("TODO(author)", "<REPLACE", "FIXME")


class ExpectedEvidence(BaseModel):
    """One span an ideal answer should rest on.

    ``source`` records the provenance of the *label*, not of the document. It is what
    lets the README report how much of the ground truth was expert-annotated versus
    author-written, which is the difference between a credible evaluation and an
    assertion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=r"^doc-\d{3,}$")
    section_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    span_text: str | None = None
    source: EvidenceSource

    @model_validator(mode="after")
    def has_some_locator(self) -> ExpectedEvidence:
        if self.section_id is None and self.page_number is None and not self.span_text:
            raise ValueError(
                "ExpectedEvidence needs at least one of section_id, page_number, or span_text; "
                "a document_id alone cannot be graded at evidence level."
            )
        return self


class EvalTask(BaseModel):
    """One graded question.

    ``expected_behavior`` is a three-way choice rather than a boolean, because SPEC
    13.4 requires the system to be able to return only the supported portion of an
    answer. A ``should_abstain: bool`` could not express "partial", so it could not
    grade the behavior the spec requires.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    allowed_document_ids: list[str] | None = None
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    required_points: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_behavior: ExpectedBehavior
    partial_coverage_note: str | None = None
    touches_synthetic: bool = False
    dataset_version: str = Field(min_length=1)

    # Provenance of the QUESTION, recorded per task rather than in a footnote.
    #
    # An author who is not a native English speaker can write the question in their own
    # language and have it translated. What rule 1 protects against is the model
    # deciding WHAT to ask -- the angle, the specificity, which fact to probe. A literal
    # translation does not decide any of that, so the task stays the author's.
    #
    # It is not free of risk, and that is why it is recorded: retrieval matches on
    # words, so whoever picks the English vocabulary influences what BM25 can find. For
    # `query_refinement`, where not sharing vocabulary with the clause IS the test, a
    # translated question is materially weaker evidence and must be reported separately.
    question_source: str | None = Field(
        default=None,
        description="The question as the author first wrote it, if not in English.",
    )
    question_source_lang: str | None = Field(
        default=None,
        description="Language tag of question_source, e.g. 'zh'. None means the author "
        "wrote the question directly in English.",
    )

    @model_validator(mode="after")
    def consistent_expectations(self) -> EvalTask:
        if self.expected_behavior == "abstain":
            if self.expected_document_ids or self.expected_evidence:
                raise ValueError(
                    f"{self.task_id}: an abstain task must not carry expected evidence. "
                    "If evidence exists, the correct behavior is 'answer' or 'partial'."
                )
        elif not self.expected_document_ids:
            raise ValueError(
                f"{self.task_id}: expected_behavior={self.expected_behavior!r} requires at "
                "least one expected_document_id, or the retrieval graders have nothing to score."
            )

        evidence_docs = {e.document_id for e in self.expected_evidence}
        if stray := evidence_docs - set(self.expected_document_ids):
            raise ValueError(
                f"{self.task_id}: expected_evidence references documents absent from "
                f"expected_document_ids: {sorted(stray)}"
            )

        if self.allowed_document_ids is not None and (
            missing := set(self.expected_document_ids) - set(self.allowed_document_ids)
        ):
            raise ValueError(
                f"{self.task_id}: expected documents {sorted(missing)} are excluded by "
                "allowed_document_ids, so this task is unpassable by construction."
            )
        return self

    @model_validator(mode="after")
    def not_a_placeholder(self) -> EvalTask:
        """Refuse scaffolding that was never filled in.

        Without this, an unedited template loads, runs, and produces a number. A number
        derived from a placeholder is worse than no number, because it looks like data.
        """
        for field_name in ("question", "category"):
            value = getattr(self, field_name)
            for marker in PLACEHOLDER_MARKERS:
                if marker in value:
                    raise ValueError(
                        f"{self.task_id}: {field_name} still contains the placeholder "
                        f"{marker!r}. Eval content is author-written; see "
                        f"evals/datasets/smoke/README.md."
                    )
        return self
