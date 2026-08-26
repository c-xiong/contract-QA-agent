"""Adapter for the ContractNLI release.

607 NDAs, each annotated against the same 17 hypotheses as Entailment, Contradiction,
or NotMentioned, with evidence spans for the first two. CC BY 4.0.

**The one integration bug worth guarding against.** A document carries a `spans` list of
`[start, end]` character offsets into its `text`. An annotation also carries a field
called `spans` -- but those are **indices into that list**, not character offsets:

    doc["spans"]            == [[0, 44], [45, 132], [133, 331], ...]
    annotation["spans"]     == [39, 40]        <- means doc["spans"][39] and [40]

Reading annotation spans as character offsets produces plausible-looking garbage: valid
substrings of the document, at the wrong place, with no error. SPEC 6.3 requires a
round-trip test proving one known annotation resolves to its literal text, and
`tests/unit/test_contractnli.py` is that test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

NliChoice = Literal["Entailment", "Contradiction", "NotMentioned"]

SPLITS = ("train.json", "dev.json", "test.json")


class ContractNliFormatError(RuntimeError):
    """The release on disk does not have the structure this adapter expects."""


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One of the 17 fixed hypotheses, phrased declaratively."""

    key: str
    short_description: str
    hypothesis: str


@dataclass(frozen=True, slots=True)
class NliAnnotation:
    """One hypothesis's label for one document, with its evidence resolved to text."""

    hypothesis_key: str
    choice: NliChoice
    span_indices: list[int]
    span_texts: list[str]
    span_offsets: list[tuple[int, int]]

    @property
    def has_evidence(self) -> bool:
        return bool(self.span_texts)


@dataclass(frozen=True, slots=True)
class NliDocument:
    """One NDA with its text, span table, and resolved annotations."""

    doc_id: int
    file_name: str
    text: str
    spans: list[tuple[int, int]]
    annotations: dict[str, NliAnnotation]
    url: str | None = None

    def resolve_span(self, index: int) -> str:
        """Return the literal text of `doc.spans[index]`.

        This is the function the round-trip test exercises. Every use of an annotation's
        span list must go through here rather than slicing `text` directly.
        """
        if not 0 <= index < len(self.spans):
            raise ContractNliFormatError(
                f"{self.file_name}: span index {index} out of range (0..{len(self.spans) - 1}). "
                "Annotation spans are indices into document.spans, not character offsets."
            )
        start, end = self.spans[index]
        return self.text[start:end]

    @property
    def annotated_count(self) -> int:
        return sum(1 for a in self.annotations.values() if a.choice != "NotMentioned")


def load_split(path: Path) -> tuple[dict[str, Hypothesis], list[NliDocument]]:
    """Load one ContractNLI split file."""
    if not path.exists():
        raise ContractNliFormatError(
            f"No ContractNLI split at {path}. Run: uv run python scripts/download_contractnli.py"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractNliFormatError(f"{path} is not valid JSON: {exc}") from exc

    if "documents" not in payload or "labels" not in payload:
        raise ContractNliFormatError(
            f"{path} lacks 'documents' and/or 'labels'. The release layout changed."
        )

    hypotheses = {
        key: Hypothesis(
            key=key,
            short_description=value.get("short_description", ""),
            hypothesis=value.get("hypothesis", ""),
        )
        for key, value in payload["labels"].items()
    }

    return hypotheses, list(_parse_documents(payload["documents"], path))


def _parse_documents(raw_documents: list[dict[str, Any]], path: Path) -> Iterator[NliDocument]:
    for raw in raw_documents:
        text = str(raw.get("text") or "")
        spans = [(int(s), int(e)) for s, e in raw.get("spans", [])]

        annotation_sets = raw.get("annotation_sets") or []
        if not annotation_sets:
            continue
        raw_annotations = annotation_sets[0].get("annotations", {})

        annotations: dict[str, NliAnnotation] = {}
        for key, value in raw_annotations.items():
            choice = value.get("choice")
            if choice not in ("Entailment", "Contradiction", "NotMentioned"):
                raise ContractNliFormatError(
                    f"{path}: unexpected choice {choice!r} for {key} in {raw.get('file_name')}"
                )
            indices = [int(i) for i in value.get("spans", [])]

            texts: list[str] = []
            offsets: list[tuple[int, int]] = []
            for index in indices:
                if not 0 <= index < len(spans):
                    raise ContractNliFormatError(
                        f"{path}: {raw.get('file_name')} annotation {key} references span "
                        f"index {index}, but the document has {len(spans)} spans."
                    )
                start, end = spans[index]
                texts.append(text[start:end])
                offsets.append((start, end))

            annotations[key] = NliAnnotation(
                hypothesis_key=key,
                choice=choice,
                span_indices=indices,
                span_texts=texts,
                span_offsets=offsets,
            )

        yield NliDocument(
            doc_id=int(raw.get("id", -1)),
            file_name=str(raw.get("file_name") or ""),
            text=text,
            spans=spans,
            annotations=annotations,
            url=str(raw["url"]) if raw.get("url") else None,
        )


def load_corpus(root: Path) -> tuple[dict[str, Hypothesis], list[NliDocument]]:
    """Load every split, concatenated. Splits are a modelling artifact, not ours.

    ContractNLI's train/dev/test split exists for training NLI models. This project does
    not train anything; it uses the labels as ground truth. Concatenating and then taking
    a deterministic sample is the honest thing to do -- pretending to respect a split we
    do not train on would be theatre.
    """
    hypotheses: dict[str, Hypothesis] = {}
    documents: list[NliDocument] = []
    for name in SPLITS:
        split_hypotheses, split_documents = load_split(root / name)
        hypotheses.update(split_hypotheses)
        documents.extend(split_documents)
    if not documents:
        raise ContractNliFormatError(f"No documents found under {root}")
    return hypotheses, documents
