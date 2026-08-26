"""Cross-reference resolution. See docs/SPEC.md section 9.5.

DECISION-BEARING MODULE. See docs/review-questions.md.

**The problem, concretely.** A liability cap in Section 8.1 is qualified by carve-outs in
Section 8.3. Retrieval returns 8.1, because "liability cap" matches 8.1 both lexically
and semantically. The generated answer states a cap, cites a real document and a real
page, and passes every deterministic citation check in this repository. It is still
wrong, because the carve-out was never retrieved.

No citation verifier catches this. The citation is valid. Only the evidence set is
incomplete, and incompleteness has no signature at the citation layer.

**The fix.** After evidence collection and before the writer, follow each selected
chunk's `outbound_references` -- populated by the chunker from phrases like "subject to
Section 8.3" -- and pull those sections into the evidence set. Everything pulled is
marked `pulled_by="cross_reference"` so its contribution is measurable and can be
ablated (SPEC 16.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.store import ChunkStore
from app.schemas.chunk import Chunk
from app.schemas.retrieval import RetrievedChunk


@dataclass(frozen=True, slots=True)
class ResolutionReport:
    """What was pulled, and why. Feeds the trace and the ablation experiment."""

    pulled: list[RetrievedChunk]
    followed: list[tuple[str, str]]
    skipped_budget: int
    skipped_unresolved: list[tuple[str, str]]

    @property
    def count(self) -> int:
        return len(self.pulled)

    def summary(self) -> str:
        if not self.followed and not self.skipped_unresolved:
            return "no outbound references to follow"
        parts = [f"pulled {len(self.pulled)} chunk(s) from {len(self.followed)} reference(s)"]
        if self.skipped_unresolved:
            parts.append(f"{len(self.skipped_unresolved)} reference(s) unresolved")
        if self.skipped_budget:
            parts.append(f"{self.skipped_budget} dropped for budget")
        return "; ".join(parts)


def resolve_cross_references(
    retrieved: list[RetrievedChunk],
    store: ChunkStore,
    *,
    max_depth: int = 1,
    max_chunks: int = 4,
) -> ResolutionReport:
    """Pull referenced sections into the evidence set.

    DECISION: depth 1 by default.
      Contract cross-references chain -- 8.1 points at 8.3, which points at Article 12,
      which points at the definitions. Following them transitively pulls in a large
      fraction of the contract and drowns the clause that was actually asked about.
      Depth 1 captures the case that motivates the feature (a cap and its immediate
      carve-out) at bounded cost. The parameter exists so depth 2 can be measured rather
      than argued about.

    DECISION: references resolve only WITHIN the citing document.
      "Section 8.3" means section 8.3 of this contract. Resolving it corpus-wide would
      match section 8.3 of forty other contracts, and pulling those in would be a
      provenance error of exactly the kind this project exists to prevent -- evidence
      attributed to a document that never said it.

    DECISION: a reference the chunker cannot resolve is recorded, not silently dropped.
      An unresolved reference means our section detection missed a section that the
      document itself cites. That is a chunker-quality signal, and it belongs in the
      trace where it can be counted, not discarded.

    DECISION: exact section matches are pulled before any subsection, across all
    references, rather than resolving one reference fully before starting the next.
      `section_path` is derived by splitting on dots, so §11.7 is a child of §11 and
      "subject to Section 11" resolves to both. Resolving reference-by-reference in
      document order let a subsection consume budget that the *referent of another
      reference* needed. Observed live on doc-003: §10 says "except in the event of a
      breach of Section 11", and the four-chunk budget went to §11 (right) plus §11.7
      COUNTERPARTS (noise) before a second reference was reached.
      Two passes fix the ordering without changing what is reachable: every exact
      referent first, then descendants while budget remains. A reference is never
      crowded out by another reference's subsections.
      Rejected: pulling only exact matches. Under a cap at 8.3, the carve-outs often ARE
      8.3(a)-(c), and dropping them would defeat the component.
    """
    if max_depth < 1 or max_chunks < 1:
        return ResolutionReport([], [], 0, [])

    # Locations already in the evidence set. A reference resolving to something already
    # retrieved is not a pull -- it would duplicate the chunk and inflate the apparent
    # contribution of this component in the ablation.
    seen_chunk_ids = {item.chunk.chunk_id for item in retrieved}

    pulled: list[RetrievedChunk] = []
    followed: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    skipped_budget = 0

    frontier = list(retrieved)
    for _ in range(max_depth):
        # Collect every candidate this level reaches, tagged with how specific it is,
        # then admit them exact-first. `exact` is 0 when the chunk IS the referenced
        # section and 1 when it is a descendant of it.
        candidates: list[tuple[int, str, str, Chunk]] = []
        for item in frontier:
            document_id = item.chunk.document_id
            for reference in item.chunk.outbound_references:
                # A section referring to itself, or to its own parent, adds nothing.
                if reference in item.chunk.section_path:
                    continue

                targets = store.find_section(document_id, reference)
                if not targets:
                    unresolved.append((document_id, reference))
                    continue
                for target in targets:
                    exact = 0 if target.section_id == reference else 1
                    candidates.append((exact, item.chunk.chunk_id, reference, target))

        next_frontier: list[RetrievedChunk] = []
        for _, source_id, reference, target in sorted(
            candidates, key=lambda c: (c[0], c[3].chunk_id)
        ):
            if target.chunk_id in seen_chunk_ids:
                continue
            if len(pulled) >= max_chunks:
                skipped_budget += 1
                continue

            seen_chunk_ids.add(target.chunk_id)
            resolved = RetrievedChunk(
                chunk=target,
                bm25_rank=None,
                dense_rank=None,
                # Score 0.0 is honest: this chunk was not ranked by any retriever, it
                # was pulled structurally. A synthetic score would make it sortable
                # against real ones and hide where it came from.
                fused_score=0.0,
                rerank_score=None,
                retrieval_query=f"cross-reference from {source_id} -> §{reference}",
                pulled_by="cross_reference",
            )
            pulled.append(resolved)
            next_frontier.append(resolved)
            followed.append((source_id, reference))

        frontier = next_frontier
        if not frontier:
            break

    return ResolutionReport(
        pulled=pulled,
        followed=followed,
        skipped_budget=skipped_budget,
        skipped_unresolved=unresolved,
    )
