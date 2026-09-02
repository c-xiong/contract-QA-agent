"""Prompts for the research agent.

Note what is NOT here. The citation format is described so the model can produce it,
but nothing in this file is load-bearing for correctness: an answer that ignores
every instruction below is caught by `app.evidence.citation_verifier`, which is
deterministic code with tests. CLAUDE.md rule 5 -- if a prompt instruction were the
only thing enforcing citation validity, that would be a bug, not a design.
"""

from __future__ import annotations

from app.schemas.evidence import Evidence

WRITER_SYSTEM = """You are a contract research assistant. You answer questions about \
commercial contracts using only the evidence excerpts provided to you.

Rules:
- Use only the provided excerpts. Do not use outside knowledge about these contracts.
- Cite every factual claim with the internal citation format: [doc-014, p. 12, §8.1]
  Use the document_id, page, and section exactly as given in the excerpt header.
  Omit the section if the excerpt header has none.
- Put exactly one locator inside each pair of brackets. When one claim needs multiple
  sources, write separate citations: [doc-014, p. 12, §8.1] [doc-035, p. 11, §9.1]
  Never combine them as [doc-014, p. 12, §8.1; doc-035, p. 11, §9.1].
- If the excerpts do not answer the question, say so explicitly and state what you
  did find. Do not guess, and do not pad an answer to look complete.
- Quote or closely paraphrase the contract language rather than summarizing loosely.
  The precise wording is usually the point.
- Be brief. Two or three sentences is usually right."""


def format_evidence(evidence: list[Evidence]) -> str:
    """Render evidence into the writer's context, headers first.

    Each excerpt is preceded by its provenance in exactly the citation format the
    model is asked to reproduce. Making the model transcribe a string it can see
    beats asking it to assemble one from fields.
    """
    if not evidence:
        return "(no evidence retrieved)"

    blocks: list[str] = []
    for item in evidence:
        locator = [item.document_id, f"p. {item.page_number}"]
        if item.section_id:
            locator.append(f"§{item.section_id}")
        header = f"[{', '.join(locator)}]"
        title = item.section_title or item.document_title
        blocks.append(f"{header} {title}\n{item.excerpt}")
    return "\n\n---\n\n".join(blocks)


def build_writer_prompt(question: str, evidence: list[Evidence]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence excerpts:\n\n{format_evidence(evidence)}\n\n"
        f"Answer the question using only these excerpts, citing each claim."
    )


REPAIR_SYSTEM = (
    WRITER_SYSTEM
    + """

Your previous answer contained citations that failed verification. Rewrite it using \
only citations that appear verbatim in the evidence headers above. Keep multiple \
locators in separate pairs of brackets; never join them with a semicolon. If a claim cannot \
be supported by an available excerpt, remove the claim."""
)


def build_repair_prompt(question: str, evidence: list[Evidence], errors: list[str]) -> str:
    problems = "\n".join(f"- {e}" for e in errors)
    return (
        f"{build_writer_prompt(question, evidence)}\n\n"
        f"Your previous answer had these citation problems:\n{problems}"
    )
