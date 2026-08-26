"""Ask the agent one question and print the cited answer.

uv run python scripts/demo.py "What is the governing law in doc-001?"
uv run python scripts/demo.py --trace "..."
uv run python scripts/demo.py --docs doc-001,doc-002 "..."
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.agent.runner import load_agent
from app.config import get_settings
from app.ingestion.store import StoreError


async def run(question: str, *, docs: list[str] | None, show_trace: bool) -> int:
    settings = get_settings()
    try:
        agent = load_agent(settings)
    except StoreError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"corpus   : {len(agent.store.documents)} documents, {len(agent.store)} chunks")
    print(f"retriever: {agent.retriever.name}")
    print(f"model    : {agent.client.model_id}")
    if docs:
        print(f"allowlist: {', '.join(docs)}")
    print(f"\nQ: {question}\n")

    result = await agent.research(question, allowed_document_ids=docs)

    print("-" * 72)
    print(result.answer)
    print("-" * 72)

    print(f"\nstatus    : {result.status}")
    print(f"retrieved : {len(result.retrieved)} chunks from {result.retrieved_document_ids}")
    print(f"citations : {len(result.citations)} verified")
    for citation in result.citations:
        print(f"    {citation.render()}")
    if result.citation_errors:
        print(f"citation errors: {len(result.citation_errors)}")
        for error in result.citation_errors:
            print(f"    [{error.code}] {error.raw}: {error.detail}")
    print(f"tokens    : {result.input_tokens} in, {result.output_tokens} out")

    if show_trace:
        print("\ntrace:")
        for event in result.trace:
            print(f"    {event.step:16s} {event.detail}")

    return 0 if result.status in {"completed", "abstained"} else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--docs", default=None, help="comma-separated document allowlist")
    parser.add_argument("--trace", action="store_true", help="print the execution trace")
    args = parser.parse_args()

    docs = [d.strip() for d in args.docs.split(",")] if args.docs else None
    return asyncio.run(run(args.question, docs=docs, show_trace=args.trace))


if __name__ == "__main__":
    raise SystemExit(main())
