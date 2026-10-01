"""Drive the MCP server over stdio as an external client, against the local corpus.

Launches `python -m app.mcp_server` as a subprocess, exactly as an MCP host would, and
invokes all five contract tools plus session_info. It then shows what the server
refuses and what it cannot enforce. No model is called and nothing leaves the machine.

    CRA_LIVE_MODEL=0 uv run python scripts/mcp_demo.py --doc doc-001

Printed source text is truncated. The corpus is public CUAD data.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

REPO = Path(__file__).resolve().parents[1]


def show(title: str, result: Any) -> dict[str, Any]:
    content = result.structured_content or {}
    printable = json.loads(json.dumps(content))
    for item in printable.get("evidence", []):
        item["text"] = item["text"][:120] + ("…" if len(item["text"]) > 120 else "")
    print(f"\n## {title}  (is_error={result.is_error})")
    print(json.dumps(printable, indent=2, ensure_ascii=False)[:2500])
    return content


def cite(item: dict[str, Any]) -> str:
    parts = [item["doc_id"]]
    if item["page"] is not None:
        parts.append(f"p. {item['page']}")
    if item["section"]:
        parts.append(f"§{item['section']}")
    return "[" + ", ".join(parts) + "]"


async def main(doc: str, term: str, query: str) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server", "--allow", doc],
        cwd=str(REPO),
        env={"CRA_LIVE_MODEL": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    async with Client(server) as client:
        print(
            f"mcp SDK {importlib.metadata.version('mcp')}; negotiated protocol {client.protocol_version}"
        )
        tools = (await client.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))

        info = show("session_info", await client.call_tool("session_info", {}))
        version = info["documents"][0]["version_id"]

        found = show("document_search", await client.call_tool("document_search", {"query": query}))
        evidence = found.get("evidence", [])
        if not evidence:
            print("No evidence found; try another --query.")
            return
        first = evidence[0]

        if first["section"]:
            show(
                "retrieve_clause",
                await client.call_tool(
                    "retrieve_clause",
                    {"doc_id": doc, "version_id": version, "clause_ref": first["section"]},
                ),
            )
        show(
            "extract_definition",
            await client.call_tool(
                "extract_definition", {"doc_id": doc, "version_id": version, "term": term}
            ),
        )
        show(
            "calculate_date (linked to admitted evidence)",
            await client.call_tool(
                "calculate_date",
                {
                    "anchor_date": "2026-01-01",
                    "offset": 30,
                    "direction": "after",
                    "convention": "calendar_days",
                    "rule_evidence_ids": [first["evidence_id"]],
                },
            ),
        )
        show(
            "verify_citation (valid locator)",
            await client.call_tool(
                "verify_citation",
                {"answer": f"Example claim. {cite(first)}", "evidence_ids": [first["evidence_id"]]},
            ),
        )

        print("\n# What the server refuses")
        show(
            "scope widening via doc_ids",
            await client.call_tool("document_search", {"query": query, "doc_ids": ["doc-999"]}),
        )
        show(
            "fabricated evidence id",
            await client.call_tool(
                "verify_citation", {"answer": "x [doc-001, p. 1]", "evidence_ids": ["e-invented"]}
            ),
        )
        show(
            "business-day convention",
            await client.call_tool(
                "calculate_date",
                {
                    "anchor_date": "2026-01-01",
                    "offset": 5,
                    "direction": "after",
                    "convention": "business_days",
                    "rule_evidence_ids": [first["evidence_id"]],
                },
            ),
        )

        print("\n# What the server cannot enforce")
        print(
            "This client already holds admitted evidence text. It could now write any answer\n"
            "and never call verify_citation; the server would not see that prose. And even a\n"
            "structurally valid citation says nothing about whether the text supports a claim."
        )
        show("final session usage", await client.call_tool("session_info", {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doc", default="doc-001")
    parser.add_argument("--term", default="Affiliate")
    parser.add_argument("--query", default="termination notice period")
    args = parser.parse_args()
    asyncio.run(main(args.doc, args.term, args.query))
