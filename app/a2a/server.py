"""A2A server: Agent Card + JSON-RPC (with SSE streaming) over the V2 research agent.

DECISION: a separate process and port, not routes on the inspector app (app/api/main.py).
  The inspector is read-only and unauthenticated; this surface takes delegated tasks and
  requires a token. Keeping them apart keeps each one's trust boundary obvious, and
  mirrors the MCP server, which is its own process too. Corpus loading and degradation
  reuse `build_agent()`. Alternative rejected: `add_a2a_routes_to_fastapi` on the
  inspector app -- one process fewer, two auth models in one app.

DECISION: V2 only. Run journals, telemetry export, scope/version checks and cancellation
  journaling exist only in V2, and an A2A task must map to a trace.

Tasks live in the SDK's `InMemoryTaskStore`: a restart drops in-flight and input-required
tasks. No push notifications, no gRPC/REST bindings.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette
from starlette.middleware import Middleware

from app.a2a.auth import BearerAuth
from app.a2a.card import build_card
from app.a2a.executor import ContractQAExecutor
from app.agent.runner import ResearchAgent
from app.config import get_settings

logger = logging.getLogger(__name__)

RPC_PATH = "/"


def build_app(agent: ResearchAgent, *, token: str, base_url: str) -> Starlette:
    card = build_card(base_url)
    handler = DefaultRequestHandler(
        agent_executor=ContractQAExecutor(agent),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, rpc_url=RPC_PATH)]
    return Starlette(routes=routes, middleware=[Middleware(BearerAuth, token=token)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.a2a", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--arm", default="bm25", help="retrieval arm (see /arms)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import uvicorn

    # Imported here: app.api.main builds the inspector app object at import time.
    from app.api.main import build_agent

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if settings.a2a_token is None:
        print("CRA_A2A_TOKEN is not set; refusing to serve without auth.", file=sys.stderr)
        return 2
    token = settings.a2a_token.get_secret_value()
    settings = settings.model_copy(update={"agent_mode": "v2"})
    agent, note = build_agent(settings, arm=args.arm)
    if note:
        logger.warning(note)
    base_url = f"http://{args.host}:{args.port}{RPC_PATH}"
    app = build_app(agent, token=token, base_url=base_url)
    logger.info(
        "A2A agent on %s (card at /.well-known/agent-card.json, model=%s, live=%s)",
        base_url,
        agent.client.model_id,
        agent.settings.live_model,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
