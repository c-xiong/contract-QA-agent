"""Bearer-token check for every A2A route except the public Agent Card.

DECISION: one static shared token, compared in constant time, fail closed.
  It authenticates a caller; it is not per-user authorization or tenant isolation
  (out of scope, CLAUDE.md rule 6). A production deployment would declare OAuth2 client
  credentials or mTLS on the card instead. Alternative rejected: no auth on a localhost
  demo, which would make the card's security declaration untrue.
"""

from __future__ import annotations

import hmac

from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.types import ASGIApp, Receive, Scope, Send

PUBLIC_PATHS = frozenset({AGENT_CARD_WELL_KNOWN_PATH})


class BearerAuth:
    """Pure ASGI middleware, so SSE responses stream through untouched."""

    def __init__(self, app: ASGIApp, token: str):
        if not token:
            raise ValueError("BearerAuth needs a non-empty token")
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        supplied = dict(scope["headers"]).get(b"authorization", b"")
        if hmac.compare_digest(supplied, self.expected):
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", b"0"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})
