"""The default suite fails on external I/O while retaining local async machinery."""

import asyncio
import socket

import pytest

from app.retrieval.embeddings import Embedder


def test_external_connections_are_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(AssertionError, match="Network/model access is disabled"),
    ):
        sock.connect(("203.0.113.1", 443))


def test_dns_is_blocked() -> None:
    with pytest.raises(AssertionError, match="Network/model access is disabled"):
        socket.getaddrinfo("example.com", 443)


def test_model_loading_is_blocked_even_with_a_warm_cache() -> None:
    with pytest.raises(AssertionError, match="Network/model access is disabled"):
        Embedder()


async def test_local_socketpair_still_works() -> None:
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.setblocking(False)
        writer.setblocking(False)
        loop = asyncio.get_running_loop()
        await loop.sock_sendall(writer, b"local")
        assert await loop.sock_recv(reader, 5) == b"local"
