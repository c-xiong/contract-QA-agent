"""Eval-driven contract research agent.

Layer map (see docs/decisions.md):

- ``app.schemas``    typed data contracts shared by every layer
- ``app.ingestion``  PDF parsing, chunking, and the chunk/document store
- ``app.retrieval``  retrievers behind one Protocol; the agent never names a concrete one
- ``app.agent``      LangGraph workflow, typed tools, model adapter
- ``app.evidence``   citation parsing and verification
"""
