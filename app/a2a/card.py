"""The Agent Card: what a remote orchestrator learns before it sends a task.

Written against a2a-sdk 1.3 (A2A spec 1.0): protobuf types, `supported_interfaces`
instead of `url`, examples on the skill rather than the card.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

SKILL_ID = "contract_qa"
BEARER = "bearer"
TEXT, JSON = "text/plain", "application/json"


def package_version() -> str:
    try:
        return version("eval-driven-contract-research-agent")
    except PackageNotFoundError:  # running from a source tree that was never installed
        return "0.0.0"


def build_card(base_url: str) -> AgentCard:
    """Build the public card. `base_url` is where the JSON-RPC endpoint is served."""
    skill = AgentSkill(
        id=SKILL_ID,
        name="Contract question answering",
        description=(
            "Answer one question about named contracts in the indexed corpus. Returns an "
            "answer with verified clause citations, or an explicit abstention when the "
            "evidence does not support one. Without a document scope the agent asks which "
            "contract (input-required) instead of guessing."
        ),
        tags=["contract", "legal", "qa", "citations"],
        examples=[
            "What is the notice period for termination for convenience?",
            "Which law governs this agreement?",
        ],
        input_modes=[TEXT, JSON],
        output_modes=[TEXT, JSON],
    )
    return AgentCard(
        name="Contract Research Agent",
        description=(
            "Answers questions about commercial contracts in a fixed public corpus "
            "(CUAD/ContractNLI plus labelled synthetic variants) with clause citations, and "
            "abstains when evidence is insufficient. Not legal advice."
        ),
        version=package_version(),
        supported_interfaces=[
            AgentInterface(
                url=base_url,
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=[TEXT, JSON],
        default_output_modes=[TEXT, JSON],
        security_schemes={
            BEARER: SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="Bearer", description="Static token from CRA_A2A_TOKEN."
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={BEARER: StringList()})],
        skills=[skill],
    )
