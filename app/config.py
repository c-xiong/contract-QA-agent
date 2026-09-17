"""Typed application configuration.

Every tunable the system reads at runtime is declared here as a validated field.
Nothing reads ``os.environ`` directly; a value that is not in this file is not a
configuration value, it is a hardcoded constant and should look like one.

Precedence: constructor argument > environment variable > ``.env`` file > default.
All variables use the ``CRA_`` prefix except ``ANTHROPIC_API_KEY``, which keeps its
conventional name so the Anthropic SDK and this file read the same thing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    # --- Paths ------------------------------------------------------------
    # Relative paths resolve against the process working directory, which for
    # every entry point in scripts/ is the repository root.
    data_dir: Path = Field(
        default=Path("data"),
        description="Root of all corpus data. Subdirectories are derived, not configured.",
    )

    # --- Model access -----------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CRA_ANTHROPIC_API_KEY"),
        description="Absent is legal: the deterministic stub path needs no key.",
    )
    model_id: str = Field(
        default="claude-sonnet-5",
        description="Single model provider by design. See docs/decisions.md",
    )
    model_timeout_s: float = Field(
        default=60.0,
        gt=0,
        description="Per-call wall clock ceiling. Enforced in code, not requested in a prompt.",
    )
    live_model: bool = Field(
        default=False,
        description=(
            "Gate for anything that spends money. False routes model calls to a "
            "deterministic stub. See .claude/rules/evals.md."
        ),
    )
    answerability_gate: bool = Field(
        default=False,
        description="Opt-in pre-write evidence gate. Disabled after the closeout ablation "
        "showed increased over-abstention; see results/experiment-d-answerability.json.",
    )

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Chunks returned per query. Bound matches SearchContractsInput (SPEC 12).",
    )

    # --- Chunking ---------------------------------------------------------
    # TODO(author): chunk sizing is an author-owned decision (CLAUDE.md rule 2,
    # SPEC 9.3). These defaults are placeholders so the module imports, not
    # findings. Set them from a retrieval experiment in Sprint 1 and record the
    # comparison in docs/decisions.md before treating either number as real.
    chunk_max_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=0, ge=0)

    # TODO(author): confirm the token-estimation method. Chunk.token_count and
    # the section 6.10 cost estimate both depend on it; Anthropic's count_tokens
    # endpoint is a network round trip per chunk and too slow for ingestion.
    # See ambiguity 4.5 in the Sprint 0 plan.
    chars_per_token: float = Field(
        default=4.0,
        gt=0,
        description="Heuristic divisor for token estimation. Approximate by construction.",
    )

    @field_validator("anthropic_api_key", mode="after")
    @classmethod
    def blank_key_is_no_key(cls, value: SecretStr | None) -> SecretStr | None:
        """Treat an empty or whitespace-only key as absent.

        `.env.example` ships `ANTHROPIC_API_KEY=` with no value, so the common first
        run has the variable SET and EMPTY. Without this, the key reads as present, the
        live client is built, and the failure surfaces deep inside the SDK as a
        TypeError about header resolution -- far from the one-line cause.
        """
        if value is None:
            return None
        return value if value.get_secret_value().strip() else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        """Downloaded corpora, exactly as acquired. Never written to by app code."""
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        """Parsed documents and chunks. Reproducible from raw_dir via scripts/ingest.py."""
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def index_dir(self) -> Path:
        """Retrieval indices. Reproducible from processed_dir."""
        return self.data_dir / "index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that configuration is read once and cannot drift mid-run. Tests that
    need a different configuration construct ``Settings(...)`` directly rather than
    mutating the singleton.
    """
    return Settings()
