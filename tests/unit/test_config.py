"""Scaffold checks for app.config.

These cover the configuration contract itself -- precedence, validation bounds,
and derived paths -- not application behavior, of which there is none yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_defaults_do_not_require_an_environment() -> None:
    settings = Settings(_env_file=None)
    assert settings.model_id == "claude-sonnet-5"
    assert settings.retrieval_top_k == 5
    assert settings.anthropic_api_key is None


def test_live_model_defaults_off() -> None:
    """Nothing spends money unless it is asked to explicitly."""
    assert Settings(_env_file=None).live_model is False


def test_environment_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRA_RETRIEVAL_TOP_K", "9")
    monkeypatch.setenv("CRA_LIVE_MODEL", "1")
    settings = Settings(_env_file=None)
    assert settings.retrieval_top_k == 9
    assert settings.live_model is True


def test_api_key_reads_the_conventional_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK and this file must read the same variable, not two that drift."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-value"


def test_api_key_is_not_exposed_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
    assert "sk-test-value" not in repr(Settings(_env_file=None))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrieval_top_k", 0),  # below SearchContractsInput's ge=1
        ("retrieval_top_k", 11),  # above SearchContractsInput's le=10
        ("model_timeout_s", 0.0),
        ("chunk_max_tokens", 0),
        ("chunk_overlap_tokens", -1),
        ("chars_per_token", 0.0),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_unknown_settings_are_rejected() -> None:
    """A typo in a variable name must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_topk=5)  # type: ignore[call-arg]


def test_derived_paths_hang_off_data_dir() -> None:
    settings = Settings(_env_file=None, data_dir=Path("/tmp/corpus"))
    assert settings.raw_dir == Path("/tmp/corpus/raw")
    assert settings.processed_dir == Path("/tmp/corpus/processed")
    assert settings.index_dir == Path("/tmp/corpus/index")


def test_settings_are_frozen() -> None:
    """Configuration cannot drift mid-run."""
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.retrieval_top_k = 7


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
