"""Claude Code CLI backend: isolation flags, parsing, limits, and resumable stops."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from app.agent.llm import ClaudeCodeClient, ModelError, ProviderLimitReached, build_client
from app.config import Settings
from evals.budget import Ledger
from tests.unit.test_v2_eval_pipeline import corpus, preflight, runner_args

MODEL = "claude-sonnet-5"


def result(text: str = "OK", **overrides: Any) -> str:
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "result": text,
        "usage": {
            "input_tokens": 700,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 5,
            "output_tokens": 40,
        },
        "modelUsage": {MODEL: {}},
    } | overrides
    return json.dumps(payload)


def fake_cli(tmp_path: Path, body: str) -> Path:
    """An executable standing in for `claude`; it records its argv, env and stdin."""
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{tmp_path}/argv"\n'
        f'env > "{tmp_path}/env"\n'
        f'cat > "{tmp_path}/stdin"\n'
        f"{body}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def settings(path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "live_model": True,
        "model_backend": "claude_code",
        "model_id": MODEL,
        "claude_code_path": str(path),
    }
    return Settings().model_copy(update=values | overrides)


async def test_cli_call_is_isolated_and_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-the-cli")
    path = fake_cli(tmp_path, f"echo '{result('Answer [doc-001]')}'")
    client = build_client(settings(path))
    assert isinstance(client, ClaudeCodeClient)
    response = await client.complete("SYSTEM PROMPT", "user question")
    assert response.text == "Answer [doc-001]"
    assert (response.input_tokens, response.output_tokens) == (725, 40)
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[argv.index("--system-prompt") + 1] == "SYSTEM PROMPT"
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--effort") + 1] == "high"
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert (tmp_path / "stdin").read_text() == "user question"
    assert "ANTHROPIC_API_KEY" not in (tmp_path / "env").read_text()
    assert "sk-should-not" not in (tmp_path / "env").read_text()
    assert os.listdir(client._workdir) == []  # empty working directory, no project files


@pytest.mark.parametrize(
    "stdout,error,message",
    [
        (result(modelUsage={"claude-opus-5-5": {}}), ModelError, "served"),
        (result(""), ModelError, "no text"),
        ("not json", ModelError, "no result"),
        (
            result("Claude AI usage limit reached", is_error=True, subtype="error"),
            ProviderLimitReached,
            "limit",
        ),
        (result("busy", is_error=True, api_error_status=429), ProviderLimitReached, "limit"),
        (result("bad request", is_error=True, api_error_status=400), ModelError, "status 400"),
    ],
)
def test_parse_classifies_failures(
    tmp_path: Path, stdout: str, error: type[Exception], message: str
) -> None:
    client = ClaudeCodeClient(settings(fake_cli(tmp_path, "true")))
    with pytest.raises(error, match=message):
        client.parse(stdout, "")
    assert client.limit_reached is (error is ProviderLimitReached)


async def test_timeout_kills_the_process(tmp_path: Path) -> None:
    client = ClaudeCodeClient(settings(fake_cli(tmp_path, "sleep 5"), claude_code_timeout_s=0.2))
    with pytest.raises(ModelError, match="exceeded"):
        await client.complete("s", "u")


def test_missing_cli_is_a_classified_error() -> None:
    with pytest.raises(ModelError, match="not on PATH"):
        ClaudeCodeClient(settings(Path("/nonexistent/claude")))


def test_preamble_overhead_is_reserved(tmp_path: Path) -> None:
    ledger = Ledger(preflight(per_call_input_overhead_tokens=800), tmp_path / "ledger.json")
    input_estimate, _, _ = ledger.reserve(prompt_chars=100)
    assert input_estimate == 50 + 800


async def test_provider_limit_stops_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A limit stops the run without scoring the case; --resume re-runs it."""
    from experiments import v2_standalone

    corpus(tmp_path, monkeypatch)
    limited = tmp_path / "limited"
    limited.write_text("")
    body = (
        f'if [ -f "{limited}" ]; then '
        f"echo '{result('Claude AI usage limit reached', is_error=True, subtype='error')}'; "
        f"else echo '{result('No answer.')}'; fi"
    )
    cli = fake_cli(tmp_path, body)
    monkeypatch.setenv("CRA_LIVE_MODEL", "1")
    monkeypatch.setenv("CRA_MODEL_BACKEND", "claude_code")
    monkeypatch.setenv("CRA_MODEL_ID", MODEL)
    monkeypatch.setenv("CRA_CLAUDE_CODE_PATH", str(cli))
    flight = tmp_path / "preflight.json"
    flight.write_text(
        preflight(
            model_id=MODEL,
            max_attempts=100,
            max_output_tokens_per_call=32_000,
            max_total_output_tokens=10_000_000,
            max_cost=1_000.0,
            per_call_input_overhead_tokens=1_000,
        ).model_dump_json()
    )
    output = tmp_path / "out" / "r.json"
    args = runner_args(output, preflight=flight)
    await v2_standalone.run(args)
    report = json.loads(output.read_text())
    assert [r["execution"] for r in report["rows"]] == ["provider_limit_stopped"]
    assert {u["reason"] for u in report["unexecuted"]} == {"provider_limit"}
    assert report["summary"]["all"]["n"] == 0  # the interrupted case is not scored
    assert report["provenance"]["frozen_config"]["model_backend"] == "claude_code"

    limited.unlink()
    args.resume = True
    await v2_standalone.run(args)
    report = json.loads(output.read_text())
    assert report["unexecuted"] == []
    assert {r["execution"] for r in report["rows"]} == {"completed"}
    assert report["summary"]["all"]["n"] == 2


async def test_judge_resumes_after_provider_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from experiments import v2_judge

    from evals.loader import load_suite

    task = load_suite("hand")[0]
    report = tmp_path / "r.json"
    report.write_text(
        json.dumps(
            {
                "provenance": {"frozen_config": {"suite": "hand"}},
                "rows": [
                    {
                        "task_id": task.task_id,
                        "split": "calibration",
                        "execution": "completed",
                        "result": {"status": "abstained", "answer": "No answer.", "evidence": []},
                    }
                ],
            }
        )
    )
    limited = tmp_path / "limited"
    limited.write_text("")
    body = (
        f'if [ -f "{limited}" ]; then '
        f"echo '{result('Claude AI usage limit reached', is_error=True, subtype='error')}'; "
        f"else echo '{result('not json')}'; fi"
    )
    monkeypatch.setenv("CRA_LIVE_MODEL", "1")
    monkeypatch.setenv("CRA_MODEL_BACKEND", "claude_code")
    monkeypatch.setenv("CRA_MODEL_ID", MODEL)
    monkeypatch.setenv("CRA_CLAUDE_CODE_PATH", str(fake_cli(tmp_path, body)))
    flight = tmp_path / "judge.json"
    flight.write_text(
        preflight(
            phase="judge",
            model_id=MODEL,
            max_output_tokens_per_call=32_000,
            max_total_output_tokens=10_000_000,
            max_cost=1_000.0,
        ).model_dump_json()
    )
    args = argparse.Namespace(
        report=report, phase="calibration", round=1, preflight=flight, resume=False
    )
    sealed = await v2_judge.run(args)
    records = [json.loads(line) for line in sealed.read_text().splitlines()]
    assert records[1] == {"task_id": task.task_id, "not_judged": "provider_limit"}
    with pytest.raises(SystemExit, match="append-only"):
        await v2_judge.run(args)

    limited.unlink()
    args.resume = True
    await v2_judge.run(args)
    records = [json.loads(line) for line in sealed.read_text().splitlines()]
    assert sum(r.get("record") == "header" for r in records) == 1
    assert len(records) == 3  # the limit record is kept; the re-judged record is appended
    assert records[2]["answer_rubric"]["verdict"] == "unknown"  # unparseable, not a pass
