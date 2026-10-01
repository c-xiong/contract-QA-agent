"""Reported usage can outrun character estimates; paid attempts must then stop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.agent.llm import ModelResponse
from evals.budget import BudgetedClient, BudgetExceeded, Ledger, Preflight


def preflight(**overrides: Any) -> Preflight:
    return Preflight.model_validate(
        {
            "experiment_id": "synthetic-budget-overrun",
            "phase": "application",
            "model_id": "synthetic-budget-model",
            "input_price_per_million_tokens": 3.0,
            "output_price_per_million_tokens": 15.0,
            "max_attempts": 10,
            "max_input_tokens_per_call": 50_000,
            "max_output_tokens_per_call": 2048,
            "max_total_input_tokens": 1_000_000,
            "max_total_output_tokens": 100_000,
            "max_cost": 10.0,
        }
        | overrides
    )


class ReportedUsageClient:
    model_id = "synthetic-budget-model"

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.calls = 0

    async def complete(self, system: str, user: str) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            "synthetic answer",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model_id=self.model_id,
        )


@pytest.mark.parametrize("input_tokens,output_tokens", [(120, 10), (100, 2049)])
async def test_any_usage_overrun_stops_next_and_resumed_attempts(
    tmp_path: Path, input_tokens: int, output_tokens: int
) -> None:
    declaration = preflight()
    path = tmp_path / "ledger.json"
    ledger = Ledger(declaration, path)
    inner = ReportedUsageClient(input_tokens, output_tokens)
    client = BudgetedClient(inner, ledger)

    # Reserve 100 input tokens. Both overruns still fit the cumulative budgets.
    answer = await client.complete("", "x" * 200)
    assert answer.text == "synthetic answer"
    assert ledger.exhausted and ledger.state.exhausted_reason == "over_reservation"
    assert ledger.state.over_reservation_attempts == 1
    assert ledger.state.settled_input_tokens == input_tokens
    assert ledger.state.settled_output_tokens == output_tokens
    with pytest.raises(BudgetExceeded, match="over_reservation"):
        await client.complete("", "x")

    resumed = Ledger(declaration, path)
    with pytest.raises(BudgetExceeded, match="over_reservation"):
        await BudgetedClient(inner, resumed).complete("", "x")
    assert inner.calls == 1 and resumed.state.attempts == 1


async def test_settlement_reports_every_breached_token_and_currency_cap(tmp_path: Path) -> None:
    declaration = preflight(
        max_input_tokens_per_call=100,
        max_total_input_tokens=100,
        max_total_output_tokens=2048,
        max_cost=(100 * 3 + 2048 * 15) / 1_000_000,
    )
    path = tmp_path / "ledger.json"
    ledger = Ledger(declaration, path)
    await BudgetedClient(ReportedUsageClient(150, 2049), ledger).complete("", "x" * 200)

    summary = ledger.summary()
    assert set(summary["settled_cap_breaches"]) == {
        "per_call_input_cap",
        "per_call_output_cap",
        "input_token_cap",
        "output_token_cap",
        "cost_cap",
    }
    assert summary["settled_cost"] == pytest.approx((150 * 3 + 2049 * 15) / 1_000_000)
    assert summary["remaining_cost"] < 0
    assert (
        Ledger(declaration, path).summary()["settled_cap_breaches"]
        == summary["settled_cap_breaches"]
    )


def test_historical_overrun_counter_exhausts_ledger_on_resume(tmp_path: Path) -> None:
    declaration = preflight()
    path = tmp_path / "ledger.json"
    ledger = Ledger(declaration, path)
    old_state = ledger.state.model_dump()
    old_state.pop("settled_cap_breaches")
    old_state.update(over_reservation_attempts=1, attempts=1, settled_input_tokens=120)
    path.write_text(json.dumps(old_state))

    resumed = Ledger(declaration, path)
    assert resumed.exhausted and resumed.state.exhausted_reason == "over_reservation"
    with pytest.raises(BudgetExceeded, match="over_reservation"):
        resumed.reserve(2)
    assert resumed.state.attempts == 1
    assert json.loads(path.read_text())["exhausted_reason"] == "over_reservation"


def test_unknown_usage_keeps_estimate_without_reporting_known_overrun(tmp_path: Path) -> None:
    ledger = Ledger(preflight(), tmp_path / "ledger.json")
    reservation = ledger.reserve(201)
    ledger.settle(reservation, None)

    assert ledger.state.settled_input_tokens == 101
    assert ledger.state.settled_output_tokens == 2048
    assert ledger.state.settled_cost == pytest.approx(reservation[2])
    assert ledger.state.unknown_usage_attempts == 1
    assert ledger.state.over_reservation_attempts == 0
    assert ledger.state.settled_cap_breaches == []
    assert not ledger.exhausted
