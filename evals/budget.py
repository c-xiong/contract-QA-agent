"""Attempt limits and estimated spend reservations for paid experiments.

DECISION: a call reserves estimated input tokens and the declared output ceiling before
  it is sent, and is refused if that reservation would breach any cap. Input estimates
  are not tokenizer bounds: an in-flight call can exceed them and the declared spend
  caps. Settlement records reported usage and cap breaches; any reservation overrun
  stops subsequent attempts. Unknown usage (timeout, provider error, cancellation)
  keeps the estimated reservation consumed, because unknown is not zero.

DECISION: the ledger is a file, rewritten atomically after every reservation and
  settlement. A resumed run loads it and continues from its cumulative totals, so an
  interruption can never reset the budget. Alternative rejected: in-memory counters
  plus a final summary, which forget spend whenever a paid run crashes.

Prices are not built in. The preflight must state the selected model's current prices;
this module refuses a paid run without them rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.llm import ModelClient, ModelError, ModelResponse

# Character-based input estimate, not a worst-case tokenizer bound. The historical
# constant name is retained for compatibility; settlement uses reported token counts.
CHARS_PER_TOKEN_BOUND = 2


class Preflight(BaseModel):
    """Declared before a paid run and hashed into its artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    phase: Literal["application", "judge"]
    model_id: str
    currency: str = "USD"
    input_price_per_million_tokens: float = Field(gt=0)
    output_price_per_million_tokens: float = Field(gt=0)
    max_attempts: int = Field(ge=0)
    max_input_tokens_per_call: int = Field(gt=0)
    max_output_tokens_per_call: int = Field(gt=0)
    max_total_input_tokens: int = Field(ge=0)
    max_total_output_tokens: int = Field(ge=0)
    max_cost: float = Field(ge=0)
    stop_conditions: list[str] = Field(default_factory=list)
    notes: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_price_per_million_tokens
            + output_tokens * self.output_price_per_million_tokens
        ) / 1_000_000

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class BudgetExceeded(ModelError):
    """A call was refused before sending by a cap check or prior ledger exhaustion.

    Subclasses ModelError so the graph classifies it as a failed model call, never as
    an abstention. Runners check `Ledger.exhausted` to stop the whole experiment.
    """


class LedgerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preflight_sha256: str
    attempts: int = 0
    refused: int = 0
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    reserved_cost: float = 0.0
    settled_input_tokens: int = 0
    settled_output_tokens: int = 0
    settled_cost: float = 0.0
    unknown_usage_attempts: int = 0
    over_reservation_attempts: int = 0
    settled_cap_breaches: list[str] = Field(default_factory=list)
    exhausted_reason: str | None = None
    updated_at: str = ""

    @property
    def input_tokens(self) -> int:
        return self.reserved_input_tokens + self.settled_input_tokens

    @property
    def output_tokens(self) -> int:
        return self.reserved_output_tokens + self.settled_output_tokens

    @property
    def cost(self) -> float:
        return self.reserved_cost + self.settled_cost


class Ledger:
    def __init__(self, preflight: Preflight, path: Path):
        self.preflight, self.path = preflight, path
        self._lock = threading.Lock()
        if path.exists():
            self.state = LedgerState.model_validate_json(path.read_text())
            if self.state.preflight_sha256 != preflight.digest():
                raise ValueError(
                    f"{path} belongs to a different preflight; a changed budget needs a new "
                    "experiment id, not a reset ledger"
                )
            # Older ledgers counted overruns without stopping. Resuming one must not
            # grant fresh attempts, even when its cumulative totals remain below caps.
            breaches = self._settled_cap_breaches()
            if self.state.over_reservation_attempts and self.state.exhausted_reason is None:
                self.state.exhausted_reason = "over_reservation"
            elif breaches and self.state.exhausted_reason is None:
                self.state.exhausted_reason = breaches[0]
            if breaches != self.state.settled_cap_breaches or self.state.exhausted_reason:
                self.state.settled_cap_breaches = breaches
                self._save()
        else:
            self.state = LedgerState(preflight_sha256=preflight.digest())
            self._save()

    @property
    def exhausted(self) -> bool:
        return self.state.exhausted_reason is not None

    def _save(self) -> None:
        self.state.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(self.state.model_dump_json(indent=2) + "\n")
        os.replace(temporary, self.path)

    def reserve(self, prompt_chars: int) -> tuple[int, int, float]:
        p, s = self.preflight, self.state
        input_estimate = math.ceil(prompt_chars / CHARS_PER_TOKEN_BOUND)
        output_ceiling = p.max_output_tokens_per_call
        cost = p.cost(input_estimate, output_ceiling)
        with self._lock:
            if s.exhausted_reason is not None:
                s.refused += 1
                self._save()
                raise BudgetExceeded(f"budget refused call: {s.exhausted_reason}")
            reason = None
            if input_estimate > p.max_input_tokens_per_call:
                reason = "per_call_input_cap"
            elif s.attempts + 1 > p.max_attempts:
                reason = "attempt_cap"
            elif s.input_tokens + input_estimate > p.max_total_input_tokens:
                reason = "input_token_cap"
            elif s.output_tokens + output_ceiling > p.max_total_output_tokens:
                reason = "output_token_cap"
            elif s.cost + cost > p.max_cost:
                reason = "cost_cap"
            if reason:
                s.refused += 1
                # A single oversized prompt does not end the experiment; a cumulative cap does.
                if reason != "per_call_input_cap":
                    s.exhausted_reason = reason
                self._save()
                raise BudgetExceeded(f"budget refused call: {reason}")
            s.attempts += 1
            s.reserved_input_tokens += input_estimate
            s.reserved_output_tokens += output_ceiling
            s.reserved_cost += cost
            self._save()
        return input_estimate, output_ceiling, cost

    def _settled_cap_breaches(
        self, input_tokens: int | None = None, output_tokens: int | None = None
    ) -> list[str]:
        """Accumulate known per-call breaches and cumulative settled accounting breaches.

        Unknown-usage attempts remain estimated within the settled totals. Historical
        per-call usage cannot be reconstructed from a ledger's aggregate counters.
        """
        p, s = self.preflight, self.state
        breaches = set(s.settled_cap_breaches)
        if input_tokens is not None and input_tokens > p.max_input_tokens_per_call:
            breaches.add("per_call_input_cap")
        if output_tokens is not None and output_tokens > p.max_output_tokens_per_call:
            breaches.add("per_call_output_cap")
        if s.settled_input_tokens > p.max_total_input_tokens:
            breaches.add("input_token_cap")
        if s.settled_output_tokens > p.max_total_output_tokens:
            breaches.add("output_token_cap")
        if s.settled_cost > p.max_cost:
            breaches.add("cost_cap")
        return sorted(breaches)

    def settle(self, reservation: tuple[int, int, float], response: ModelResponse | None) -> None:
        input_estimate, output_ceiling, cost = reservation
        p, s = self.preflight, self.state
        with self._lock:
            if response is None:
                # Unknown usage keeps its full reservation, moved to settled spend.
                s.unknown_usage_attempts += 1
                actual_in, actual_out, actual_cost = input_estimate, output_ceiling, cost
            else:
                actual_in, actual_out = response.input_tokens, response.output_tokens
                actual_cost = p.cost(actual_in, actual_out)
                if actual_in > input_estimate or actual_out > output_ceiling:
                    s.over_reservation_attempts += 1
                    if s.exhausted_reason is None:
                        s.exhausted_reason = "over_reservation"
            s.reserved_input_tokens -= input_estimate
            s.reserved_output_tokens -= output_ceiling
            s.reserved_cost -= cost
            s.settled_input_tokens += actual_in
            s.settled_output_tokens += actual_out
            s.settled_cost += actual_cost
            s.settled_cap_breaches = self._settled_cap_breaches(
                actual_in if response is not None else None,
                actual_out if response is not None else None,
            )
            if s.settled_cap_breaches and s.exhausted_reason is None:
                s.exhausted_reason = s.settled_cap_breaches[0]
            self._save()

    def summary(self) -> dict[str, Any]:
        s = self.state
        return s.model_dump() | {
            "preflight": self.preflight.model_dump(),
            "remaining_attempts": self.preflight.max_attempts - s.attempts,
            "remaining_cost": self.preflight.max_cost - s.cost,
        }


class BudgetedClient:
    """Wraps a live client; every provider attempt passes through the ledger."""

    def __init__(self, inner: ModelClient, ledger: Ledger):
        if inner.model_id != ledger.preflight.model_id:
            raise ValueError(
                f"preflight prices {ledger.preflight.model_id}, but the client is {inner.model_id}"
            )
        self.inner, self.ledger = inner, ledger
        self.model_id = inner.model_id

    async def complete(self, system: str, user: str) -> ModelResponse:
        reservation = self.ledger.reserve(len(system) + len(user))
        response: ModelResponse | None = None
        try:
            response = await self.inner.complete(system, user)
            return response
        finally:
            # Also runs on ModelError and cancellation: those keep the whole reservation.
            self.ledger.settle(reservation, response)


def load_preflight(path: Path) -> Preflight:
    return Preflight.model_validate(json.loads(path.read_text()))
