from pathlib import Path
from typing import Any

import pytest

from ai_trader.approval import ApprovedOrder, order_text, run_with_approval
from tests.test_demo_execution import FakeBroker, plan, run

PLANNED = {"ticker": "AAA_US_EQ", "quantity": 0.2, "limit_price": 100.0, "time_validity": "DAY"}


def dry() -> dict[str, Any]:
    return {"status": "dry-run", "violations": [], "submitted": [], "reconciled": [], "planned": dict(PLANNED)}


def test_order_text_states_side_size_price_and_demo_only() -> None:
    text = order_text(PLANNED, "AAA", 0.9)
    assert "BUY 0.2 x AAA_US_EQ (AAA)" in text and "100.00" in text and "EUR 18.00" in text and "DEMO" in text
    assert order_text({**PLANNED, "quantity": -1.5}, "AAA", 1.0).startswith("SELL 1.5")


@pytest.mark.parametrize("answer", ["", "no", "y", "YES please", "yes!", "\n"])
def test_anything_but_yes_cancels_and_never_sends(answer: str) -> None:
    sent: list[ApprovedOrder] = []
    result = run_with_approval(dry, lambda a: sent.append(a) or {}, lambda _: answer,
                               symbols={"AAA_US_EQ": "AAA"}, fx_native_to_eur=1.0)
    assert result["status"] == "cancelled" and sent == []


@pytest.mark.parametrize("answer", ["yes", " YES ", "Yes\n"])
def test_yes_sends_exactly_the_shown_order(answer: str) -> None:
    seen: list[ApprovedOrder] = []
    prompts: list[str] = []

    def send(order: ApprovedOrder) -> dict[str, Any]:
        seen.append(order)
        return {"status": "submitted", "submitted": [{"ticker": order[0]}]}

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    result = run_with_approval(dry, send, ask, symbols={"AAA_US_EQ": "AAA"}, fx_native_to_eur=1.0)
    assert seen == [("AAA_US_EQ", 0.2, 100.0)] and result["status"] == "submitted"
    assert result["approved_order"]["limit_price"] == 100.0 and "AAA_US_EQ" in prompts[0]


@pytest.mark.parametrize("blocked", [
    {"status": "blocked", "violations": ["x"], "submitted": []},
    {"status": "completed", "violations": [], "submitted": []},
    {"status": "dry-run", "violations": ["pending broker orders"], "submitted": [], "planned": dict(PLANNED)},
    {"status": "dry-run", "violations": [], "submitted": []},
])
def test_nothing_to_approve_means_no_question_and_no_send(blocked: dict[str, Any]) -> None:
    def never(_: str) -> str:
        raise AssertionError("must not ask")

    def never_send(_: ApprovedOrder) -> dict[str, Any]:
        raise AssertionError("must not send")

    assert run_with_approval(lambda: dict(blocked), never_send, never, symbols={}, fx_native_to_eur=1.0) == blocked


def test_executor_refuses_an_order_that_differs_from_the_approved_one(tmp_path: Path) -> None:
    broker = FakeBroker()
    wrong_quantity = run(broker, tmp_path, approved=("AAA_US_EQ", 0.3, 100.0))
    wrong_price = run(broker, tmp_path, approved=("AAA_US_EQ", 0.2, 101.0))
    wrong_ticker = run(broker, tmp_path, approved=("BBB_US_EQ", 0.2, 100.0))
    for blocked in (wrong_quantity, wrong_price, wrong_ticker):
        assert blocked["status"] == "blocked" and "approved" in " ".join(blocked["violations"])
    assert broker.sent == []


def test_executor_sends_the_approved_order_once(tmp_path: Path) -> None:
    broker = FakeBroker()
    result = run(broker, tmp_path, selected=plan(("AAA_US_EQ", 0.2)), approved=("AAA_US_EQ", 0.2, 100.0))
    assert result["status"] == "submitted" and broker.sent == [("AAA_US_EQ", 0.2, 100.0)]
