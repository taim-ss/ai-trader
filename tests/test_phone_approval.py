import json
from typing import Any

import httpx
import pytest

from ai_trader.approval import ApprovedOrder, run_with_decider
from ai_trader.phone_approval import PhoneApprover, PhoneConfig, load_phone_config, sign, token_body, verify

ALERT = "alert-topic-the-phone-listens-to"
APPROVAL = "approval-topic-" + "x" * 20
SECRET = "ab" * 32
ORDER: ApprovedOrder = ("AAA_US_EQ", 0.2, 100.0)
CONFIG = PhoneConfig(ALERT, APPROVAL, SECRET)


def test_token_is_bound_to_decision_order_and_expiry() -> None:
    body = token_body(SECRET, "approve", "id1", ORDER, 2000)
    assert verify(SECRET, body, "id1", ORDER, 1000) == "approve"
    assert verify(SECRET, token_body(SECRET, "reject", "id1", ORDER, 2000), "id1", ORDER, 1000) == "reject"
    assert verify(SECRET, body, "id2", ORDER, 1000) is None  # other proposal
    assert verify(SECRET, body, "id1", ("AAA_US_EQ", 0.3, 100.0), 1000) is None  # other order
    assert verify(SECRET, body, "id1", ORDER, 2000) is None  # expired
    assert verify("cd" * 32, body, "id1", ORDER, 1000) is None  # wrong key
    assert verify(SECRET, body.replace("APPROVE", "REJECT"), "id1", ORDER, 1000) is None  # decision swapped
    assert verify(SECRET, body.replace("2000", "9999"), "id1", ORDER, 1000) is None  # expiry extended
    for junk in ("", "APPROVE", "APPROVE id1 x y", "hello world", body + " extra"):
        assert verify(SECRET, junk, "id1", ORDER, 1000) is None
    assert sign(SECRET, "APPROVE", "id1", *ORDER, 2000) != sign(SECRET, "REJECT", "id1", *ORDER, 2000)


@pytest.mark.parametrize("changes", [{"approval_topic": ALERT}, {"approval_topic": "short"}, {"secret": "ab"},
                                      {"alert_topic": ""}, {"approval_topic": "has space" + "x" * 20}])
def test_weak_or_confused_config_is_rejected(changes: dict[str, str]) -> None:
    values = {"alert_topic": ALERT, "approval_topic": APPROVAL, "secret": SECRET, **changes}
    with pytest.raises(ValueError):
        PhoneConfig(**values)


class FakeNtfy:
    """In-memory ntfy: records the published proposal and serves approval-topic messages."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.inbox: list[str] = []
        self.fail_polls = 0
        self.polls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.published.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "x"})
        self.polls += 1
        if self.fail_polls > 0:
            self.fail_polls -= 1
            return httpx.Response(500)
        lines = [json.dumps({"event": "open"})] + [json.dumps({"event": "message", "message": m}) for m in self.inbox]
        return httpx.Response(200, text="\n".join(lines) + "\nnot json")


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def approver(ntfy: FakeNtfy, clock: Clock, ttl: int = 600) -> PhoneApprover:
    client = httpx.Client(transport=httpx.MockTransport(ntfy.handler))
    return PhoneApprover(CONFIG, client, ttl_seconds=ttl, poll_seconds=10, sleep=clock.sleep, clock=clock)


def tap(ntfy: FakeNtfy, which: str) -> str:
    button = next(a for a in ntfy.published[-1]["actions"] if a["label"].lower() == which)
    assert button["url"].endswith("/" + APPROVAL) and button["method"] == "POST"
    return str(button["body"])


def test_proposal_is_published_with_two_buttons_to_the_phone_topic() -> None:
    ntfy, clock = FakeNtfy(), Clock()
    result = approver(ntfy, clock, ttl=30).request("BUY something", ORDER, title="Approve LIVE order?")
    assert result == "timeout"  # nobody tapped
    sent = ntfy.published[0]
    assert sent["topic"] == ALERT and sent["title"] == "Approve LIVE order?" and "BUY something" in sent["message"]
    assert [a["label"] for a in sent["actions"]] == ["Approve", "Reject"]
    assert SECRET not in json.dumps(sent)  # the key itself never leaves the machine


def test_a_valid_approve_tap_returns_approve_and_a_reject_tap_returns_reject() -> None:
    for which, expected in (("approve", "approve"), ("reject", "reject")):
        ntfy, clock = FakeNtfy(), Clock()
        real = ntfy.handler

        def handler(request: httpx.Request, ntfy: FakeNtfy = ntfy, which: str = which, real: Any = real) -> httpx.Response:
            if request.method == "GET" and ntfy.published:
                ntfy.inbox = ["somebody else", tap(ntfy, which)]
            return real(request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        phone = PhoneApprover(CONFIG, client, poll_seconds=10, sleep=clock.sleep, clock=clock)
        assert phone.request("BUY", ORDER) == expected


def test_taps_for_other_orders_or_forged_tokens_never_approve() -> None:
    ntfy, clock = FakeNtfy(), Clock()
    ntfy.inbox = [token_body(SECRET, "approve", "someotherid", ORDER, 99999),
                  token_body("cd" * 32, "approve", "whatever", ORDER, 99999), "APPROVE a 99999 deadbeef"]
    assert approver(ntfy, clock, ttl=60).request("BUY", ORDER) == "timeout"


def test_flaky_polls_are_retried_but_never_count_as_approval() -> None:
    ntfy, clock = FakeNtfy(), Clock()
    ntfy.fail_polls = 3
    assert approver(ntfy, clock, ttl=60).request("BUY", ORDER) == "timeout" and ntfy.polls >= 4


def test_publish_failure_raises_so_no_order_can_follow() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        PhoneApprover(CONFIG, client, clock=Clock(), sleep=Clock().sleep).request("BUY", ORDER)


def test_phone_decision_gates_the_executor_flow() -> None:
    planned = {"ticker": "AAA_US_EQ", "quantity": 0.2, "limit_price": 100.0, "time_validity": "DAY"}
    dry = lambda: {"status": "dry-run", "violations": [], "submitted": [], "planned": dict(planned)}  # noqa: E731
    sent: list[ApprovedOrder] = []

    def send(order: ApprovedOrder) -> dict[str, Any]:
        sent.append(order)
        return {"status": "submitted", "violations": [], "submitted": [{"ticker": order[0]}]}

    for outcome in ("reject", "timeout", "anything else"):
        result = run_with_decider(dry, send, lambda text, order, o=outcome: o, symbols={}, fx_native_to_eur=1.0)
        assert result["status"] == "cancelled" and sent == []
    result = run_with_decider(dry, send, lambda text, order: "approve", symbols={}, fx_native_to_eur=1.0)
    assert result["status"] == "submitted" and sent == [ORDER]


def test_load_phone_config_reads_private_file_and_needs_an_alert_topic(tmp_path: Any) -> None:
    path = tmp_path / "private_notify.yaml"
    path.write_text(f"approval_topic: {APPROVAL}\nsecret: {SECRET}\n", encoding="utf-8")
    assert load_phone_config(path, alert_topic=ALERT) == CONFIG
    with pytest.raises(ValueError):
        load_phone_config(path, alert_topic="")
    path.write_text("approval_topic: short\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_phone_config(path, alert_topic=ALERT)
