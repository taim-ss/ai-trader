"""One-tap phone approval over ntfy.sh: the bot posts the exact order, a human taps Approve or Reject.

Approve/Reject are ntfy "http" action buttons that POST a signed token to a private approval topic.
The token is an HMAC bound to the decision, a fresh order id, ticker, quantity, limit price and an
expiry, so it cannot be replayed for another order or after it expires. The bot only reads (polls)
the approval topic: no inbound server is needed. Anyone who learns the approval topic could tap a
proposal the bot already made; they cannot create or change an order. Treat both topics as secrets.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from ai_trader.approval import ApprovedOrder

NTFY_URL = "https://ntfy.sh"
DEFAULT_TTL_SECONDS = 600  # an unanswered proposal dies after 10 minutes; quotes go stale anyway


@dataclass(frozen=True)
class PhoneConfig:
    alert_topic: str  # where the phone is subscribed (the existing NTFY_TOPIC)
    approval_topic: str  # private topic the buttons POST to; not subscribed anywhere
    secret: str  # hex HMAC key, never sent anywhere
    base_url: str = NTFY_URL

    def __post_init__(self) -> None:
        for name in ("alert_topic", "approval_topic", "secret", "base_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or any(c.isspace() or c == "/" for c in value if name != "base_url"):
                raise ValueError(f"{name} must be a non-empty string without spaces or slashes")
        if self.alert_topic == self.approval_topic:
            raise ValueError("the approval topic must differ from the alert topic")
        if len(self.approval_topic) < 24 or len(self.secret) < 32:
            raise ValueError("approval topic and secret are too short to be private")


def read_user_env(name: str) -> str:
    """Process environment first, then the Windows user environment (a long-running shell may predate it)."""
    import os

    value = os.environ.get(name, "").strip()
    if value or sys.platform != "win32":
        return value
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, name)[0]).strip()
    except OSError:
        return ""


def load_phone_config(path: Path, alert_topic: str | None = None) -> PhoneConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    topic = alert_topic if alert_topic is not None else read_user_env("NTFY_TOPIC")
    if not topic:
        raise ValueError("NTFY_TOPIC is not set; the phone has nowhere to be notified")
    return PhoneConfig(alert_topic=topic, approval_topic=str(raw.get("approval_topic", "")),
                       secret=str(raw.get("secret", "")), base_url=str(raw.get("base_url", NTFY_URL)))


def sign(secret: str, decision: str, order_id: str, ticker: str, quantity: float, limit_price: float,
         expires: int) -> str:
    message = f"{decision}|{order_id}|{ticker}|{quantity!r}|{limit_price!r}|{expires}".encode()
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()[:40]


def token_body(secret: str, decision: str, order_id: str, order: ApprovedOrder, expires: int) -> str:
    ticker, quantity, limit_price = order
    return f"{decision.upper()} {order_id} {expires} {sign(secret, decision.upper(), order_id, ticker, quantity, limit_price, expires)}"


def verify(secret: str, body: str, order_id: str, order: ApprovedOrder, now_epoch: float) -> str | None:
    """'approve' or 'reject' for a valid, unexpired token for exactly this order; otherwise None."""
    parts = body.strip().split()
    if len(parts) != 4 or parts[0] not in ("APPROVE", "REJECT") or parts[1] != order_id:
        return None
    try:
        expires = int(parts[2])
    except ValueError:
        return None
    if expires <= now_epoch:
        return None
    ticker, quantity, limit_price = order
    expected = sign(secret, parts[0], order_id, ticker, quantity, limit_price, expires)
    return parts[0].lower() if hmac.compare_digest(parts[3], expected) else None


class PhoneApprover:
    """Posts a proposal with Approve/Reject buttons and waits for a signed tap. Never decides by itself."""

    def __init__(self, config: PhoneConfig, client: httpx.Client, *, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 poll_seconds: float = 5.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        if ttl_seconds < 30 or poll_seconds <= 0:
            raise ValueError("ttl must be at least 30 seconds and polling interval positive")
        self.config, self.client, self.ttl, self.poll = config, client, ttl_seconds, poll_seconds
        self.sleep, self.clock = sleep, clock

    def request(self, text: str, order: ApprovedOrder, *, title: str = "Approve order?") -> str:
        """Returns 'approve', 'reject' or 'timeout'. A publish failure raises: no proposal, no order."""
        order_id = uuid.uuid4().hex[:12]
        started = int(self.clock())
        expires = started + self.ttl
        approval_url = f"{self.config.base_url}/{self.config.approval_topic}"
        actions = [{"action": "http", "label": label, "url": approval_url, "method": "POST", "clear": True,
                    "body": token_body(self.config.secret, decision, order_id, order, expires)}
                   for label, decision in (("Approve", "approve"), ("Reject", "reject"))]
        payload = {"topic": self.config.alert_topic, "title": title, "priority": 5, "tags": ["warning"],
                   "message": f"{text}\nExpires in {self.ttl // 60} min. Tap Approve to send it, Reject or ignore to cancel.",
                   "actions": actions}
        response = self.client.post(self.config.base_url + "/", content=json.dumps(payload),
                                    headers={"Content-Type": "application/json"}, timeout=15.0)
        response.raise_for_status()
        while self.clock() < expires:
            decision = self._poll(order_id, order, started - 1)
            if decision is not None:
                return decision
            self.sleep(self.poll)
        return "timeout"

    def _poll(self, order_id: str, order: ApprovedOrder, since_epoch: int) -> str | None:
        try:
            response = self.client.get(f"{self.config.base_url}/{self.config.approval_topic}/json",
                                       params={"poll": "1", "since": str(since_epoch)}, timeout=15.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return None  # a flaky poll is retried until the proposal expires; it never approves
        for line in response.text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            body = event.get("message") if isinstance(event, dict) and event.get("event") == "message" else None
            if isinstance(body, str):
                decision = verify(self.config.secret, body, order_id, order, self.clock())
                if decision is not None:
                    return decision
        return None
