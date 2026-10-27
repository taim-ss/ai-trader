"""Broker boundary. Phase 3 defines the interface only; no network calls here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

TRADING212_DEMO_URL = "https://demo.trading212.com/api/v0"
TRADING212_LIVE_URL = "https://live.trading212.com/api/v0"


@dataclass(frozen=True)
class Order:
    ticker: str
    quantity: float  # negative to sell (Trading 212 convention)
    side: Literal["buy", "sell"]


class BrokerClient(Protocol):
    """Implemented by the paper adapter first. Live adapters need owner approval."""

    def account_summary(self) -> dict[str, object]: ...

    def place_order(self, order: Order) -> dict[str, object]: ...
