"""Trading 212 paper (demo) adapter. Refuses any base URL except the demo environment.

Auth: HTTP Basic with API key and secret (see docs.trading212.com/api). Credentials are
read from environment variables only; nothing is stored in code or config.
"""

from __future__ import annotations

import base64
import os
import time
from math import isfinite
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from ai_trader.broker import TRADING212_DEMO_URL

KEY_ENV = "TRADING212_DEMO_API_KEY"
SECRET_ENV = "TRADING212_DEMO_API_SECRET"


def credentials_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Read demo credentials from the environment. Raises a clear error if missing."""
    env = os.environ if environ is None else environ
    key = env.get(KEY_ENV, "").strip()
    secret = env.get(SECRET_ENV, "").strip()
    if not key or not secret:
        raise RuntimeError(
            f"set {KEY_ENV} and {SECRET_ENV} as environment variables (demo keys only)"
        )
    return key, secret


class Trading212Paper:
    """Paper-only client. Live URLs are rejected at construction."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = TRADING212_DEMO_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if base_url != TRADING212_DEMO_URL:
            raise ValueError("paper adapter only talks to the Trading 212 demo environment")
        token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode("ascii")
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Basic {token}"},
            timeout=10.0,
            transport=transport,
        )
        self._last_call: dict[str, float] = {}
        self._rate_enabled = transport is None
        self._instruments_cache: list[dict[str, Any]] | None = None
        self._instruments_at = 0.0

    @property
    def base_url(self) -> str:
        """Immutable environment identity for the execution engine's demo check."""
        return TRADING212_DEMO_URL

    def _pace(self, endpoint: str, seconds: float) -> None:
        if self._rate_enabled:
            remaining = seconds - (time.monotonic() - self._last_call.get(endpoint, -1e9))
            if remaining > 0:
                time.sleep(remaining)
            self._last_call[endpoint] = time.monotonic()

    @staticmethod
    def _object(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("broker response must be an object")
        return data

    @staticmethod
    def _objects(data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ValueError("broker response must be a list of objects")
        return data

    def account_summary(self) -> dict[str, Any]:
        """GET /equity/account/summary. Raises for HTTP errors so failures are never silent."""
        self._pace("summary", 5.1)
        response = self._client.get("/equity/account/summary")
        response.raise_for_status()
        return self._object(response.json())

    def positions(self) -> list[dict[str, Any]]:
        """GET /equity/positions. Each item has `quantity` and `instrument.ticker`."""
        self._pace("positions", 1.1)
        response = self._client.get("/equity/positions")
        response.raise_for_status()
        return self._objects(response.json())

    def instruments(self) -> list[dict[str, Any]]:
        """GET /equity/metadata/instruments (rate limited to 1 request per 50 seconds)."""
        if self._instruments_cache is not None and time.monotonic() - self._instruments_at < 600:
            return [dict(item) for item in self._instruments_cache]
        self._pace("instruments", 50.1)
        response = self._client.get("/equity/metadata/instruments")
        response.raise_for_status()
        self._instruments_cache = self._objects(response.json())
        self._instruments_at = time.monotonic()
        return [dict(item) for item in self._instruments_cache]

    def pending_orders(self) -> list[dict[str, Any]]:
        self._pace("pending", 5.1)
        response = self._client.get("/equity/orders")
        response.raise_for_status()
        return self._objects(response.json())

    def order(self, order_id: str | int) -> dict[str, Any]:
        value = str(order_id)
        if not value.isascii() or not value.isdigit():
            raise ValueError("order ID must be numeric")
        self._pace("order", 1.1)
        response = self._client.get(f"/equity/orders/{value}")
        response.raise_for_status()
        return self._object(response.json())

    def _history(self, endpoint: str) -> list[dict[str, Any]]:
        """Read every page, rejecting off-origin or unexpected pagination paths."""
        path = endpoint + "?limit=50"
        seen: set[str] = set()
        records: list[dict[str, Any]] = []
        for _ in range(100):
            if path in seen:
                raise ValueError("broker pagination cycle")
            seen.add(path)
            self._pace(endpoint, 3.1)
            response = self._client.get(path)
            response.raise_for_status()
            page = self._object(response.json())
            records.extend(self._objects(page.get("items")))
            next_path = page.get("nextPagePath")
            if next_path is None:
                return records
            if not isinstance(next_path, str):
                raise ValueError("invalid broker pagination path")
            parsed = urlsplit(next_path)
            if parsed.scheme or parsed.netloc or parsed.fragment or parsed.path != "/api/v0" + endpoint:
                raise ValueError("unsafe broker pagination path")
            path = next_path.removeprefix("/api/v0")
        raise ValueError("broker history exceeded 100 pages; refusing incomplete reconciliation")

    def order_history(self) -> list[dict[str, Any]]:
        return self._history("/equity/history/orders")

    def transactions(self) -> list[dict[str, Any]]:
        return self._history("/equity/history/transactions")

    def place_limit_order(self, ticker: str, quantity: float, limit_price: float) -> dict[str, Any]:
        """Submit exactly once; callers must journal intent and reconcile ambiguity."""
        if not ticker or ticker != ticker.strip() or ticker.lower() == ticker:
            raise ValueError("ticker must be a canonical broker instrument")
        if isinstance(quantity, bool) or not isfinite(quantity) or quantity == 0:
            raise ValueError("quantity must be finite and non-zero")
        if isinstance(limit_price, bool) or not isfinite(limit_price) or limit_price <= 0:
            raise ValueError("limit price must be finite and positive")
        self._pace("limit", 2.1)
        response = self._client.post(
            "/equity/orders/limit",
            json={"ticker": ticker, "quantity": quantity, "limitPrice": limit_price, "timeValidity": "DAY"},
        )
        response.raise_for_status()
        return self._object(response.json())

    def place_market_order(self, ticker: str, quantity: float) -> dict[str, Any]:
        """POST /equity/orders/market. Positive quantity buys, negative sells.

        `ticker` must be the API ticker (for example AAPL_US_EQ) from the instruments endpoint.
        """
        if not ticker or ticker != ticker.strip() or ticker.lower() == ticker:
            raise ValueError("ticker must be the API ticker in upper case, e.g. AAPL_US_EQ")
        if isinstance(quantity, bool) or not isfinite(quantity) or quantity == 0:
            raise ValueError("quantity must be finite and non-zero")
        response = self._client.post(
            "/equity/orders/market",
            json={"ticker": ticker, "quantity": quantity, "extendedHours": False},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def close(self) -> None:
        self._client.close()
