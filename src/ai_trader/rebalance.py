"""Turn target weights into signed share orders for the paper account.

Pure functions only: no network, no broker. The caller supplies prices, FX and holdings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from math import isfinite
from typing import Any

from ai_trader.risk import RiskLimits, validate_order, validate_targets

SHARE_DECIMALS = 4


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    api_ticker: str
    quantity: float  # signed change in shares: positive buys, negative sells
    value_account: float  # absolute value in the account currency


@dataclass(frozen=True)
class RebalancePlan:
    orders: tuple[PlannedOrder, ...]
    violations: tuple[str, ...]


def resolve_api_tickers(
    symbols: list[str], instruments: list[dict[str, Any]], expected_currency: str | None = None,
) -> dict[str, str]:
    """Map plain symbols (NVDA) to Trading 212 tickers (NVDA_US_EQ).

    Refuses to guess: a symbol with zero or several matches raises ValueError.
    """
    mapping: dict[str, str] = {}
    for symbol in symbols:
        matches = sorted(
            str(item["ticker"])
            for item in instruments
            if str(item.get("ticker", "")).split("_")[0] == symbol
        )
        if len(matches) != 1:
            raise ValueError(f"{symbol}: expected one Trading 212 instrument, found {len(matches)}")
        if expected_currency is not None:
            item = next(item for item in instruments if item.get("ticker") == matches[0])
            if item.get("currencyCode") != expected_currency:
                raise ValueError(f"{symbol}: instrument currency must be {expected_currency}")
        mapping[symbol] = matches[0]
    return mapping


def plan_rebalance(
    targets: Mapping[str, float],
    prices_usd: Mapping[str, float],
    fx_usd_to_account: float,
    holdings_qty: Mapping[str, float],
    portfolio_value: float,
    api_tickers: Mapping[str, str],
    limits: RiskLimits,
    min_value: float = 1.0,
    fx_account_to_eur: float = 1.0,
) -> RebalancePlan:
    """Compute the orders that move current holdings to the target weights."""
    if isinstance(portfolio_value, bool) or not isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("portfolio value must be positive")
    if isinstance(fx_usd_to_account, bool) or not isfinite(fx_usd_to_account) or fx_usd_to_account <= 0:
        raise ValueError("fx rate must be positive")
    if isinstance(fx_account_to_eur, bool) or not isfinite(fx_account_to_eur) or fx_account_to_eur <= 0:
        raise ValueError("account-to-EUR fx rate must be finite and positive")
    if isinstance(min_value, bool) or not isfinite(min_value) or min_value < 0:
        raise ValueError("minimum order value must be finite and nonnegative")
    if any(isinstance(weight, bool) or not isfinite(weight) for weight in targets.values()):
        raise ValueError("target weights must be finite numbers")
    for symbol, quantity in holdings_qty.items():
        if isinstance(quantity, bool) or not isfinite(quantity) or (quantity < 0 and not limits.allow_shorting):
            raise ValueError(f"{symbol}: holding quantity must be finite and nonnegative")

    violations = list(validate_targets(targets, limits))
    symbols = sorted(set(targets) | set(holdings_qty))
    orders: list[PlannedOrder] = []
    for symbol in symbols:
        if symbol not in prices_usd or symbol not in api_tickers:
            raise ValueError(f"{symbol}: missing price or Trading 212 ticker")
        if isinstance(prices_usd[symbol], bool):
            raise ValueError(f"{symbol}: price must be a finite positive number")
        price_account = prices_usd[symbol] * fx_usd_to_account
        if not isfinite(price_account) or price_account <= 0:
            raise ValueError(f"{symbol}: price must be positive")
        target_shares = targets.get(symbol, 0.0) * portfolio_value / price_account
        if not isfinite(target_shares):
            raise ValueError(f"{symbol}: target shares must be finite")
        # Truncate toward zero: rounding a sell up can exceed the shares actually owned.
        delta = float(Decimal(str(target_shares - holdings_qty.get(symbol, 0.0))).quantize(
            Decimal(10) ** -SHARE_DECIMALS, rounding=ROUND_DOWN,
        ))
        value = abs(delta) * price_account
        if delta == 0 or value < min_value:
            continue
        violations.extend(f"{symbol}: {msg}" for msg in validate_order(value * fx_account_to_eur, limits))
        orders.append(
            PlannedOrder(
                symbol=symbol,
                api_ticker=api_tickers[symbol],
                quantity=delta,
                value_account=value,
            )
        )
    return RebalancePlan(orders=tuple(orders), violations=tuple(violations))
