"""Human approval step for orders: show the exact order, send it only after the typed confirmation phrase.

The order that is sent must equal the order that was shown (ticker, quantity, limit price); the
executor re-checks this immediately before the POST. Anything other than the phrase cancels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

ApprovedOrder = tuple[str, float, float]  # api ticker, signed quantity, limit price in the instrument currency
DEMO_VENUE = "Trading 212 DEMO account (virtual money)"
LIVE_VENUE = "Trading 212 LIVE account (REAL MONEY)"


def order_text(planned: Mapping[str, Any], symbol: str, fx_native_to_eur: float, venue: str = DEMO_VENUE) -> str:
    quantity = float(planned["quantity"])
    price = float(planned["limit_price"])
    side = "BUY" if quantity > 0 else "SELL"
    euro = abs(quantity) * price * fx_native_to_eur
    return (f"{side} {abs(quantity):g} x {planned['ticker']} ({symbol}) with a {planned['time_validity']} limit of "
            f"{price:.2f}, about EUR {euro:.2f}, on the {venue}.")


def run_with_approval(
    dry_run: Callable[[], dict[str, Any]],
    send: Callable[[ApprovedOrder], dict[str, Any]],
    ask: Callable[[str], str],
    *,
    symbols: Mapping[str, str],
    fx_native_to_eur: float,
    confirm_phrase: str = "yes",
    venue: str = DEMO_VENUE,
) -> dict[str, Any]:
    """Dry-run first; ask a human; send exactly the shown order, or cancel. Never sends without the phrase."""
    result = dry_run()
    planned = result.get("planned")
    if result.get("status") != "dry-run" or result.get("violations") or not isinstance(planned, dict):
        return result  # nothing to approve: blocked, completed, or no order planned
    approved: ApprovedOrder = (str(planned["ticker"]), float(planned["quantity"]), float(planned["limit_price"]))
    text = order_text(planned, symbols.get(approved[0], approved[0]), fx_native_to_eur, venue)
    answer = ask(f"\nOrder ready:\n  {text}\nType {confirm_phrase} to send this one order, anything else cancels: ")
    if answer.strip().lower() != confirm_phrase.lower():
        result["status"] = "cancelled"
        result["note"] = "not approved; nothing was sent"
        return result
    sent = send(approved)
    sent["approved_order"] = {"ticker": approved[0], "quantity": approved[1], "limit_price": approved[2]}
    return sent
