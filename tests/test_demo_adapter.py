import json

import httpx
import pytest

from ai_trader.paper_adapter import Trading212Paper


def test_limit_order_is_demo_day_only_and_never_retried():
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("unknown", request=request)

    broker = Trading212Paper("test", "test", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.ReadTimeout):
            broker.place_limit_order("SPY_US_EQ", .1, 100)
        assert len(seen) == 1
        assert str(seen[0].url) == "https://demo.trading212.com/api/v0/equity/orders/limit"
        assert json.loads(seen[0].content) == {
            "ticker": "SPY_US_EQ", "quantity": .1, "limitPrice": 100, "timeValidity": "DAY",
        }
    finally:
        broker.close()


def test_history_reads_all_pages_without_leaving_demo_origin():
    paths = []

    def handler(request):
        paths.append(str(request.url))
        if len(paths) == 1:
            return httpx.Response(200, json={"items": [{"order": {"id": 1}}],
                "nextPagePath": "/api/v0/equity/history/orders?limit=50&cursor=1"})
        return httpx.Response(200, json={"items": [{"order": {"id": 2}}], "nextPagePath": None})

    broker = Trading212Paper("test", "test", transport=httpx.MockTransport(handler))
    try:
        assert len(broker.order_history()) == 2
        assert len(paths) == 2 and all(path.startswith(broker.base_url) for path in paths)
    finally:
        broker.close()


@pytest.mark.parametrize("next_path", ["https://live.trading212.com/api/v0/equity/history/orders",
    "//example.com/api/v0/equity/history/orders", "/api/v0/equity/orders", "/api/v0/equity/history/orders#fragment"])
def test_history_refuses_external_or_wrong_pagination(next_path):
    broker = Trading212Paper("test", "test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"items": [], "nextPagePath": next_path})))
    try:
        with pytest.raises(ValueError, match="unsafe"):
            broker.order_history()
    finally:
        broker.close()


def test_read_methods_and_validation():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"id": 10} if request.url.path.endswith("/10") else [])

    broker = Trading212Paper("test", "test", transport=httpx.MockTransport(handler))
    try:
        assert broker.pending_orders() == []
        assert broker.order(10)["id"] == 10
        with pytest.raises(ValueError):
            broker.order("https://other")
        assert all(request.method == "GET" for request in requests)
    finally:
        broker.close()
