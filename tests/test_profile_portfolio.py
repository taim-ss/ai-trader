import pytest

from ai_trader.diagnosis import diagnose
from ai_trader.portfolio import parse_holdings_csv
from ai_trader.profile import parse_profile

VALID_PROFILE = {
    "goal": "CV project and short-term gains",
    "horizon": "3m",
    "risk_tolerance": "medium",
    "budget_eur": 240.8,
    "reserve_months": 3,
    "mode": "paper",
}


def test_parse_profile_accepts_valid_input() -> None:
    profile = parse_profile(VALID_PROFILE)
    assert profile.horizon == "3m"
    assert profile.mode == "paper"
    assert profile.budget_eur == 240.8


def test_parse_profile_defaults_to_advise_mode() -> None:
    raw = {k: v for k, v in VALID_PROFILE.items() if k != "mode"}
    assert parse_profile(raw).mode == "advise"


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal", "  "),
        ("horizon", "2w"),
        ("risk_tolerance", "extreme"),
        ("budget_eur", 0),
        ("budget_eur", True),
        ("reserve_months", -1),
        ("mode", "live-now"),
    ],
)
def test_parse_profile_rejects_invalid_fields(field: str, value: object) -> None:
    raw = {**VALID_PROFILE, field: value}
    with pytest.raises(ValueError):
        parse_profile(raw)


CSV_OK = "ticker,quantity,value_eur\nnvda,0.25,48.64\ntsm,0.1,39.08\nnow,0.27,37.77\n"


def test_parse_holdings_uppercases_tickers_and_computes_weights() -> None:
    portfolio = parse_holdings_csv(CSV_OK)
    weights = portfolio.weights()
    assert set(weights) == {"NVDA", "TSM", "NOW"}
    assert round(sum(weights.values()), 6) == 1.0


@pytest.mark.parametrize(
    "text,message",
    [
        ("ticker,quantity\nNVDA,1\n", "must contain columns"),
        ("ticker,quantity,value_eur\n,1,10\n", "empty ticker"),
        ("ticker,quantity,value_eur\nNVDA,1,10\nNVDA,2,20\n", "duplicate"),
        ("ticker,quantity,value_eur\nNVDA,abc,10\n", "must be numbers"),
        ("ticker,quantity,value_eur\nNVDA,-1,10\n", "negative"),
        ("ticker,quantity,value_eur\n", "no holdings"),
    ],
)
def test_parse_holdings_rejects_bad_files(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_holdings_csv(text)


def test_diagnosis_flags_names_over_cap_and_low_diversification() -> None:
    portfolio = parse_holdings_csv(CSV_OK)
    result = diagnose(portfolio, single_name_cap=0.15)
    assert result.holdings_count == 3
    assert result.largest_ticker == "NVDA"
    assert set(result.names_over_cap) == {"NVDA", "TSM", "NOW"}
    assert any("above the 15% cap" in f for f in result.findings)
    assert any("only 3 holdings" in f for f in result.findings)


def test_diagnosis_rejects_invalid_cap() -> None:
    portfolio = parse_holdings_csv(CSV_OK)
    with pytest.raises(ValueError):
        diagnose(portfolio, single_name_cap=0)
