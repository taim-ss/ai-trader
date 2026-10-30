"""One-time phone setup: create the private approval topic and signing key, then send a test tap.

  python scripts/setup_phone.py            # create config\\private_notify.yaml if missing, send a test notification
  python scripts/setup_phone.py --wait 180 # also wait up to 180 s for you to tap Approve on the test

The alert topic is your existing NTFY_TOPIC (the one your phone already follows). The approval topic and
key are random, stored only in the git-ignored config\\private_notify.yaml, and never printed.
"""
from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import httpx
import truststore

from ai_trader.phone_approval import PhoneApprover, load_phone_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "private_notify.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for a test tap (0 = send only)")
    args = parser.parse_args()
    truststore.inject_into_ssl()  # this PC verifies TLS through the Windows certificate store
    if not CONFIG.exists():
        CONFIG.write_text(
            "# Private phone-approval settings. Git-ignored. Never share: anyone with the approval topic and key\n"
            "# could tap Approve on a proposal the bot already made.\n"
            f"approval_topic: tb-appr-{secrets.token_urlsafe(24)}\n"
            f"secret: {secrets.token_hex(32)}\n", encoding="utf-8")
        print("created config/private_notify.yaml (values not shown)")
    phone = load_phone_config(CONFIG)
    test_order = ("TEST_US_EQ", 0.0, 1.0)  # a made-up order id: it can never match a real one
    with httpx.Client() as client:
        approver = PhoneApprover(phone, client, ttl_seconds=max(args.wait, 30) if args.wait else 30)
        if args.wait:
            outcome = approver.request("TEST ONLY: tap Approve to prove the phone link works. No order exists.",
                                       test_order, title="Trading bot: phone link test")
            print(f"phone test result: {outcome}")
            return 0 if outcome == "approve" else 1
        approver.client.post(phone.base_url + "/", json={
            "topic": phone.alert_topic, "title": "Trading bot: phone link ready",
            "message": "Setup complete. Real order proposals will arrive here with Approve / Reject buttons.",
            "tags": ["white_check_mark"]}, timeout=15.0).raise_for_status()
        print("sent a plain notification to your phone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
