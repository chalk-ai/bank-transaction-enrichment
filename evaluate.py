#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Score a deployed enrichment service against the labels the synthetic generator knows.

Sends every memo template (a fresh random fill per seed) to `/api/enrich` and reports merchant
and category accuracy, plus latency per batch:

    ./evaluate.py https://<web_url> --seeds 3
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import time
from datetime import UTC, datetime

import httpx

import transactions as T


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def merchant_ok(got: str | None, expected: str, memo: str) -> bool:
    """Lenient match on the first word; fees, interest and own-account transfers have no merchant.

    For person-to-person payments the person named in the memo is as good a counterparty as the app.
    """
    if expected in ("Zelle", "Venmo") and got and norm(got) in norm(memo):
        return True
    if expected in ("Bank fee", "Interest", "Savings transfer"):
        return got is None or any(w in norm(got) for w in ("fee", "interest", "sav", "transfer"))
    if not got:
        return False
    return norm(expected.split(maxsplit=1)[0])[:5] in norm(got) or norm(got.split(maxsplit=1)[0])[:5] in norm(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    today = datetime.now(tz=UTC).date()
    templates = T.TEMPLATES + T.INCOME_TEMPLATES
    txs = [
        T.make_transaction(t, random.Random(seed * 1000 + i), today)
        for seed in range(args.seeds)
        for i, t in enumerate(templates)
    ]

    merchant_hits = category_hits = 0
    latencies: list[float] = []
    with httpx.Client(timeout=300) as client:
        for i in range(0, len(txs), args.batch):
            chunk = txs[i : i + args.batch]
            t0 = time.perf_counter()
            r = client.post(f"{args.url.rstrip('/')}/api/enrich", json={"transactions": chunk})
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
            for tx in r.json()["transactions"]:
                exp = tx["expected"]
                got_cat = tx["personal_finance_category"]["primary"]
                m_ok = merchant_ok(tx["merchant_name"], exp["merchant"], tx["name"])
                c_ok = got_cat == exp["category"]
                merchant_hits += m_ok
                category_hits += c_ok
                if args.verbose or not (m_ok and c_ok):
                    print(
                        f"{'  ' if m_ok else 'M!'}{'  ' if c_ok else 'C!'} {tx['name']:<48} "
                        f"-> {tx['merchant_name']!s:<22} {got_cat:<26} "
                        f"(expected {exp['merchant']}, {exp['category']})"
                    )
    n = len(txs)
    print(f"\n{n} transactions in batches of {args.batch}")
    print(f"merchant  {merchant_hits}/{n} = {merchant_hits / n:.0%}")
    print(f"category  {category_hits}/{n} = {category_hits / n:.0%}")
    print(f"latency   p50 {statistics.median(latencies):.0f} ms per batch of up to {args.batch}")


if __name__ == "__main__":
    main()
