"""Synthetic bank-feed transactions with realistic raw memos.

Each transaction follows the shape of a bank-aggregator transaction object (Plaid-style field
names: `transaction_id`, `name`, `amount`, `payment_channel`, ...), with the enrichment fields
left empty. Every template also records the merchant and category a careful human would assign,
so the pipeline's output can be scored, not just eyeballed.

Amounts follow the aggregator convention: positive is money leaving the account, negative is
money coming in.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

# Primary personal-finance categories, with the description the decision model sees.
CATEGORIES: dict[str, str] = {
    "INCOME": "Paychecks, salary, dividends, interest, tax refunds and other money earned.",
    "TRANSFER_IN": "Money moved in from your own accounts or sent to you by people (Venmo, Zelle, cash deposits).",
    "TRANSFER_OUT": "Money moved out to your own accounts, savings, investments, or sent to people.",
    "LOAN_PAYMENTS": "Credit card payments, car, student, personal loan and mortgage payments.",
    "BANK_FEES": "ATM fees, overdraft, foreign transaction, wire and monthly account fees.",
    "ENTERTAINMENT": "Streaming, music, games, movies, concerts, sporting events and hobbies.",
    "FOOD_AND_DRINK": "Restaurants, coffee shops, fast food, bars, food delivery and groceries.",
    "GENERAL_MERCHANDISE": "Online marketplaces, department, discount, clothing, electronics and other stores.",
    "HOME_IMPROVEMENT": "Hardware stores, furniture, repairs, maintenance and home security.",
    "MEDICAL": "Doctors, dentists, pharmacies, hospitals, eye care and other health care.",
    "PERSONAL_CARE": "Gyms, fitness, hair, beauty, spa and laundry services.",
    "GENERAL_SERVICES": "Insurance, accounting, education, childcare, postage, storage and software services.",
    "GOVERNMENT_AND_NON_PROFIT": "Taxes, government fees, fines and donations to charities.",
    "TRANSPORTATION": "Gas, parking, tolls, public transit, rideshare, taxis and car service.",
    "TRAVEL": "Airlines, hotels, vacation rentals, rental cars and travel agencies.",
    "RENT_AND_UTILITIES": "Rent, electricity, gas, water, internet, phone and cable bills.",
}

STATES = {
    "SAN FRANCISCO": "CA",
    "OAKLAND": "CA",
    "SEATTLE": "WA",
    "AUSTIN": "TX",
    "BROOKLYN": "NY",
    "NEW YORK": "NY",
    "CHICAGO": "IL",
    "DENVER": "CO",
    "PORTLAND": "OR",
    "BOSTON": "MA",
}


@dataclass(frozen=True)
class Template:
    """A memo pattern. `{city}`, `{st}`, `{n4}`, `{n5}`, `{ref}` and `{mmdd}` are filled per draw."""

    memo: str
    merchant: str
    category: str
    low: float
    high: float
    channel: str


TEMPLATES: list[Template] = [
    Template("SQ *BLUE BOTTLE COFFE {city} {st}", "Blue Bottle Coffee", "FOOD_AND_DRINK", 4.5, 14, "in store"),
    Template("TST* TARTINE BAKERY {city} {st}", "Tartine Bakery", "FOOD_AND_DRINK", 8, 42, "in store"),
    Template("DD *DOORDASH CHIPOTLE 855-973-1040 CA", "Chipotle", "FOOD_AND_DRINK", 12, 38, "online"),
    Template("UBER   *EATS PENDING help.uber.com CA", "Uber Eats", "FOOD_AND_DRINK", 15, 55, "online"),
    Template("WHOLEFDS {city3} #{n5} {city} {st}", "Whole Foods Market", "FOOD_AND_DRINK", 18, 160, "in store"),
    Template("TRADER JOE S #{n3} {city} {st}", "Trader Joe's", "FOOD_AND_DRINK", 14, 120, "in store"),
    Template("STARBUCKS STORE {n5} {city} {st}", "Starbucks", "FOOD_AND_DRINK", 3.5, 18, "in store"),
    Template("POS DEBIT {mmdd} SAFEWAY #{n4} {city} {st}", "Safeway", "FOOD_AND_DRINK", 22, 140, "in store"),
    Template("AMZN Mktp US*{ref} Amzn.com/bill WA", "Amazon", "GENERAL_MERCHANDISE", 9, 220, "online"),
    Template("TARGET        000{n5} {city} {st}", "Target", "GENERAL_MERCHANDISE", 12, 190, "in store"),
    Template("PAYPAL *ETSY INC 402-935-7733 CA", "Etsy", "GENERAL_MERCHANDISE", 14, 95, "online"),
    Template("BESTBUY 00{n5} {city} {st}", "Best Buy", "GENERAL_MERCHANDISE", 25, 900, "in store"),
    Template("UNIQLO USA LLC {city} {st}", "Uniqlo", "GENERAL_MERCHANDISE", 20, 140, "in store"),
    Template("THE HOME DEPOT #{n4} {city} {st}", "The Home Depot", "HOME_IMPROVEMENT", 8, 420, "in store"),
    Template("IKEA {city} {st} {ref}", "IKEA", "HOME_IMPROVEMENT", 30, 650, "in store"),
    Template("CVS/PHARMACY #{n5} {city} {st}", "CVS Pharmacy", "MEDICAL", 6, 75, "in store"),
    Template("ONE MEDICAL {ref} 888-663-6331 CA", "One Medical", "MEDICAL", 30, 199, "online"),
    Template("DELTA DENTAL INS PREMIUM", "Delta Dental", "GENERAL_SERVICES", 18, 60, "other"),
    Template("EQUINOX #{n3} {city} {st}", "Equinox", "PERSONAL_CARE", 240, 280, "in store"),
    Template("GREAT CLIPS #{n4} {city} {st}", "Great Clips", "PERSONAL_CARE", 18, 45, "in store"),
    Template("NETFLIX.COM 866-579-7172 CA", "Netflix", "ENTERTAINMENT", 15.49, 22.99, "online"),
    Template("Spotify USA {ref} NY", "Spotify", "ENTERTAINMENT", 11.99, 16.99, "online"),
    Template("STEAMGAMES.COM 4259522985 WA", "Steam", "ENTERTAINMENT", 5, 70, "online"),
    Template("TM *TICKETMASTER 800-653-8000 CA", "Ticketmaster", "ENTERTAINMENT", 45, 380, "online"),
    Template("SHELL OIL 5744{n4} {city} {st}", "Shell", "TRANSPORTATION", 28, 85, "in store"),
    Template("UBER   *TRIP HELP.UBER.COM CA", "Uber", "TRANSPORTATION", 9, 64, "online"),
    Template("LYFT   *RIDE {weekday} {mmdd} LYFT.COM CA", "Lyft", "TRANSPORTATION", 8, 52, "online"),
    Template("CLIPPER SERVICE CONCORD CA", "Clipper", "TRANSPORTATION", 20, 40, "online"),
    Template("FASTRAK CSC 877-229-8655 CA", "FasTrak", "TRANSPORTATION", 10, 60, "online"),
    Template("DELTA AIR 0062{n5}{n5} ATLANTA GA", "Delta Air Lines", "TRAVEL", 180, 860, "online"),
    Template("AIRBNB * {REF} AIRBNB.COM CA", "Airbnb", "TRAVEL", 140, 1200, "online"),
    Template("MARRIOTT {city} DOWNTOWN {city} {st}", "Marriott", "TRAVEL", 160, 640, "in store"),
    Template("PGANDE WEB ONLINE {mmdd} WEB PMTS {ref}", "PG&E", "RENT_AND_UTILITIES", 60, 240, "online"),
    Template("COMCAST CALIFORNIA 800-COMCAST CA", "Comcast", "RENT_AND_UTILITIES", 70, 140, "online"),
    Template("VERIZON WRLS P{n4}-01 {mmdd}", "Verizon", "RENT_AND_UTILITIES", 55, 160, "online"),
    Template("ACH DEBIT AVALON BAY RENT WEB ID: {n5}{n4}", "AvalonBay", "RENT_AND_UTILITIES", 2400, 3600, "other"),
    Template("GEICO *AUTO 800-841-3000 DC", "GEICO", "GENERAL_SERVICES", 95, 210, "online"),
    Template("ADOBE  *CREATIVE CLD 408-536-6000 CA", "Adobe", "GENERAL_SERVICES", 22.99, 59.99, "online"),
    Template("USPS PO {n9} {city} {st}", "USPS", "GENERAL_SERVICES", 7, 45, "in store"),
    Template("CHASE CREDIT CRD AUTOPAY {mmdd} PPD ID: {n9}", "Chase", "LOAN_PAYMENTS", 300, 2500, "other"),
    Template("NAVIENT FUNDING ACH PMT {ref}", "Navient", "LOAN_PAYMENTS", 180, 640, "other"),
    Template("ATM WITHDRAWAL FEE NON-NETWORK", "Bank fee", "BANK_FEES", 2.5, 3.5, "other"),
    Template("FOREIGN TRANSACTION FEE", "Bank fee", "BANK_FEES", 0.5, 9, "other"),
    Template("IRS USATAXPYMT {ref}", "IRS", "GOVERNMENT_AND_NON_PROFIT", 400, 4800, "other"),
    Template(
        "RED CROSS *DONATION 800-733-2767 DC", "American Red Cross", "GOVERNMENT_AND_NON_PROFIT", 25, 250, "online"
    ),
    Template("VENMO PAYMENT {n9} WEB ID: 3264681992", "Venmo", "TRANSFER_OUT", 12, 300, "other"),
    Template(
        "ONLINE TRANSFER TO SAV ...{n4} TRANSACTION#: {n9}", "Savings transfer", "TRANSFER_OUT", 100, 2000, "other"
    ),
]

INCOME_TEMPLATES: list[Template] = [
    Template("ACME CORP PAYROLL PPD ID: {n9}", "Acme Corp", "INCOME", -6400, -3100, "other"),
    Template("INTEREST PAYMENT", "Interest", "INCOME", -18, -0.4, "other"),
    Template("ZELLE FROM {first} {last} ON {mmdd} REF # {REF}", "Zelle", "TRANSFER_IN", -400, -20, "other"),
    Template("VENMO CASHOUT PPD ID: 5264681992", "Venmo", "TRANSFER_IN", -320, -25, "other"),
]

FIRST = ["JAMIE", "PRIYA", "DIEGO", "MEI", "SAM", "OLU", "NORA", "RAVI"]
LAST = ["CHEN", "PATEL", "GARCIA", "KIM", "OKAFOR", "LARSEN", "NGUYEN", "SHAH"]
WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def _fill(memo: str, rng: random.Random, when: date) -> tuple[str, str | None]:
    city = rng.choice(list(STATES))

    def digits(n: int) -> str:
        return "".join(rng.choice(string.digits) for _ in range(n))

    ref = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(9))
    text = memo.format(
        city=city,
        city3=city.replace(" ", "")[:3],
        st=STATES[city],
        n3=digits(3),
        n4=digits(4),
        n5=digits(5),
        n9=digits(9),
        ref=ref,
        REF=ref,
        mmdd=when.strftime("%m/%d"),
        weekday=WEEKDAYS[when.weekday()],
        first=rng.choice(FIRST),
        last=rng.choice(LAST),
    )
    # Card networks truncate the descriptor; long memos lose their tail, as in real feeds.
    return text[:48], (city.title() if "{city}" in memo else None)


def make_transaction(t: Template, rng: random.Random, when: date) -> dict[str, Any]:
    memo, city = _fill(t.memo, rng, when)
    amount = round(rng.uniform(t.low, t.high), 2)
    return {
        "transaction_id": "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(24)),
        "account_id": "BxBXxLj1m4HMXBm9WZZmCWVbPjX16EHwv99vp",
        "amount": amount,
        "iso_currency_code": "USD",
        "date": when.isoformat(),
        "authorized_date": (when - timedelta(days=rng.choice([0, 0, 1, 2]))).isoformat(),
        "name": memo,
        "merchant_name": None,
        "payment_channel": t.channel,
        "pending": False,
        "location": {"city": city, "region": None, "country": "US" if city else None},
        "personal_finance_category": None,
        # Not part of the aggregator schema: what a careful human would label this transaction.
        "expected": {"merchant": t.merchant, "category": t.category},
    }


def sample(n: int, seed: int | None) -> list[dict[str, Any]]:
    """A month-like mix: mostly spending, a paycheck or two, and a few transfers."""
    rng = random.Random(seed)
    today = datetime.now(tz=UTC).date()
    out = []
    for _ in range(n):
        t = rng.choice(INCOME_TEMPLATES) if rng.random() < 0.15 else rng.choice(TEMPLATES)
        out.append(make_transaction(t, rng, today - timedelta(days=rng.randint(0, 30))))
    return sorted(out, key=lambda x: x["date"], reverse=True)
