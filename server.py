"""Transaction enrichment: clean raw bank memos with ModernBERT-GLiNER, then categorize with Clef-Flash.

Runs inside the GPU scaling group that deploy.py creates. Two models share one GPU:
  memo cleanup    knowledgator/modern-gliner-bi-large-v1.0, a zero-shot entity extractor with a
                  ModernBERT-large text encoder. It tags the merchant, payment processor,
                  location and the reference/phone/store-number noise in each memo, which
                  yields a short list of merchant candidates.
  decisions       Cloudflare/clef-flash, a 9B decision model. In one forward pass per batch it
                  picks the merchant from those candidates and answers typed questions
                  (category, recurring, essential) with a probability for every option.

HTTP API:
  GET  /                the demo page (index.html)
  GET  /status          model names and readiness
  GET  /api/sample      ?n=12&seed=7 -> synthetic bank-feed transactions
  POST /api/enrich      {"transactions": [...]} -> each transaction with "enrichment" filled in
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

import transactions as T

GLINER_MODEL = os.environ.get("GLINER_MODEL", "knowledgator/modern-gliner-bi-large-v1.0")
GLINER_REVISION = os.environ.get("GLINER_REVISION", "244e68379b54770302fb976059f7782169c9fd54")
CLEF_MODEL = os.environ.get("CLEF_MODEL", "Cloudflare/clef-flash")
CLEF_REVISION = os.environ.get("CLEF_REVISION", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3")
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")

MAX_TRANSACTIONS = 25
GLINER_THRESHOLD = 0.3
# Waiting requests beyond this get a 503 instead of an ever-growing queue on one GPU.
MAX_WAITING = 8

log = logging.getLogger("enrich")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- memo cleanup ---------------------------------------------------------------------------

MERCHANT_LABELS = ("company", "brand")
NOISE_LABELS = ("phone number", "reference number", "date", "city", "state")
GLINER_LABELS = [*MERCHANT_LABELS, "payment processor", "person", "store number", *NOISE_LABELS]
MAX_CANDIDATES = 8
NO_MERCHANT = "none"

US_STATES = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)
# Card-network and ACH boilerplate that carries no meaning on its own.
BOILERPLATE = re.compile(
    r"\b(pos debit|ach debit|ach pmt|checkcard|pending|web id|ppd id|transaction#|ref #|"
    r"online transfer|web pmts|autopay|payment)\b:?",
    re.I,
)
SUFFIXES = re.compile(r"(\.com|\s+(inc|llc|usa|store|us))+$", re.I)
# Card-processor codes that prefix the real merchant ("SQ *", "TST*", "DD *"). Longer names before
# a star, like "Uber *Eats" or "Paypal *Etsy", are kept: they are part of a brand or tagged on their own.
PROCESSOR_CODE = re.compile(r"^\s*(sq|tst|dd|sp|pp|py|in|cke)\s*\*\s*", re.I)


def tidy(name: str) -> str:
    name = PROCESSOR_CODE.sub("", name)
    name = re.sub(r"[*#]+|\.{2,}", " ", name)
    name = re.sub(r"\b(\w+) S\b", r"\1's", name)  # "Trader Joe S" is "Trader Joe's" with the apostrophe dropped
    words = name.split()
    # "Denver Co" is a place, not a business: drop a trailing state code after another word.
    if len(words) >= 2 and words[-1].upper() in US_STATES and len(words[-1]) == 2:
        words = words[:-1]
    name = SUFFIXES.sub("", " ".join(words).strip(" .,-/:"))
    return re.sub(r"\s+", " ", name).strip(" .,-/:")


def merchant_candidates(memo: str, entities: list[dict[str, Any]]) -> list[str]:
    """Strings that might name the merchant, for the decision model to choose between.

    Entities come from the title-cased memo; `str.title()` keeps every character at its index.
    """
    titled = memo.title()
    places = {e["text"].lower() for e in entities if e["label"] in ("city", "state")}
    out: list[str] = []

    def add(text: str) -> None:
        t = tidy(text)
        if len(t) >= 2 and t.lower() not in places and t.lower() not in (c.lower() for c in out):
            out.append(t)

    for e in sorted(entities, key=lambda e: -e["score"]):
        if e["label"] in (*MERCHANT_LABELS, "payment processor", "person"):
            add(e["text"])
        elif e["label"] == "store number":
            # "Safeway #1485" is tagged whole; the words before the number are the store's name.
            add(re.sub(r"#?\s*\d[\w-]*", " ", e["text"]))
    # Whatever survives once every noise span, boilerplate and digit-bearing token is removed.
    keep = list(titled)
    for e in entities:
        if e["label"] in NOISE_LABELS:
            keep[e["start"] : e["end"]] = " " * (e["end"] - e["start"])
    rest = BOILERPLATE.sub(" ", "".join(keep))
    rest = re.sub(r"\S*\d\S*", " ", rest).split()
    if rest:
        add(" ".join(rest[:4]))
        add(" ".join(rest[:2]))
    return out[:MAX_CANDIDATES]


def location_of(memo: str, entities: list[dict[str, Any]]) -> str | None:
    best = {}
    for e in sorted(entities, key=lambda e: -e["score"]):
        best.setdefault(e["label"], e["text"])
    city, state = best.get("city"), best.get("state")
    if state is None and (m := re.search(r"\b([A-Z]{2})\s*$", memo)) and m.group(1) in US_STATES:
        state = m.group(1)
    return ", ".join(x for x in (city, state.upper() if state else None) if x) or None


def spans_for_display(memo: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": e["start"],
            "end": e["end"],
            "label": e["label"],
            "text": memo[e["start"] : e["end"]],
            "score": round(float(e["score"]), 3),
        }
        for e in sorted(entities, key=lambda e: e["start"])
    ]


# --- decisions ------------------------------------------------------------------------------

BASE_QUESTIONS: dict[str, dict[str, Any]] = {
    "category": {
        "type": "choice",
        "instructions": "Which spending category does this bank transaction belong to?",
        "criteria": T.CATEGORIES,
    },
    "recurring": {
        "type": "noul",
        "instructions": "Is this a recurring charge or deposit, such as a subscription, bill, rent, loan payment or paycheck?",
    },
    "essential": {
        "type": "score",
        "instructions": "How essential is this money movement for day-to-day living?",
        "criteria": ["Discretionary", "Somewhat necessary", "Essential"],
    },
}


def questions_for(candidates: list[str]) -> dict[str, dict[str, Any]]:
    """The fixed questions, plus a per-transaction choice of which candidate names the merchant."""
    criteria = {f"c{i}": f"The merchant or counterparty is: {c}" for i, c in enumerate(candidates)}
    criteria[NO_MERCHANT] = (
        "There is no merchant: a bank fee, interest, or a transfer between the account holder's own accounts."
    )
    merchant = {
        "type": "choice",
        "instructions": "Which name in the memo is the business or person on the other side of this transaction?",
        "criteria": criteria,
    }
    return {"merchant": merchant, **BASE_QUESTIONS}


def decision_state(tx: dict[str, Any], location: str | None) -> dict[str, Any]:
    return {
        "memo": tx["name"],
        "location": location,
        "amount_usd": abs(float(tx["amount"])),
        "direction": "money out of the account" if tx["amount"] > 0 else "money into the account",
        "payment_channel": tx.get("payment_channel"),
    }


class Models:
    def __init__(self) -> None:
        super().__init__()
        from gliner import GLiNER

        t0 = time.monotonic()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gliner = GLiNER.from_pretrained(GLINER_MODEL, revision=GLINER_REVISION).to(self.device)
        self.gliner.eval()
        clef_path = snapshot_download(CLEF_MODEL, revision=CLEF_REVISION)
        sys.path.insert(0, clef_path)
        import joint_schema_model as jsm

        self.jsm = jsm
        self.clef, self.processor = jsm.load_release_model(clef_path, device=self.device)
        # One GPU, two models: requests run their GPU work one at a time.
        self.gpu = threading.Lock()
        self.enrich(T.sample(4, 0))
        log.info("models ready on %s in %.1fs", self.device, time.monotonic() - t0)

    @torch.inference_mode()
    def enrich(self, txs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        with self.gpu:
            t0 = time.perf_counter()
            memos = [tx["name"] for tx in txs]
            entities = self.gliner.batch_predict_entities(
                [m.title() for m in memos], GLINER_LABELS, threshold=GLINER_THRESHOLD
            )
            candidates = [merchant_candidates(m, e) for m, e in zip(memos, entities)]
            locations = [location_of(m, e) for m, e in zip(memos, entities)]
            t1 = time.perf_counter()
            tok = self.processor.tokenizer
            questions = [questions_for(c) for c in candidates]
            encoded = [
                self.jsm.encode_record(
                    tok,
                    {"state": decision_state(tx, loc), "questions": q},
                    processor=self.processor,
                )
                for tx, loc, q in zip(txs, locations, questions)
            ]
            batch = self.jsm.collate_records(encoded, tok.pad_token_id, torch.device(self.device))
            logits = self.clef(batch)
            t2 = time.perf_counter()

        out = []
        for tx, memo, ents, cands, loc, qs, enc, record_logits in zip(
            txs, memos, entities, candidates, locations, questions, encoded, logits
        ):
            answers = {
                q.question_id: self.jsm.systemone_answer(
                    qs[q.question_id], dict(zip(q.option_ids, ql.float().softmax(-1).tolist()))
                )
                for q, ql in zip(enc.questions, record_logits)
            }
            pick = answers["merchant"]
            merchant = None if pick["choice"] == NO_MERCHANT else cands[int(pick["choice"][1:])]
            cat = answers["category"]
            ranked = sorted(cat["probabilities"].items(), key=lambda kv: -kv[1])
            out.append(
                {
                    **tx,
                    "merchant_name": merchant,
                    "personal_finance_category": {
                        "primary": cat["choice"],
                        "confidence": cat["confidence"],
                    },
                    "enrichment": {
                        "spans": spans_for_display(memo, ents),
                        "merchant_candidates": cands,
                        "merchant_confidence": pick["confidence"],
                        "location": loc,
                        "category_top3": [{"category": k, "probability": v} for k, v in ranked[:3]],
                        "recurring": answers["recurring"]["noul"],
                        "essential": round(answers["essential"]["score"] / 2, 4),
                        "input_tokens": len(enc.input_ids),
                    },
                }
            )
        return out, {
            "cleanup_ms": round((t1 - t0) * 1000, 1),
            "decisions_ms": round((t2 - t1) * 1000, 1),
        }


# --- app ------------------------------------------------------------------------------------

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # uvicorn accepts no connections until this returns, so readiness waits for warm models.
    state["models"] = await asyncio.to_thread(Models)
    state["waiting"] = 0
    yield


app = FastAPI(lifespan=lifespan)


class EnrichRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_TRANSACTIONS)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "cleanup_model": GLINER_MODEL, "decision_model": CLEF_MODEL})


@app.get("/api/sample")
async def sample(n: int = Query(default=12, ge=1, le=MAX_TRANSACTIONS), seed: int | None = None) -> JSONResponse:
    return JSONResponse({"transactions": T.sample(n, seed)})


@app.post("/api/enrich")
async def enrich(req: EnrichRequest) -> JSONResponse:
    txs = []
    for tx in req.transactions:
        name, amount = tx.get("name"), tx.get("amount")
        if not isinstance(name, str) or not name.strip() or not isinstance(amount, (int, float)):
            raise HTTPException(422, "every transaction needs a non-empty 'name' and a numeric 'amount'")
        txs.append({**tx, "name": name.strip()[:200]})
    if state["waiting"] >= MAX_WAITING:
        raise HTTPException(503, "busy, try again in a moment")
    state["waiting"] += 1
    try:
        t0 = time.perf_counter()
        out, timings = await asyncio.to_thread(state["models"].enrich, txs)
    finally:
        state["waiting"] -= 1
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse({"transactions": out, "timings_ms": timings})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
