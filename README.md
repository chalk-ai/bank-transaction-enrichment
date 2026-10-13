# Transaction enrichment on Chalk GPUs

Turn raw bank memos like `SQ *BLUE BOTTLE COFFE OAKLAND CA` into a clean merchant, a location and
a spending category, with calibrated probabilities. Two open models run in one Chalk scaling
group on a single GPU:

| Step | Model | What it does |
| --- | --- | --- |
| Memo cleanup | [`knowledgator/modern-gliner-bi-large-v1.0`](https://huggingface.co/knowledgator/modern-gliner-bi-large-v1.0) | A zero-shot entity extractor with a ModernBERT-large text encoder. It tags the company, payment processor, person, city, state, and the reference, phone and store-number noise in each memo. Those spans become a short list of merchant candidates. |
| Decisions | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) | A 9B decision model. For each transaction it picks the merchant from the candidates (or "no merchant" for fees and own-account transfers) and answers typed questions: one of 16 spending categories, whether it recurs, and how essential it is. Every option gets a probability, from one forward pass per batch. |

```
raw memo ─▶ GLiNER (ModernBERT) ─▶ spans + merchant candidates ─▶ Clef-Flash ─▶ merchant, category, recurring, essential
```

The demo feeds it synthetic transactions in the common bank-aggregator format (the field names
Plaid uses: `name`, `amount`, `payment_channel`, `merchant_name`, `personal_finance_category`),
and fills in `merchant_name` and `personal_finance_category`.

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Builds the image and deploys a `chalkcompute.ScalingGroup` on one L40S. |
| `server.py` | FastAPI app: memo cleanup, decisions, `/api/sample`, `/api/enrich` and the page. |
| `transactions.py` | Synthetic transactions with realistic memos, each with the merchant and category a careful human would assign. |
| `index.html` | The demo page: raw memos with tagged spans, and the enrichment beside them. |
| `evaluate.py` | Scores a deployed service against the generator's labels. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule a 48 GB GPU
(Clef-Flash is about 19 GB in bf16). Credentials come from `chalk config` in the current directory,
or from `CHALK_API_SERVER`, `CHALK_CLIENT_ID`, `CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID`.

```bash
./deploy.py --gpu nvidia-l40s
./evaluate.py https://<web_url> --seeds 3
```

The replica downloads Clef-Flash at startup, so a cold start takes a few minutes; the GLiNER
models are baked into the image.

```bash
curl -s https://<web_url>/api/enrich -H 'content-type: application/json' -d '{
  "transactions": [{"name": "TST* TARTINE BAKERY SAN FRANCISCO CA", "amount": 18.5, "payment_channel": "in store"}]
}'
```

## Results

On 153 synthetic transactions (every memo template, three random fills each), on one L40S:

| | |
| --- | --- |
| Merchant matches the label | 95% |
| Category matches the label | 93% |
| Latency | about 1.9 s for 10 transactions, 5.4 s for 25, in one batch |

The misses are informative: `UBER *EATS` lands in transportation, `PGANDE` (PG&E) is unreadable
to both models, and abbreviations like `AMZN` and `WHOLEFDS` are extracted but not expanded, since
cleanup picks a span from the memo rather than generating a new name.

## Notes

- The `pytorch/pytorch:2.11` image uses the system Python, which refuses `pip install` (PEP 668),
  so the image sets `PIP_BREAK_SYSTEM_PACKAGES=1`.
- GLiNER's bi-encoder loads its label encoder (`BAAI/bge-base-en-v1.5`) and ModernBERT's config by
  name. The weights are in the GLiNER checkpoint, so only those repos' JSON and tokenizer files are
  fetched.
- Clef-Flash's loader needs `accelerate`, and its processor needs `torchvision`.
- Delete the group when you are done:
  `uv run --with chalkcompute python -c "from chalkcompute import ScalingGroup; ScalingGroup.from_name('transaction-enrichment').delete()"`

## License

Apache-2.0. Clef-Flash and modern-gliner-bi-large-v1.0 are both released under Apache-2.0.
