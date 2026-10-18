# AI Visibility Indexing API

Standalone HTTP service that estimates how often AI answer engines cite a brand for a keyword. There is no UI. `POST /v1/audit` fans each keyword out to OpenAI, Anthropic, Gemini, and Perplexity, then reduces the four judgments into a Vibe-Share Score.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The server listens on `0.0.0.0:8741` by default.

- Interactive reference: `http://127.0.0.1:8741/docs`
- Health: `http://127.0.0.1:8741/health`
- Audit: `POST http://127.0.0.1:8741/v1/audit`

```bash
curl -s http://127.0.0.1:8741/v1/audit \
  -H 'content-type: application/json' \
  -d '{
    "client_domain": "example.com",
    "brand_name": "Example",
    "keywords": ["example pricing", "best widget platforms"]
  }'
```

## Request and response

`AuditRequest` requires `client_domain`, `brand_name`, and `keywords`. A URL in `client_domain` is reduced to its hostname. Blank keywords are rejected. Exact duplicate keywords are removed. The default cap is 15 keywords (`MAX_KEYWORDS`).

`AuditResponse` returns:

- `timestamp` — UTC ISO-8601 time the audit finished
- `keywords_processed` — number of keywords in `results`
- `results` — one object per keyword

Each keyword object includes:

- `vibe_share_score` — `(successful citations / successful provider responses) * 100`, rounded half-up to two decimals
- `successful_citations` — providers that returned a usable judgment and cited the brand or domain
- `total_successful_provider_responses` — providers that returned a usable judgment, cited or not
- `providers` — OpenAI, Anthropic, Gemini, and Perplexity, in that order
- `competitors` — names reported by successful providers, with the target brand removed
- `optimization_payload` — present only when the score is below 50

A provider that times out, returns HTTP 429, returns any other non-200, or returns an unusable payload is stored as `{"provider", "status": "failed", "cited": false}`. That failure is left out of the score denominator and does not fail the HTTP request. There is no retry: repeating a 429 makes the limit worse.

When every provider fails, the denominator is zero and the score is `0.0`. The optimization sentence then names the full provider set:

`Action Required: Brand is currently missing from OpenAI, Anthropic, Gemini, Perplexity search index indices. Review crawl structure.`

A score of exactly 50 does not generate that sentence.

## Provider mode

Outbound calls use one shared `httpx.AsyncClient` with a strict 15-second timeout. Redirects are disabled, and proxy variables from the environment are ignored unless `VISIBILITY_TRUST_ENV=true`.

| `VISIBILITY_PROVIDER_MODE` | Behavior |
| --- | --- |
| `auto` (default) | Call a provider when its API key is set. Otherwise score that provider with the deterministic local auditor. |
| `live` | Always call the provider. A missing key is a failed result. |
| `mock` | Never leave the process. Every provider uses the local auditor. |

The local auditor is a stable hash, not a claim about a live index. If the keyword contains the brand name or the domain's first label, every provider counts the brand as cited. Results are marked `evaluation_source: "local_auditor"`. Live judgments are marked `evaluation_source: "live"`.

Copy `.env.example` to `.env` and export the variables before starting the process. `python main.py` does not load `.env` by itself.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | OpenAI chat completions |
| `ANTHROPIC_API_KEY` | empty | Claude Messages |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | empty | Gemini generateContent. Sent as a header, not a query parameter. |
| `PERPLEXITY_API_KEY` | empty | Perplexity Agent API |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Anthropic model. Sampling parameters are omitted because Haiku 5.5 rejects them. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model. Thinking level defaults to `low`. |
| `PERPLEXITY_MODEL` | `perplexity/sonar` | Agent API model. Web search is forced. |
| `MAX_OUTBOUND_CONCURRENCY` | `16` | Cap on simultaneous provider calls |
| `HOST` / `PORT` | `0.0.0.0` / `8741` | Bind address |
| `LOG_LEVEL` | `INFO` | Log level for the JSON logger |

Live prompts include the brand, domain, and keyword and are sent to the provider you configured. Keys are never written to logs. With a full live set of 15 keywords, budget a client timeout of at least 60 seconds: the in-process fan-out is concurrent, and the 15-second provider cap still applies to each call.

## Billing

After a successful `POST /v1/audit`, a background task reports `keywords_processed` to Stripe as a legacy usage record on the account's metered subscription item. The call is `POST /v1/subscription_items/{id}/usage_records` with `action=increment`, pinned to Stripe API version `2025-02-24.acacia`. That is the supported way to keep using usage records on stripe-python 16, which removed `stripe.UsageRecord`. The idempotency key is `idxgrid_usage_{subscription_item_id}_{timestamp}` so a network retry does not bill the same execution twice.

`POST /v1/webhooks/stripe` verifies `Stripe-Signature` with `stripe.Webhook.construct_event`. `customer.subscription.created`, `customer.subscription.updated`, and `customer.subscription.deleted` write `stripe_customer_id`, `stripe_subscription_id`, `stripe_subscription_item_id`, and `subscription_status` on the Supabase `accounts` table. The table definition is `supabase/accounts.sql`. Match is on `stripe_customer_id`, with a fallback to `metadata.client_domain` when the customer id is not linked yet.

Failed usage reports and failed account updates are stored in `data/billing_registry.sqlite`. `replay_failed_usage_reports()` resubmits them with the original idempotency key and usage timestamp. Set `STRIPE_API_KEY` to a restricted key (`rk_`) that can create usage records. Keep `STRIPE_WEBHOOK_SECRET` and `SUPABASE_SECRET_KEY` in a secrets vault.

If you charge US or EU customers, enable Stripe Tax on the subscription price and confirm an active registration. Stripe calculates no tax until that registration exists. See [Collect taxes for recurring payments](https://docs.stripe.com/billing/taxes/collect-taxes.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests stub HTTP with `httpx.MockTransport`. They do not call the providers.
