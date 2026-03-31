# LLM API Monetization Proxy

**Monetize any LLM API in minutes.** A drop-in OpenAI-compatible proxy that adds per-token billing via [Mainlayer](https://mainlayer.xyz) — the payment infrastructure for AI agents.

---

## What it does

Point your LLM clients at this proxy instead of directly at OpenAI (or any other provider). The proxy:

1. **Checks credits** — verifies the caller's Mainlayer wallet has sufficient credits before forwarding the request.
2. **Forwards the request** — proxies the call to any OpenAI-compatible upstream (OpenAI, Groq, Together AI, Ollama, etc.).
3. **Tracks usage** — reports actual token consumption back to Mainlayer for billing.
4. **Returns the response** — passes the upstream LLM response back to the caller unchanged.

AI agents pay automatically — no invoices, no manual top-ups, no subscriptions.

---

## Pricing model

| Metric | Rate |
|--------|------|
| Tokens | $0.01 per 1,000 tokens |
| Minimum charge | 10 tokens per request |
| Billing granularity | Actual usage (not estimates) |

The rate is configurable via `PRICE_PER_1K_TOKENS` — set your own margin on top of upstream costs.

---

## Quick start

### 1. Clone and configure

```bash
git clone <your-repo>
cd llm-api-monetization
cp .env.example .env
```

Edit `.env`:

```env
MAINLAYER_API_KEY=ml_your_key_here
MAINLAYER_RESOURCE_ID=my-llm-api
LLM_API_KEY=sk-your-openai-key
LLM_BASE_URL=https://api.openai.com
LLM_MODEL=gpt-4o-mini
PRICE_PER_1K_TOKENS=0.01
```

### 2. Run

**Docker (recommended):**

```bash
docker compose up
```

**Local:**

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload
```

The proxy listens on `http://localhost:8000`.

---

## API reference

The proxy exposes an OpenAI-compatible interface. Replace `https://api.openai.com` with `http://localhost:8000` in your client configuration.

### POST /v1/chat/completions

Identical to the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).

**Additional required header:**

| Header | Description |
|--------|-------------|
| `X-Payer-Wallet` | Your Mainlayer wallet address for billing |

**Example request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Payer-Wallet: wallet_your_address" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Explain monads in one sentence."}],
    "max_tokens": 100
  }'
```

**Success response (200):** Standard OpenAI `ChatCompletion` object.

**Payment required (402):**

```json
{
  "detail": {
    "error": "payment_required",
    "resource_id": "my-llm-api",
    "pricing": {
      "tokens_estimated": 120,
      "cost_usd": 0.0012,
      "price_per_1k_tokens_usd": 0.01
    },
    "pay_endpoint": "https://api.mainlayer.xyz/pay",
    "message": "Insufficient credits. Fund your Mainlayer wallet to continue."
  }
}
```

### GET /health

Liveness probe. Returns `200 ok` if the proxy and upstream LLM are healthy.

### GET /usage/summary

Returns cumulative token usage for a wallet in the current session.

```bash
curl http://localhost:8000/usage/summary \
  -H "X-Payer-Wallet: wallet_your_address"
```

---

## How agents pay automatically

AI agents integrate with Mainlayer using the standard flow:

1. **Agent calls the proxy** with its wallet address in the `X-Payer-Wallet` header.
2. **If credits are sufficient**, the request goes through and tokens are deducted.
3. **If credits are low**, the proxy returns `402 Payment Required` with a structured `pay_endpoint` and cost estimate.
4. **The agent calls `https://api.mainlayer.xyz/pay`** to top up its wallet.
5. **Agent retries** — the next call succeeds.

No human in the loop. Fully automated billing for agentic workflows.

---

## Drop-in replacement for OpenAI

Replace the base URL in any existing OpenAI client:

**Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used",  # auth is handled by X-Payer-Wallet
    default_headers={"X-Payer-Wallet": "wallet_your_address"},
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**JavaScript / TypeScript:**

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-used",
  defaultHeaders: { "X-Payer-Wallet": "wallet_your_address" },
});

const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hello!" }],
});
```

**LangChain:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    openai_api_key="not-used",
    model="gpt-4o-mini",
    default_headers={"X-Payer-Wallet": "wallet_your_address"},
)
```

---

## Supported upstream providers

Any OpenAI-compatible API works:

| Provider | LLM_BASE_URL |
|----------|-------------|
| OpenAI | `https://api.openai.com` |
| Groq | `https://api.groq.com/openai` |
| Together AI | `https://api.together.xyz` |
| OpenRouter | `https://openrouter.ai/api` |
| Fireworks | `https://api.fireworks.ai/inference` |
| Ollama (local) | `http://localhost:11434` |
| LM Studio (local) | `http://localhost:1234` |
| vLLM (local) | `http://localhost:8000` |

---

## Configuration reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAINLAYER_API_KEY` | Yes | — | Mainlayer API key |
| `MAINLAYER_RESOURCE_ID` | Yes | `llm-api-default` | Resource ID from Mainlayer dashboard |
| `MAINLAYER_BASE_URL` | No | `https://api.mainlayer.xyz` | Mainlayer API base URL |
| `LLM_API_KEY` | Yes | — | Upstream LLM API key |
| `LLM_BASE_URL` | No | `https://api.openai.com` | Upstream API base URL |
| `LLM_MODEL` | No | `gpt-4o-mini` | Default model |
| `PRICE_PER_1K_TOKENS` | No | `0.01` | USD per 1,000 tokens |
| `MIN_TOKENS_PER_REQUEST` | No | `10` | Minimum tokens to charge |
| `FAIL_OPEN` | No | `false` | Allow requests if Mainlayer is unreachable |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |
| `PORT` | No | `8000` | HTTP listen port |

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite has 25+ tests covering pricing calculations, entitlement gating, LLM forwarding, usage tracking, and error handling.

---

## Architecture

```
Client / Agent
     │
     │  POST /v1/chat/completions
     │  X-Payer-Wallet: wallet_abc
     ▼
┌─────────────────────────────────┐
│     LLM API Monetization Proxy  │
│                                 │
│  1. Validate wallet header      │
│  2. Check Mainlayer entitlement │◄──► Mainlayer API
│  3. Forward to upstream LLM    │◄──► OpenAI / Groq / etc.
│  4. Track token usage           │──►  Mainlayer API
│  5. Return response             │
└─────────────────────────────────┘
```

**Files:**

| File | Purpose |
|------|---------|
| `src/main.py` | FastAPI app, route handlers, dependency injection |
| `src/mainlayer.py` | Mainlayer API client (entitlement + usage tracking) |
| `src/llm_client.py` | Upstream LLM client (OpenAI-compatible) |
| `src/pricing.py` | Token-to-USD pricing calculator |
| `src/models.py` | Pydantic models (OpenAI-compatible + Mainlayer types) |
| `tests/test_api.py` | Full test suite |

---

## License

MIT
