"""
LLM API Monetization Proxy — powered by Mainlayer.

This FastAPI application sits in front of any OpenAI-compatible LLM API and
adds per-token billing via Mainlayer.  Clients send standard OpenAI-format
requests plus an X-Payer-Wallet header.  The proxy:

  1. Checks Mainlayer to confirm the wallet has credits.
  2. Forwards the request to the upstream LLM.
  3. Tracks actual token usage back to Mainlayer.
  4. Returns the LLM response verbatim (OpenAI-compatible).

Environment variables (see .env.example):
  MAINLAYER_API_KEY   — Mainlayer API key
  MAINLAYER_RESOURCE_ID — unique resource ID registered in Mainlayer
  LLM_API_KEY         — Upstream LLM API key
  LLM_BASE_URL        — Upstream base URL (default: https://api.openai.com)
  LLM_MODEL           — Default model
  PRICE_PER_1K_TOKENS — USD price per 1,000 tokens (default: 0.01)
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.llm_client import LLMClient, LLMAuthError, LLMNetworkError, LLMRateLimitError
from src.mainlayer import MainlayerClient, MainlayerAuthError, MainlayerNetworkError
from src.models import (
    ChatCompletionResponse,
    ChatRequest,
    HealthResponse,
    PaymentRequiredDetail,
    UsageRecord,
    UsageSummaryResponse,
)
from src.pricing import calculate_request_price, calculate_actual_cost

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION = "1.0.0"

RESOURCE_ID: str = os.getenv("MAINLAYER_RESOURCE_ID", "llm-api-default")
PAY_ENDPOINT: str = "https://api.mainlayer.xyz/pay"

# Whether to fail open (allow requests) if Mainlayer is unreachable.
# Set FAIL_OPEN=true in dev; keep false (fail closed) in production.
FAIL_OPEN: bool = os.getenv("FAIL_OPEN", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Shared client instances (created at startup, reused across requests)
# ---------------------------------------------------------------------------

_mainlayer_client: Optional[MainlayerClient] = None
_llm_client: Optional[LLMClient] = None

# ---------------------------------------------------------------------------
# In-memory usage store (replace with Redis / database in production)
# ---------------------------------------------------------------------------

_usage_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    global _mainlayer_client, _llm_client  # noqa: PLW0603

    logger.info("Starting LLM API Monetization Proxy v%s", VERSION)
    logger.info("Resource ID: %s", RESOURCE_ID)
    logger.info("Upstream LLM: %s", os.getenv("LLM_BASE_URL", "https://api.openai.com"))
    logger.info("Default model: %s", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    logger.info("Fail open: %s", FAIL_OPEN)

    _mainlayer_client = MainlayerClient()
    _llm_client = LLMClient()

    yield

    logger.info("Shutting down LLM API Monetization Proxy")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="LLM API Monetization Proxy",
    description=(
        "A drop-in OpenAI-compatible proxy that adds per-token billing "
        "via Mainlayer. Monetize any LLM API in minutes."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency injectors
# ---------------------------------------------------------------------------


def get_mainlayer() -> MainlayerClient:
    if _mainlayer_client is None:
        raise RuntimeError("MainlayerClient not initialized")
    return _mainlayer_client


def get_llm() -> LLMClient:
    if _llm_client is None:
        raise RuntimeError("LLMClient not initialized")
    return _llm_client


# ---------------------------------------------------------------------------
# Helper: update in-memory usage summary
# ---------------------------------------------------------------------------


def _record_local_usage(wallet: str, tokens: int, cost: float, model: str) -> None:
    if wallet not in _usage_store:
        _usage_store[wallet] = {
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "request_count": 0,
        }
    entry = _usage_store[wallet]
    entry["total_tokens"] += tokens
    entry["total_cost_usd"] = round(entry["total_cost_usd"] + cost, 8)
    entry["request_count"] += 1


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health(llm: LLMClient = Depends(get_llm)) -> HealthResponse:
    """
    Liveness probe.  Also verifies connectivity to the upstream LLM.
    """
    llm_ok = await llm.check_health()
    status = "ok" if llm_ok else "degraded"
    return HealthResponse(
        status=status,
        version=VERSION,
        upstream_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        resource_id=RESOURCE_ID,
    )


@app.get(
    "/usage/summary",
    response_model=UsageSummaryResponse,
    tags=["Usage"],
)
async def usage_summary(
    x_payer_wallet: str = Header(..., description="Wallet address to query"),
) -> UsageSummaryResponse:
    """
    Return the cumulative token usage and cost for a wallet (this session).

    In production, this should query a persistent database or Mainlayer's
    wallet API rather than the in-memory store.
    """
    entry = _usage_store.get(x_payer_wallet, {
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "request_count": 0,
    })
    return UsageSummaryResponse(
        wallet=x_payer_wallet,
        total_tokens=entry["total_tokens"],
        total_cost_usd=entry["total_cost_usd"],
        request_count=entry["request_count"],
    )


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Chat"],
    responses={
        200: {"description": "Successful completion"},
        402: {"description": "Payment required — insufficient Mainlayer credits"},
        429: {"description": "Upstream LLM rate limit exceeded"},
        503: {"description": "Upstream LLM unavailable"},
    },
)
async def chat_completions(
    request: ChatRequest,
    x_payer_wallet: Optional[str] = Header(
        None,
        description=(
            "The payer's Mainlayer wallet address. "
            "Required for billing. Passed as X-Payer-Wallet header."
        ),
    ),
    mainlayer: MainlayerClient = Depends(get_mainlayer),
    llm: LLMClient = Depends(get_llm),
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completion endpoint with Mainlayer billing.

    Drop-in replacement for `POST https://api.openai.com/v1/chat/completions`.
    Add the `X-Payer-Wallet` header to identify the billing account.

    **Flow:**
    1. Validate the payer wallet header.
    2. Check Mainlayer entitlement (credits balance).
    3. Forward to upstream LLM.
    4. Track actual token usage with Mainlayer.
    5. Return the LLM response unchanged.
    """
    request_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # 1. Validate wallet header
    # ------------------------------------------------------------------

    if not x_payer_wallet:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_wallet",
                "message": (
                    "X-Payer-Wallet header is required. "
                    "Provide your Mainlayer wallet address."
                ),
            },
        )

    logger.info(
        "[%s] Chat request wallet=%s model=%s messages=%d",
        request_id,
        x_payer_wallet,
        request.model,
        len(request.messages),
    )

    # ------------------------------------------------------------------
    # 2. Check Mainlayer entitlement
    # ------------------------------------------------------------------

    try:
        entitlement = await mainlayer.check_entitlement(RESOURCE_ID, x_payer_wallet)
    except MainlayerAuthError as exc:
        logger.error("[%s] Mainlayer auth error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail={"error": "mainlayer_auth_error", "message": str(exc)}) from exc
    except MainlayerNetworkError as exc:
        if FAIL_OPEN:
            logger.warning(
                "[%s] Mainlayer unreachable (fail open): %s", request_id, exc
            )
            entitlement = None  # Skip entitlement check
        else:
            logger.error("[%s] Mainlayer unreachable (fail closed): %s", request_id, exc)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "billing_unavailable",
                    "message": "Billing service temporarily unavailable. Try again shortly.",
                },
            ) from exc

    if entitlement is not None and (
        not entitlement.active or entitlement.remaining_credits <= 0
    ):
        pricing = calculate_request_price(request)
        detail = PaymentRequiredDetail(
            resource_id=RESOURCE_ID,
            pricing=pricing,
            pay_endpoint=PAY_ENDPOINT,
            message=(
                f"Insufficient credits. Estimated cost: "
                f"${pricing.cost_usd:.6f} "
                f"({pricing.tokens_estimated} tokens at "
                f"${pricing.price_per_1k_tokens_usd}/1K tokens). "
                "Fund your Mainlayer wallet to continue."
            ),
        )
        logger.info(
            "[%s] Payment required wallet=%s credits=%.4f",
            request_id,
            x_payer_wallet,
            entitlement.remaining_credits,
        )
        raise HTTPException(
            status_code=402,
            detail=detail.model_dump(),
        )

    # ------------------------------------------------------------------
    # 3. Forward to upstream LLM
    # ------------------------------------------------------------------

    try:
        llm_response = await llm.complete(request)
    except LLMAuthError as exc:
        logger.error("[%s] LLM auth error: %s", request_id, exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "llm_auth_error", "message": str(exc)},
        ) from exc
    except LLMRateLimitError as exc:
        logger.warning("[%s] LLM rate limit: %s", request_id, exc)
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit", "message": str(exc)},
        ) from exc
    except LLMNetworkError as exc:
        logger.error("[%s] LLM network error: %s", request_id, exc)
        raise HTTPException(
            status_code=503,
            detail={"error": "llm_unavailable", "message": str(exc)},
        ) from exc

    # ------------------------------------------------------------------
    # 4. Track usage
    # ------------------------------------------------------------------

    total_tokens = llm_response.usage.total_tokens
    actual_cost = calculate_actual_cost(total_tokens)

    usage_record = UsageRecord(
        wallet=x_payer_wallet,
        resource_id=RESOURCE_ID,
        tokens_used=total_tokens,
        cost_usd=actual_cost,
        model=llm_response.model,
        request_id=request_id,
    )

    await mainlayer.track_usage(usage_record)
    _record_local_usage(x_payer_wallet, total_tokens, actual_cost, llm_response.model)

    logger.info(
        "[%s] Completed wallet=%s tokens=%d cost=$%.6f",
        request_id,
        x_payer_wallet,
        total_tokens,
        actual_cost,
    )

    # ------------------------------------------------------------------
    # 5. Return LLM response
    # ------------------------------------------------------------------

    return llm_response


# ---------------------------------------------------------------------------
# Global exception handler — ensures all errors return JSON
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "message": "An unexpected error occurred."},
    )
