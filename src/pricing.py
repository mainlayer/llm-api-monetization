"""Pricing calculator: converts token counts to USD costs.

Default rate: $0.01 per 1,000 tokens (configurable via PRICE_PER_1K_TOKENS env var).
All monetary values are expressed in USD.

Key functions:
- estimate_tokens(): Pre-request estimation for 402 responses
- tokens_to_usd(): Token count to USD cost conversion
- calculate_request_price(): Full PricingInfo object for client
- calculate_actual_cost(): Billing calculation from actual token usage
- usd_to_credits(): USD to Mainlayer credit units (100 credits = $1 USD)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.models import ChatRequest, PricingInfo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Price in USD per 1,000 tokens.  Override with PRICE_PER_1K_TOKENS env var.
PRICE_PER_1K_TOKENS_USD: float = float(
    os.getenv("PRICE_PER_1K_TOKENS", "0.01")
)

# Average characters per token (rough approximation used for estimation).
_CHARS_PER_TOKEN: float = 4.0

# Minimum tokens to charge per request (covers overhead).
MIN_TOKENS_PER_REQUEST: int = int(os.getenv("MIN_TOKENS_PER_REQUEST", "10"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_chars_in_request(request: ChatRequest) -> int:
    """Return total character count across all messages in a request."""
    total = 0
    for msg in request.messages:
        if isinstance(msg.content, str):
            total += len(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
    return total


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def estimate_tokens(request: ChatRequest) -> int:
    """Estimate the number of tokens for a request before it is sent upstream.

    This is used to show pricing information in 402 Payment Required responses.
    The estimate is intentionally conservative and includes:
    - Message formatting overhead (~4 tokens per message)
    - Prompt tokens (4 chars per token approximation)
    - Expected completion buffer (max_tokens or 256 default)
    - Minimum charge threshold (MIN_TOKENS_PER_REQUEST)

    Args:
        request : ChatRequest
            The incoming OpenAI-format chat completion request.

    Returns:
        int
            Estimated token count (conservative, may exceed actual usage).

    Note:
        This estimate should be higher than actual usage to ensure adequate
        client funding. Actual token count (from upstream LLM) is used for
        final billing.
    """
    char_count = _count_chars_in_request(request)
    prompt_tokens = max(int(char_count / _CHARS_PER_TOKEN), MIN_TOKENS_PER_REQUEST)

    # Add a completion buffer based on max_tokens or a sensible default.
    max_completion = request.max_tokens or 256
    estimated_total = prompt_tokens + max_completion

    # Apply per-message overhead (~4 tokens each: role, separators, etc.)
    overhead = len(request.messages) * 4
    return estimated_total + overhead


def tokens_to_usd(tokens: int, price_per_1k: Optional[float] = None) -> float:
    """
    Convert a token count to a USD cost.

    Args:
        tokens: Number of tokens consumed.
        price_per_1k: Price per 1,000 tokens in USD.  Defaults to
                      PRICE_PER_1K_TOKENS_USD (from env or module default).

    Returns:
        Cost in USD, rounded to 8 decimal places for precision.
    """
    rate = price_per_1k if price_per_1k is not None else PRICE_PER_1K_TOKENS_USD
    cost = (tokens / 1000.0) * rate
    return round(cost, 8)


def calculate_request_price(request: ChatRequest) -> PricingInfo:
    """
    Build a full PricingInfo object for a request.

    This is called when constructing a 402 Payment Required response so the
    client agent knows exactly what to pay before retrying.

    Args:
        request: The incoming ChatRequest.

    Returns:
        PricingInfo with estimated tokens and cost.
    """
    tokens = estimate_tokens(request)
    cost = tokens_to_usd(tokens)
    return PricingInfo(
        tokens_estimated=tokens,
        cost_usd=cost,
        price_per_1k_tokens_usd=PRICE_PER_1K_TOKENS_USD,
    )


def calculate_actual_cost(total_tokens: int) -> float:
    """
    Calculate the actual USD cost after a request completes.

    Args:
        total_tokens: Actual token count from the upstream LLM response.

    Returns:
        Cost in USD.
    """
    return tokens_to_usd(total_tokens)


# ---------------------------------------------------------------------------
# Credits helpers
# ---------------------------------------------------------------------------

def usd_to_credits(usd: float, credits_per_usd: float = 100.0) -> float:
    """
    Convert a USD amount to Mainlayer credit units.

    Mainlayer uses integer credits internally.  The default conversion is
    100 credits per $1 USD, so $0.01 = 1 credit.

    Args:
        usd: Dollar amount.
        credits_per_usd: Exchange rate.

    Returns:
        Credit amount (may be fractional for display purposes).
    """
    return round(usd * credits_per_usd, 4)


def credits_to_usd(credits: float, credits_per_usd: float = 100.0) -> float:
    """
    Convert Mainlayer credit units to USD.

    Args:
        credits: Credit amount.
        credits_per_usd: Exchange rate.

    Returns:
        USD amount.
    """
    return round(credits / credits_per_usd, 8)


def tokens_to_credits(tokens: int) -> float:
    """
    Convenience function: tokens -> credits directly.

    Args:
        tokens: Number of tokens.

    Returns:
        Credit cost.
    """
    return usd_to_credits(tokens_to_usd(tokens))
