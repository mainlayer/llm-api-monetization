"""
Test suite for the LLM API Monetization Proxy.

Tests cover:
  - Pricing calculations
  - Mainlayer entitlement gating
  - Upstream LLM forwarding
  - Usage tracking
  - Error handling
  - Edge cases
  - API health and summary endpoints
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app
from src.mainlayer import MainlayerClient, MainlayerAuthError, MainlayerNetworkError
from src.llm_client import LLMClient, LLMAuthError, LLMNetworkError, LLMRateLimitError
from src.models import (
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponseMessage,
    EntitlementInfo,
    PricingInfo,
    UsageInfo,
    UsageRecord,
)
from src.pricing import (
    calculate_actual_cost,
    calculate_request_price,
    credits_to_usd,
    estimate_tokens,
    tokens_to_credits,
    tokens_to_usd,
    usd_to_credits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chat_request(
    messages: list[dict] | None = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 100,
) -> ChatRequest:
    if messages is None:
        messages = [{"role": "user", "content": "Hello!"}]
    return ChatRequest(
        model=model,
        messages=[ChatMessage(**m) for m in messages],
        max_tokens=max_tokens,
    )


def make_llm_response(
    content: str = "Hello there!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "gpt-4o-mini",
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test123",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_active_entitlement(credits: float = 100.0) -> EntitlementInfo:
    return EntitlementInfo(
        active=True,
        remaining_credits=credits,
        resource_id="llm-api-default",
        wallet="wallet-abc",
    )


def make_empty_entitlement() -> EntitlementInfo:
    return EntitlementInfo(
        active=True,
        remaining_credits=0.0,
        resource_id="llm-api-default",
        wallet="wallet-abc",
    )


def make_inactive_entitlement() -> EntitlementInfo:
    return EntitlementInfo(
        active=False,
        remaining_credits=0.0,
        resource_id="llm-api-default",
        wallet="wallet-abc",
    )


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------


class TestPricing:
    def test_tokens_to_usd_basic(self):
        cost = tokens_to_usd(1000)
        assert cost == pytest.approx(0.01, rel=1e-6)

    def test_tokens_to_usd_zero(self):
        assert tokens_to_usd(0) == 0.0

    def test_tokens_to_usd_fractional(self):
        cost = tokens_to_usd(500)
        assert cost == pytest.approx(0.005, rel=1e-6)

    def test_tokens_to_usd_custom_rate(self):
        cost = tokens_to_usd(1000, price_per_1k=0.05)
        assert cost == pytest.approx(0.05, rel=1e-6)

    def test_tokens_to_usd_large(self):
        cost = tokens_to_usd(100_000)
        assert cost == pytest.approx(1.0, rel=1e-6)

    def test_calculate_actual_cost(self):
        cost = calculate_actual_cost(2000)
        assert cost == pytest.approx(0.02, rel=1e-6)

    def test_estimate_tokens_basic(self):
        request = make_chat_request(messages=[{"role": "user", "content": "Hello"}])
        tokens = estimate_tokens(request)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_long_message(self):
        long_msg = "word " * 500
        request = make_chat_request(messages=[{"role": "user", "content": long_msg}])
        tokens = estimate_tokens(request)
        assert tokens > 100

    def test_calculate_request_price_returns_pricing_info(self):
        request = make_chat_request()
        info = calculate_request_price(request)
        assert isinstance(info, PricingInfo)
        assert info.tokens_estimated > 0
        assert info.cost_usd >= 0
        assert info.price_per_1k_tokens_usd == pytest.approx(0.01, rel=0.1)

    def test_usd_to_credits_default_rate(self):
        credits = usd_to_credits(0.01)
        assert credits == pytest.approx(1.0, rel=1e-4)

    def test_credits_to_usd_roundtrip(self):
        original_usd = 0.50
        credits = usd_to_credits(original_usd)
        recovered = credits_to_usd(credits)
        assert recovered == pytest.approx(original_usd, rel=1e-6)

    def test_tokens_to_credits(self):
        credits = tokens_to_credits(1000)
        assert credits == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# API endpoint tests (using TestClient with mocked dependencies)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mainlayer():
    client = AsyncMock(spec=MainlayerClient)
    return client


@pytest.fixture
def mock_llm():
    client = AsyncMock(spec=LLMClient)
    client.check_health = AsyncMock(return_value=True)
    return client


@pytest.fixture
def api_client(mock_mainlayer, mock_llm):
    """FastAPI test client with mocked Mainlayer and LLM dependencies."""
    from src.main import get_mainlayer, get_llm, _usage_store

    _usage_store.clear()

    app.dependency_overrides[get_mainlayer] = lambda: mock_mainlayer
    app.dependency_overrides[get_llm] = lambda: mock_llm

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_ok(self, api_client, mock_llm):
        mock_llm.check_health = AsyncMock(return_value=True)
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "resource_id" in data

    def test_health_degraded_when_llm_down(self, api_client, mock_llm):
        mock_llm.check_health = AsyncMock(return_value=False)
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


class TestChatCompletions:
    def test_successful_completion(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_mainlayer.track_usage = AsyncMock(return_value=True)
        mock_llm.complete = AsyncMock(return_value=make_llm_response())

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-abc"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == "Hello there!"

    def test_missing_wallet_header_returns_400(self, api_client, mock_mainlayer, mock_llm):
        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 400
        assert "missing_wallet" in response.text

    def test_no_credits_returns_402(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_empty_entitlement()
        )

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-broke"},
        )

        assert response.status_code == 402
        data = response.json()["detail"]
        assert data["error"] == "payment_required"
        assert "pay_endpoint" in data
        assert "pricing" in data

    def test_inactive_entitlement_returns_402(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_inactive_entitlement()
        )

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-inactive"},
        )

        assert response.status_code == 402

    def test_llm_rate_limit_returns_429(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_llm.complete = AsyncMock(
            side_effect=LLMRateLimitError("Rate limit exceeded")
        )

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-abc"},
        )

        assert response.status_code == 429

    def test_llm_network_error_returns_503(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_llm.complete = AsyncMock(
            side_effect=LLMNetworkError("Cannot connect")
        )

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-abc"},
        )

        assert response.status_code == 503

    def test_usage_is_tracked_after_completion(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_mainlayer.track_usage = AsyncMock(return_value=True)
        mock_llm.complete = AsyncMock(
            return_value=make_llm_response(prompt_tokens=20, completion_tokens=10)
        )

        api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-abc"},
        )

        mock_mainlayer.track_usage.assert_called_once()
        record: UsageRecord = mock_mainlayer.track_usage.call_args[0][0]
        assert record.tokens_used == 30
        assert record.wallet == "wallet-abc"

    def test_mainlayer_network_error_fail_closed(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            side_effect=MainlayerNetworkError("Mainlayer down")
        )

        with patch("src.main.FAIL_OPEN", False):
            response = api_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"X-Payer-Wallet": "wallet-abc"},
            )

        assert response.status_code == 503

    def test_response_includes_usage_info(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_mainlayer.track_usage = AsyncMock(return_value=True)
        mock_llm.complete = AsyncMock(return_value=make_llm_response())

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-abc"},
        )

        data = response.json()
        assert "usage" in data
        assert "total_tokens" in data["usage"]

    def test_response_is_openai_compatible(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_mainlayer.track_usage = AsyncMock(return_value=True)
        mock_llm.complete = AsyncMock(return_value=make_llm_response())

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-abc"},
        )

        data = response.json()
        # Must have all standard OpenAI fields
        for field in ("id", "object", "created", "model", "choices", "usage"):
            assert field in data, f"Missing field: {field}"
        assert data["object"] == "chat.completion"

    def test_402_response_includes_pay_endpoint(self, api_client, mock_mainlayer, mock_llm):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_empty_entitlement()
        )

        response = api_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Payer-Wallet": "wallet-broke"},
        )

        detail = response.json()["detail"]
        assert detail["pay_endpoint"] == "https://api.mainlayer.xyz/pay"


class TestUsageSummaryEndpoint:
    def test_usage_summary_zero_for_new_wallet(self, api_client):
        response = api_client.get(
            "/usage/summary",
            headers={"X-Payer-Wallet": "new-wallet-xyz"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_tokens"] == 0
        assert data["total_cost_usd"] == 0.0
        assert data["request_count"] == 0

    def test_usage_summary_accumulates_after_requests(
        self, api_client, mock_mainlayer, mock_llm
    ):
        mock_mainlayer.check_entitlement = AsyncMock(
            return_value=make_active_entitlement()
        )
        mock_mainlayer.track_usage = AsyncMock(return_value=True)
        mock_llm.complete = AsyncMock(
            return_value=make_llm_response(prompt_tokens=10, completion_tokens=5)
        )

        wallet = "wallet-summary-test"

        for _ in range(3):
            api_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"X-Payer-Wallet": wallet},
            )

        response = api_client.get(
            "/usage/summary",
            headers={"X-Payer-Wallet": wallet},
        )

        data = response.json()
        assert data["request_count"] == 3
        assert data["total_tokens"] == 45  # 15 per request × 3

    def test_usage_summary_missing_wallet_returns_422(self, api_client):
        response = api_client.get("/usage/summary")
        assert response.status_code == 422


class TestModels:
    def test_chat_request_serialization(self):
        req = ChatRequest(
            model="gpt-4o-mini",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.model == "gpt-4o-mini"
        assert len(req.messages) == 1

    def test_usage_record_model(self):
        record = UsageRecord(
            wallet="wallet-x",
            resource_id="res-1",
            tokens_used=500,
            cost_usd=0.005,
            model="gpt-4o-mini",
            request_id="req-123",
        )
        assert record.tokens_used == 500
        assert record.cost_usd == pytest.approx(0.005)

    def test_entitlement_defaults(self):
        ent = EntitlementInfo(active=False)
        assert ent.remaining_credits == 0.0
        assert ent.wallet is None
