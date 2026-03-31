"""
Mainlayer API client.

Mainlayer is the billing and entitlement layer for AI agents.  This client
handles:
  - Checking whether a wallet has sufficient credits to make a request
  - Recording usage after a request completes
  - Surfacing structured errors when payment is required

Base URL: https://api.mainlayer.fr
Auth:     Authorization: Bearer <MAINLAYER_API_KEY>
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

from src.models import EntitlementInfo, UsageRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAINLAYER_BASE_URL: str = os.getenv(
    "MAINLAYER_BASE_URL", "https://api.mainlayer.fr"
).rstrip("/")

MAINLAYER_API_KEY: str = os.getenv("MAINLAYER_API_KEY", "")

# HTTP timeouts (seconds)
_CONNECT_TIMEOUT: float = 5.0
_READ_TIMEOUT: float = 10.0

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MainlayerError(Exception):
    """Base exception for Mainlayer client errors."""


class MainlayerAuthError(MainlayerError):
    """Raised when the API key is missing or rejected."""


class MainlayerNetworkError(MainlayerError):
    """Raised when the Mainlayer API is unreachable."""


class MainlayerUsageTrackingError(MainlayerError):
    """Raised when usage tracking fails (non-fatal, logged and swallowed)."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MainlayerClient:
    """
    Async HTTP client for the Mainlayer entitlement and usage API.

    Usage:
        client = MainlayerClient()

        # 1. Check entitlement before serving the request
        ent = await client.check_entitlement(resource_id, wallet)
        if not ent.active or ent.remaining_credits <= 0:
            raise PaymentRequired(...)

        # 2. Track actual usage after the LLM responds
        await client.track_usage(record)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or MAINLAYER_API_KEY
        self._base_url = (base_url or MAINLAYER_BASE_URL).rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "mainlayer-llm-proxy/1.0",
        }
        self._timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=5.0, pool=5.0
        )

    # ------------------------------------------------------------------
    # Entitlement check
    # ------------------------------------------------------------------

    async def check_entitlement(
        self, resource_id: str, wallet: str
    ) -> EntitlementInfo:
        """
        Check whether *wallet* is entitled to access *resource_id*.

        Calls: GET /entitlement/{resource_id}?wallet={wallet}

        Returns an EntitlementInfo.  If the request fails (network error,
        unexpected status), raises MainlayerError so the caller can decide
        whether to fail open or fail closed.

        Args:
            resource_id: The Mainlayer resource identifier for this API.
            wallet:      The payer's wallet address or identifier.

        Returns:
            EntitlementInfo with active status and remaining credits.

        Raises:
            MainlayerAuthError:    Invalid or missing API key.
            MainlayerNetworkError: Cannot reach Mainlayer API.
            MainlayerError:        Any other Mainlayer error.
        """
        if not self._api_key:
            raise MainlayerAuthError(
                "MAINLAYER_API_KEY is not set. "
                "Set it via environment variable or pass api_key to the client."
            )

        url = f"{self._base_url}/entitlement/{resource_id}"
        params = {"wallet": wallet}

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout
            ) as client:
                response = await client.get(url, params=params)
        except httpx.ConnectError as exc:
            raise MainlayerNetworkError(
                f"Cannot connect to Mainlayer API at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise MainlayerNetworkError(
                f"Mainlayer API timed out: {exc}"
            ) from exc

        if response.status_code == 401:
            raise MainlayerAuthError(
                "Mainlayer API rejected the API key. "
                "Check your MAINLAYER_API_KEY."
            )

        if response.status_code == 404:
            # Resource not configured — treat as inactive
            logger.warning("Resource %s not found in Mainlayer", resource_id)
            return EntitlementInfo(active=False, resource_id=resource_id, wallet=wallet)

        if response.status_code != 200:
            raise MainlayerError(
                f"Mainlayer entitlement check returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        data = response.json()
        return EntitlementInfo(
            active=data.get("active", False),
            remaining_credits=float(data.get("remaining_credits", 0)),
            resource_id=data.get("resource_id", resource_id),
            wallet=data.get("wallet", wallet),
            plan=data.get("plan"),
            reset_at=data.get("reset_at"),
        )

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    async def track_usage(self, record: UsageRecord) -> bool:
        """
        Report actual token usage to Mainlayer after a successful LLM call.

        Calls: POST /usage

        This method intentionally swallows errors and logs them rather than
        raising — a usage tracking failure must not break the client's
        experience.  The host app can implement retry logic on top.

        Args:
            record: UsageRecord describing the completed request.

        Returns:
            True if tracking succeeded, False otherwise.
        """
        url = f"{self._base_url}/usage"
        payload = {
            "wallet": record.wallet,
            "resource_id": record.resource_id,
            "tokens_used": record.tokens_used,
            "cost_usd": record.cost_usd,
            "model": record.model,
            "request_id": record.request_id,
            "timestamp": int(time.time()),
        }

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout
            ) as client:
                response = await client.post(url, json=payload)

            if response.status_code not in (200, 201, 202, 204):
                logger.warning(
                    "Mainlayer usage tracking returned HTTP %d for wallet=%s "
                    "tokens=%d: %s",
                    response.status_code,
                    record.wallet,
                    record.tokens_used,
                    response.text[:200],
                )
                return False

            logger.info(
                "Tracked %d tokens (cost $%.6f) for wallet %s",
                record.tokens_used,
                record.cost_usd,
                record.wallet,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to track usage for wallet=%s: %s",
                record.wallet,
                exc,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Wallet info (optional, used by /usage/summary endpoint)
    # ------------------------------------------------------------------

    async def get_wallet_info(self, wallet: str) -> dict:
        """
        Fetch summary usage data for a wallet from Mainlayer.

        Calls: GET /wallet/{wallet}

        Args:
            wallet: Wallet address or identifier.

        Returns:
            Raw dict from the Mainlayer API.

        Raises:
            MainlayerError on non-200 responses.
        """
        url = f"{self._base_url}/wallet/{wallet}"

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=self._timeout
            ) as client:
                response = await client.get(url)
        except httpx.ConnectError as exc:
            raise MainlayerNetworkError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise MainlayerNetworkError(str(exc)) from exc

        if response.status_code != 200:
            raise MainlayerError(
                f"Mainlayer wallet lookup returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        return response.json()
