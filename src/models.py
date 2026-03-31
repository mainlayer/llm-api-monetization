"""
Pydantic models for OpenAI-compatible request/response structures.
These models allow the LLM API monetization proxy to serve as a drop-in
replacement for the OpenAI API while adding Mainlayer billing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[str, str]


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None

    class Config:
        extra = "allow"  # Forward unknown fields to upstream LLM


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatResponseMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None


# ---------------------------------------------------------------------------
# Mainlayer-specific models
# ---------------------------------------------------------------------------

class EntitlementInfo(BaseModel):
    """Represents the entitlement state returned by the Mainlayer API."""
    active: bool
    remaining_credits: float = 0.0
    resource_id: str = ""
    wallet: Optional[str] = None
    plan: Optional[str] = None
    reset_at: Optional[str] = None


class UsageRecord(BaseModel):
    """A single usage tracking record sent to Mainlayer."""
    wallet: str
    resource_id: str
    tokens_used: int
    cost_usd: float
    model: str
    request_id: str


class PricingInfo(BaseModel):
    """Pricing details returned in payment-required errors."""
    tokens_estimated: int
    cost_usd: float
    price_per_1k_tokens_usd: float = 0.01


class PaymentRequiredDetail(BaseModel):
    """Structured 402 error body."""
    error: str = "payment_required"
    resource_id: str
    pricing: PricingInfo
    pay_endpoint: str = "https://api.mainlayer.xyz/pay"
    message: str = (
        "Insufficient credits. Fund your Mainlayer wallet to continue."
    )


# ---------------------------------------------------------------------------
# Health / info models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    upstream_model: str
    resource_id: str


class UsageSummaryResponse(BaseModel):
    wallet: str
    total_tokens: int
    total_cost_usd: float
    request_count: int
