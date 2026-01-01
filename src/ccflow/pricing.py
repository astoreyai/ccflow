"""
Pricing - Model pricing and cost calculation.

Provides accurate cost calculation based on current Anthropic pricing
with support for multiple models and caching tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ModelTier(str, Enum):
    """Model pricing tiers."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a model in USD per million tokens."""

    input_per_million: float
    output_per_million: float

    # Caching pricing (if supported)
    cache_write_per_million: float | None = None
    cache_read_per_million: float | None = None

    # Batch API discount (typically 50% off)
    batch_discount: float = 0.5

    def input_cost(self, tokens: int, *, batch: bool = False) -> float:
        """Calculate input cost for tokens."""
        rate = self.input_per_million
        if batch:
            rate *= self.batch_discount
        return (tokens / 1_000_000) * rate

    def output_cost(self, tokens: int, *, batch: bool = False) -> float:
        """Calculate output cost for tokens."""
        rate = self.output_per_million
        if batch:
            rate *= self.batch_discount
        return (tokens / 1_000_000) * rate

    def total_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        batch: bool = False,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Calculate total cost."""
        cost = self.input_cost(input_tokens, batch=batch)
        cost += self.output_cost(output_tokens, batch=batch)

        if self.cache_write_per_million and cache_write_tokens > 0:
            cost += (cache_write_tokens / 1_000_000) * self.cache_write_per_million

        if self.cache_read_per_million and cache_read_tokens > 0:
            cost += (cache_read_tokens / 1_000_000) * self.cache_read_per_million

        return cost


# =============================================================================
# Current Anthropic Pricing (as of January 2025)
# =============================================================================

# Simple tier-based pricing using latest models
HAIKU_PRICING = ModelPricing(
    input_per_million=0.80,
    output_per_million=4.00,
    cache_write_per_million=1.00,
    cache_read_per_million=0.08,
)

SONNET_PRICING = ModelPricing(
    input_per_million=3.00,
    output_per_million=15.00,
    cache_write_per_million=3.75,
    cache_read_per_million=0.30,
)

OPUS_PRICING = ModelPricing(
    input_per_million=15.00,
    output_per_million=75.00,
    cache_write_per_million=18.75,
    cache_read_per_million=1.50,
)

# Model name to pricing mapping
MODEL_PRICING: dict[str, ModelPricing] = {
    "haiku": HAIKU_PRICING,
    "sonnet": SONNET_PRICING,
    "opus": OPUS_PRICING,
}

# Default pricing if model not found
DEFAULT_PRICING = SONNET_PRICING


def get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model.

    Args:
        model: Model name or alias

    Returns:
        ModelPricing for the model
    """
    model_lower = model.lower().strip()

    # Direct lookup
    if model_lower in MODEL_PRICING:
        return MODEL_PRICING[model_lower]

    # Partial match
    for key, pricing in MODEL_PRICING.items():
        if key in model_lower or model_lower in key:
            return pricing

    # Tier-based fallback
    if "opus" in model_lower:
        return CLAUDE_3_OPUS
    elif "haiku" in model_lower:
        return CLAUDE_3_HAIKU
    elif "sonnet" in model_lower:
        return CLAUDE_3_5_SONNET

    logger.debug("pricing_not_found", model=model, using_default=True)
    return DEFAULT_PRICING


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    batch: bool = False,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Calculate cost for a request.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        batch: Whether this is a batch request
        cache_write_tokens: Tokens written to cache
        cache_read_tokens: Tokens read from cache

    Returns:
        Cost in USD
    """
    pricing = get_pricing(model)
    return pricing.total_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch=batch,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )


# =============================================================================
# Usage Tracking
# =============================================================================


@dataclass
class UsageStats:
    """Aggregated usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0

    # Per-model breakdown
    model_usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Add usage and calculate cost.

        Returns:
            Cost for this request
        """
        cost = calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_write_tokens += cache_write_tokens
        self.cache_read_tokens += cache_read_tokens
        self.total_cost_usd += cost
        self.request_count += 1

        # Track per-model
        if model not in self.model_usage:
            self.model_usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            }

        self.model_usage[model]["input_tokens"] += input_tokens
        self.model_usage[model]["output_tokens"] += output_tokens
        self.model_usage[model]["cost_usd"] += cost
        self.model_usage[model]["requests"] += 1

        return cost

    def merge(self, other: UsageStats) -> None:
        """Merge another UsageStats into this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.total_cost_usd += other.total_cost_usd
        self.request_count += other.request_count

        for model, usage in other.model_usage.items():
            if model not in self.model_usage:
                self.model_usage[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "requests": 0,
                }

            self.model_usage[model]["input_tokens"] += usage["input_tokens"]
            self.model_usage[model]["output_tokens"] += usage["output_tokens"]
            self.model_usage[model]["cost_usd"] += usage["cost_usd"]
            self.model_usage[model]["requests"] += usage["requests"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_cost_usd": self.total_cost_usd,
            "request_count": self.request_count,
            "model_usage": self.model_usage,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        self.total_cost_usd = 0.0
        self.request_count = 0
        self.model_usage.clear()


# =============================================================================
# Token Extraction
# =============================================================================


def extract_usage_from_events(events: list[dict]) -> dict[str, int]:
    """Extract token usage from CLI events.

    Looks for usage info in stop, result, and system messages.

    Args:
        events: List of CLI output events

    Returns:
        Dict with input_tokens, output_tokens, and optionally cache tokens
    """
    usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
    }

    for event in events:
        event_type = event.get("type", "")

        # Check for usage in various event types
        if event_type in ("stop", "result", "system"):
            event_usage = event.get("usage", {})
            if event_usage:
                usage["input_tokens"] = max(
                    usage["input_tokens"], event_usage.get("input_tokens", 0)
                )
                usage["output_tokens"] = max(
                    usage["output_tokens"], event_usage.get("output_tokens", 0)
                )

                # Cache tokens if present
                if "cache_creation_input_tokens" in event_usage:
                    usage["cache_write_tokens"] = event_usage[
                        "cache_creation_input_tokens"
                    ]
                if "cache_read_input_tokens" in event_usage:
                    usage["cache_read_tokens"] = event_usage["cache_read_input_tokens"]

        # Check for total_cost_usd in result events
        if event_type == "result" and "total_cost_usd" in event:
            usage["total_cost_usd"] = event["total_cost_usd"]

        # Extract model from init or message events
        if event_type == "init" and "model" in event:
            usage["model"] = event["model"]
        if "message" in event:
            msg = event["message"]
            if isinstance(msg, dict) and "model" in msg:
                usage["model"] = msg["model"]

    return usage


def extract_model_from_events(events: list[dict]) -> str:
    """Extract model name from CLI events.

    Args:
        events: List of CLI output events

    Returns:
        Model name or "unknown"
    """
    for event in events:
        event_type = event.get("type", "")

        if event_type == "init" and "model" in event:
            return event["model"]

        if "message" in event:
            msg = event["message"]
            if isinstance(msg, dict) and "model" in msg:
                return msg["model"]

    return "unknown"


# =============================================================================
# Global Usage Tracker
# =============================================================================

_global_usage: UsageStats | None = None


def get_usage_tracker() -> UsageStats:
    """Get or create the global usage tracker."""
    global _global_usage
    if _global_usage is None:
        _global_usage = UsageStats()
    return _global_usage


def reset_usage_tracker() -> None:
    """Reset the global usage tracker."""
    global _global_usage
    _global_usage = None


def track_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Track usage in the global tracker.

    Returns:
        Cost for this request
    """
    tracker = get_usage_tracker()
    return tracker.add(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )
