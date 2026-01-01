"""Tests for pricing module - cost calculation and usage tracking."""


from ccflow.pricing import (
    HAIKU_PRICING,
    OPUS_PRICING,
    SONNET_PRICING,
    ModelPricing,
    UsageStats,
    calculate_cost,
    extract_model_from_events,
    extract_usage_from_events,
    get_pricing,
    get_usage_tracker,
    reset_usage_tracker,
    track_usage,
)

# =============================================================================
# ModelPricing Tests
# =============================================================================


class TestModelPricing:
    """Tests for model pricing calculations."""

    def test_input_cost(self):
        """Should calculate input cost correctly."""
        pricing = ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
        )
        # 1M tokens at $3/million = $3
        assert pricing.input_cost(1_000_000) == 3.0
        # 500K tokens = $1.5
        assert pricing.input_cost(500_000) == 1.5
        # 0 tokens = $0
        assert pricing.input_cost(0) == 0.0

    def test_output_cost(self):
        """Should calculate output cost correctly."""
        pricing = ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
        )
        # 1M tokens at $15/million = $15
        assert pricing.output_cost(1_000_000) == 15.0
        # 100K tokens = $1.5
        assert pricing.output_cost(100_000) == 1.5

    def test_batch_discount(self):
        """Should apply batch discount correctly."""
        pricing = ModelPricing(
            input_per_million=10.0,
            output_per_million=30.0,
            batch_discount=0.5,
        )
        # Normal: $10/M
        assert pricing.input_cost(1_000_000, batch=False) == 10.0
        # Batch: $5/M (50% off)
        assert pricing.input_cost(1_000_000, batch=True) == 5.0

    def test_total_cost(self):
        """Should calculate total cost correctly."""
        pricing = ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
        )
        # 100K input + 20K output
        cost = pricing.total_cost(
            input_tokens=100_000,
            output_tokens=20_000,
        )
        # (100K/1M)*3 + (20K/1M)*15 = 0.3 + 0.3 = 0.6
        assert abs(cost - 0.6) < 0.001

    def test_cache_costs(self):
        """Should include cache costs when provided."""
        pricing = ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cache_write_per_million=3.75,
            cache_read_per_million=0.30,
        )
        cost = pricing.total_cost(
            input_tokens=100_000,
            output_tokens=20_000,
            cache_write_tokens=50_000,
            cache_read_tokens=200_000,
        )
        # Input: 0.3, Output: 0.3, Cache write: 0.1875, Cache read: 0.06
        expected = 0.3 + 0.3 + 0.1875 + 0.06
        assert abs(cost - expected) < 0.001


# =============================================================================
# Pricing Constants Tests
# =============================================================================


class TestPricingConstants:
    """Tests for pricing constant values."""

    def test_sonnet_pricing(self):
        """Sonnet should have correct pricing."""
        assert SONNET_PRICING.input_per_million == 3.0
        assert SONNET_PRICING.output_per_million == 15.0

    def test_haiku_pricing(self):
        """Haiku should have correct pricing."""
        assert HAIKU_PRICING.input_per_million == 0.8
        assert HAIKU_PRICING.output_per_million == 4.0

    def test_opus_pricing(self):
        """Opus should have correct pricing."""
        assert OPUS_PRICING.input_per_million == 15.0
        assert OPUS_PRICING.output_per_million == 75.0


# =============================================================================
# get_pricing Tests
# =============================================================================


class TestGetPricing:
    """Tests for pricing lookup."""

    def test_exact_model_name(self):
        """Should find pricing by exact model name."""
        assert get_pricing("sonnet") == SONNET_PRICING
        assert get_pricing("haiku") == HAIKU_PRICING
        assert get_pricing("opus") == OPUS_PRICING

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert get_pricing("SONNET") == SONNET_PRICING
        assert get_pricing("Haiku") == HAIKU_PRICING
        assert get_pricing("OPUS") == OPUS_PRICING

    def test_partial_match(self):
        """Should match partial model names containing tier."""
        pricing = get_pricing("claude-opus-4")
        assert pricing.input_per_million == 15.0  # Opus pricing

    def test_unknown_model_fallback(self):
        """Should fall back to sonnet for unknown models."""
        pricing = get_pricing("unknown-model-xyz")
        assert pricing == SONNET_PRICING  # Default


# =============================================================================
# calculate_cost Tests
# =============================================================================


class TestCalculateCost:
    """Tests for cost calculation function."""

    def test_calculate_cost_sonnet(self):
        """Should calculate cost for Sonnet model."""
        cost = calculate_cost(
            model="sonnet",
            input_tokens=100_000,
            output_tokens=20_000,
        )
        # (100K/1M)*3 + (20K/1M)*15 = 0.3 + 0.3 = 0.6
        assert abs(cost - 0.6) < 0.001

    def test_calculate_cost_haiku(self):
        """Should calculate cost for Haiku model."""
        cost = calculate_cost(
            model="haiku",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        # (1M/1M)*0.8 + (500K/1M)*4 = 0.8 + 2.0 = 2.8
        assert abs(cost - 2.8) < 0.001

    def test_calculate_cost_opus(self):
        """Should calculate cost for Opus model."""
        cost = calculate_cost(
            model="opus",
            input_tokens=50_000,
            output_tokens=10_000,
        )
        # (50K/1M)*15 + (10K/1M)*75 = 0.75 + 0.75 = 1.5
        assert abs(cost - 1.5) < 0.001

    def test_calculate_cost_batch(self):
        """Should apply batch discount."""
        normal_cost = calculate_cost(
            model="sonnet",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        batch_cost = calculate_cost(
            model="sonnet",
            input_tokens=1_000_000,
            output_tokens=500_000,
            batch=True,
        )
        assert batch_cost == normal_cost * 0.5


# =============================================================================
# UsageStats Tests
# =============================================================================


class TestUsageStats:
    """Tests for usage statistics tracking."""

    def test_initial_stats(self):
        """New stats should be zeroed."""
        stats = UsageStats()
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_cost_usd == 0.0
        assert stats.request_count == 0

    def test_total_tokens(self):
        """Should calculate total tokens correctly."""
        stats = UsageStats(input_tokens=1000, output_tokens=500)
        assert stats.total_tokens == 1500

    def test_add_usage(self):
        """Should add usage and calculate cost."""
        stats = UsageStats()
        cost = stats.add(
            model="sonnet",
            input_tokens=100_000,
            output_tokens=20_000,
        )

        assert cost > 0
        assert stats.input_tokens == 100_000
        assert stats.output_tokens == 20_000
        assert stats.request_count == 1
        assert stats.total_cost_usd == cost

    def test_add_multiple_requests(self):
        """Should accumulate usage from multiple requests."""
        stats = UsageStats()

        stats.add("sonnet", 100_000, 20_000)
        stats.add("sonnet", 50_000, 10_000)

        assert stats.input_tokens == 150_000
        assert stats.output_tokens == 30_000
        assert stats.request_count == 2

    def test_per_model_tracking(self):
        """Should track usage per model."""
        stats = UsageStats()

        stats.add("sonnet", 100_000, 20_000)
        stats.add("haiku", 50_000, 10_000)
        stats.add("sonnet", 25_000, 5_000)

        assert "sonnet" in stats.model_usage
        assert "haiku" in stats.model_usage

        assert stats.model_usage["sonnet"]["input_tokens"] == 125_000
        assert stats.model_usage["sonnet"]["requests"] == 2
        assert stats.model_usage["haiku"]["input_tokens"] == 50_000

    def test_merge_stats(self):
        """Should merge stats correctly."""
        stats1 = UsageStats()
        stats1.add("sonnet", 100_000, 20_000)

        stats2 = UsageStats()
        stats2.add("haiku", 50_000, 10_000)

        stats1.merge(stats2)

        assert stats1.input_tokens == 150_000
        assert stats1.output_tokens == 30_000
        assert stats1.request_count == 2

    def test_to_dict(self):
        """Should convert to dictionary."""
        stats = UsageStats()
        stats.add("sonnet", 100_000, 20_000)

        data = stats.to_dict()

        assert "input_tokens" in data
        assert "output_tokens" in data
        assert "total_tokens" in data
        assert "total_cost_usd" in data
        assert "model_usage" in data

    def test_reset(self):
        """Should reset all statistics."""
        stats = UsageStats()
        stats.add("sonnet", 100_000, 20_000)

        stats.reset()

        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_cost_usd == 0.0
        assert stats.request_count == 0
        assert stats.model_usage == {}


# =============================================================================
# Token Extraction Tests
# =============================================================================


class TestExtractUsage:
    """Tests for extracting usage from CLI events."""

    def test_extract_from_stop_event(self):
        """Should extract usage from stop event."""
        events = [
            {"type": "init", "session_id": "test-123"},
            {"type": "text", "content": "Hello"},
            {
                "type": "stop",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        ]

        usage = extract_usage_from_events(events)

        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 500

    def test_extract_from_result_event(self):
        """Should extract usage from result event."""
        events = [
            {"type": "init"},
            {
                "type": "result",
                "usage": {"input_tokens": 2000, "output_tokens": 800},
                "total_cost_usd": 0.05,
            },
        ]

        usage = extract_usage_from_events(events)

        assert usage["input_tokens"] == 2000
        assert usage["output_tokens"] == 800
        assert usage.get("total_cost_usd") == 0.05

    def test_extract_cache_tokens(self):
        """Should extract cache tokens if present."""
        events = [
            {
                "type": "stop",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 800,
                },
            },
        ]

        usage = extract_usage_from_events(events)

        assert usage.get("cache_write_tokens") == 200
        assert usage.get("cache_read_tokens") == 800

    def test_extract_model_from_init(self):
        """Should extract model from init event."""
        events = [
            {"type": "init", "model": "claude-3-5-sonnet"},
            {"type": "stop", "usage": {}},
        ]

        model = extract_model_from_events(events)
        assert model == "claude-3-5-sonnet"

    def test_extract_model_from_message(self):
        """Should extract model from message event."""
        events = [
            {
                "type": "assistant",
                "message": {"model": "claude-3-opus", "content": "Hello"},
            },
        ]

        model = extract_model_from_events(events)
        assert model == "claude-3-opus"

    def test_extract_model_unknown(self):
        """Should return 'unknown' if model not found."""
        events = [{"type": "text", "content": "Hello"}]
        model = extract_model_from_events(events)
        assert model == "unknown"


# =============================================================================
# Global Usage Tracker Tests
# =============================================================================


class TestGlobalUsageTracker:
    """Tests for global usage tracking functions."""

    def setup_method(self):
        """Reset tracker before each test."""
        reset_usage_tracker()

    def test_get_usage_tracker_singleton(self):
        """Should return singleton tracker."""
        tracker1 = get_usage_tracker()
        tracker2 = get_usage_tracker()
        assert tracker1 is tracker2

    def test_reset_usage_tracker(self):
        """Should reset the global tracker."""
        tracker1 = get_usage_tracker()
        tracker1.add("sonnet", 1000, 500)

        reset_usage_tracker()

        tracker2 = get_usage_tracker()
        assert tracker2.input_tokens == 0
        assert tracker1 is not tracker2

    def test_track_usage_function(self):
        """track_usage should add to global tracker."""
        cost = track_usage("sonnet", 100_000, 20_000)

        tracker = get_usage_tracker()
        assert tracker.input_tokens == 100_000
        assert tracker.output_tokens == 20_000
        assert tracker.total_cost_usd == cost
