"""Tests for TOON integration."""

import json
import pytest

from ccflow.toon_integration import (
    ToonSerializer,
    should_use_toon,
    encode_context,
    TOON_AVAILABLE,
)
from ccflow.types import ToonConfig


class TestToonSerializer:
    """Tests for ToonSerializer class."""

    def test_encode_disabled_returns_json(self):
        """Test that disabled TOON returns JSON."""
        config = ToonConfig(enabled=False)
        serializer = ToonSerializer(config)

        data = {"name": "Alice", "age": 30}
        result = serializer.encode(data)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == data

    def test_encode_simple_object(self):
        """Test encoding simple object."""
        config = ToonConfig(enabled=True)
        serializer = ToonSerializer(config)

        data = {"name": "Alice", "age": 30}
        result = serializer.encode(data)

        # Result should be a string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_for_prompt(self):
        """Test formatting data for prompt injection."""
        config = ToonConfig(enabled=False)  # Use JSON for predictable output
        serializer = ToonSerializer(config)

        data = {"key": "value"}
        result = serializer.format_for_prompt(data, label="Test Data")

        assert "[Test Data]" in result
        assert "```json" in result
        assert '"key"' in result

    def test_format_for_prompt_with_toon_hint(self):
        """Test format includes TOON hint when enabled."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        config = ToonConfig(enabled=True)
        serializer = ToonSerializer(config)

        data = {"items": [{"id": 1}, {"id": 2}]}
        result = serializer.format_for_prompt(data, include_format_hint=True)

        assert "TOON format" in result or "toon" in result

    def test_compression_tracking(self, sample_portfolio_data):
        """Test that compression ratio is tracked."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        config = ToonConfig(enabled=True, track_savings=True)
        serializer = ToonSerializer(config)

        serializer.encode(sample_portfolio_data)

        # After encoding, compression ratio should be set
        # (exact value depends on data)
        ratio = config.last_compression_ratio
        assert isinstance(ratio, float)


class TestShouldUseToon:
    """Tests for should_use_toon heuristic."""

    def test_uniform_array_returns_true(self, sample_orders_data):
        """Test that uniform array returns True."""
        result = should_use_toon(sample_orders_data)
        assert result is True

    def test_small_array_returns_false(self):
        """Test that small arrays return False."""
        data = [{"id": 1}, {"id": 2}]  # Only 2 items
        result = should_use_toon(data)
        assert result is False

    def test_non_uniform_array_returns_false(self):
        """Test that non-uniform array returns False."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2},  # Different keys
            {"id": 3, "name": "Bob", "extra": "field"},
        ]
        result = should_use_toon(data)
        assert result is False

    def test_shallow_dict_returns_true(self):
        """Test that shallow dict returns True."""
        data = {
            "level1": {
                "level2": {
                    "value": 123
                }
            }
        }
        result = should_use_toon(data)
        assert result is True

    def test_deep_nesting_returns_false(self):
        """Test that deeply nested structures return False."""
        data = {
            "l1": {
                "l2": {
                    "l3": {
                        "l4": {
                            "l5": "deep"
                        }
                    }
                }
            }
        }
        result = should_use_toon(data)
        assert result is False

    def test_dict_with_uniform_array_returns_true(self, sample_portfolio_data):
        """Test dict containing uniform array returns True."""
        result = should_use_toon(sample_portfolio_data)
        assert result is True


class TestEncodeContext:
    """Tests for encode_context convenience function."""

    def test_encode_context_basic(self):
        """Test basic context encoding."""
        data = {"key": "value"}
        result = encode_context(data, label="MyContext")

        assert "[MyContext]" in result
        assert "```" in result
        assert "key" in result

    def test_encode_context_with_config(self):
        """Test context encoding with custom config."""
        config = ToonConfig(enabled=False)
        data = {"items": [1, 2, 3]}
        result = encode_context(data, config=config, label="Data")

        assert "[Data]" in result
        assert "```json" in result
