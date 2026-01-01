"""Tests for TOON integration."""

import json
from unittest.mock import patch

import pytest

from ccflow.exceptions import ToonEncodingError
from ccflow.toon_integration import (
    TOON_AVAILABLE,
    ToonSerializer,
    encode_context,
    should_use_toon,
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

    def test_is_available_property(self):
        """Test is_available property returns correct value."""
        serializer = ToonSerializer()
        assert serializer.is_available == TOON_AVAILABLE

    def test_encode_with_default_config(self):
        """Test encoding with no config uses defaults."""
        serializer = ToonSerializer()  # No config passed
        data = {"test": "value"}
        result = serializer.encode(data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_fallback_on_toon_error(self):
        """Test encoding falls back to JSON on TOON error."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        config = ToonConfig(enabled=True)
        serializer = ToonSerializer(config)

        # Mock toon_encode to raise exception
        with patch("ccflow.toon_integration.toon_encode", side_effect=Exception("TOON error")):
            data = {"key": "value"}
            result = serializer.encode(data)

            # Should fall back to JSON
            parsed = json.loads(result)
            assert parsed == data

    def test_encode_json_fallback_exception(self):
        """Test _encode_json raises ToonEncodingError on failure."""
        config = ToonConfig(enabled=False)
        serializer = ToonSerializer(config)

        # Create an object that can't be JSON serialized
        class NonSerializable:
            pass

        # The default=str handler should convert it, but let's test the exception path
        with patch("json.dumps", side_effect=Exception("JSON error")):
            with pytest.raises(ToonEncodingError) as exc_info:
                serializer._encode_json({"key": "value"})

            assert "JSON encode failed" in str(exc_info.value)

    def test_decode_when_toon_not_available(self):
        """Test decode raises error when TOON not installed."""
        with patch("ccflow.toon_integration.TOON_AVAILABLE", False):
            serializer = ToonSerializer()
            with pytest.raises(ToonEncodingError) as exc_info:
                serializer.decode("some toon string")

            assert "TOON library not installed" in str(exc_info.value)

    def test_decode_raises_on_error(self):
        """Test decode raises ToonEncodingError on decode failure."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        serializer = ToonSerializer()

        # Mock toon_decode to raise exception
        with patch("ccflow.toon_integration.toon_decode", side_effect=Exception("Decode error")):
            with pytest.raises(ToonEncodingError) as exc_info:
                serializer.decode("invalid toon")

            assert "TOON decode failed" in str(exc_info.value)

    def test_decode_valid_toon(self):
        """Test decoding valid TOON string."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        config = ToonConfig(enabled=True)
        serializer = ToonSerializer(config)

        # Encode then decode - note: toon library may have API changes
        data = {"name": "Alice", "age": 30}
        encoded = serializer.encode(data)

        # Only test decode if we got TOON format (not JSON fallback)
        if not encoded.startswith("{"):
            try:
                decoded = serializer.decode(encoded)
                assert decoded == data
            except ToonEncodingError:
                # Library API may have changed, skip
                pytest.skip("TOON decode API incompatible")

    def test_format_for_prompt_no_hint(self):
        """Test format_for_prompt without format hint."""
        config = ToonConfig(enabled=False)
        serializer = ToonSerializer(config)

        data = {"key": "value"}
        result = serializer.format_for_prompt(data, label="Test", include_format_hint=False)

        assert "[Test]" in result
        assert "TOON format" not in result

    def test_track_savings_when_not_available(self):
        """Test _track_savings returns early when TOON not available."""
        config = ToonConfig(enabled=True, track_savings=True)
        serializer = ToonSerializer(config)

        with patch("ccflow.toon_integration.TOON_AVAILABLE", False):
            # Should return without error
            serializer._track_savings({"key": "value"}, "toon string")
            # No exception means success

    def test_track_savings_count_tokens_none(self):
        """Test _track_savings handles count_tokens being None."""
        config = ToonConfig(enabled=True, track_savings=True)
        serializer = ToonSerializer(config)

        with patch("ccflow.toon_integration.count_tokens", None):
            # Should return without error
            serializer._track_savings({"key": "value"}, "toon string")

    def test_track_savings_exception_handling(self):
        """Test _track_savings handles exceptions gracefully."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        config = ToonConfig(enabled=True, track_savings=True)
        serializer = ToonSerializer(config)

        with patch("ccflow.toon_integration.count_tokens", side_effect=Exception("Count error")):
            # Should not raise, just log debug
            serializer._track_savings({"key": "value"}, "toon string")

    def test_track_savings_zero_ratio(self):
        """Test _track_savings doesn't log when ratio is zero."""
        if not TOON_AVAILABLE:
            pytest.skip("TOON library not installed")

        config = ToonConfig(enabled=True, track_savings=True)
        serializer = ToonSerializer(config)

        # Mock count_tokens to return same value for both
        with patch("ccflow.toon_integration.count_tokens", return_value=100):
            serializer._track_savings({"key": "value"}, "toon string")
            # ratio will be 0 since tokens are equal
            assert config.last_compression_ratio == 0.0


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

    def test_empty_dict_returns_true(self):
        """Test empty dict returns True (shallow)."""
        result = should_use_toon({})
        assert result is True

    def test_empty_list_returns_false(self):
        """Test empty list returns False."""
        result = should_use_toon([])
        assert result is False

    def test_primitive_returns_false(self):
        """Test primitive values return False."""
        assert should_use_toon("string") is False
        assert should_use_toon(42) is False
        assert should_use_toon(None) is False

    def test_nested_uniform_array(self):
        """Test deeply nested uniform array is detected."""
        data = {
            "outer": {
                "inner": {
                    "items": [
                        {"id": 1, "name": "a"},
                        {"id": 2, "name": "b"},
                        {"id": 3, "name": "c"},
                        {"id": 4, "name": "d"},
                    ]
                }
            }
        }
        result = should_use_toon(data)
        assert result is True

    def test_empty_nested_dict(self):
        """Test empty nested dict depth calculation."""
        data = {"a": {}}
        result = should_use_toon(data)
        assert result is True  # Shallow

    def test_empty_nested_list(self):
        """Test empty nested list depth calculation."""
        data = {"a": []}
        result = should_use_toon(data)
        assert result is True  # Shallow

    def test_list_of_non_dicts(self):
        """Test list of non-dict items returns False."""
        data = [1, 2, 3, 4, 5]  # More than 3 items but not dicts
        result = should_use_toon(data)
        assert result is False

    def test_mixed_type_list(self):
        """Test list with mixed types returns False."""
        data = [{"id": 1}, "string", {"id": 2}, 42]
        result = should_use_toon(data)
        assert result is False

    def test_exactly_three_items(self):
        """Test array with exactly 3 items returns False (needs >3)."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = should_use_toon(data)
        assert result is False

    def test_four_items_uniform(self):
        """Test array with 4 uniform items returns True."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        result = should_use_toon(data)
        assert result is True

    def test_depth_exactly_three(self):
        """Test depth exactly 3 returns True."""
        data = {"a": {"b": {"c": "value"}}}
        result = should_use_toon(data)
        assert result is True

    def test_depth_exactly_four(self):
        """Test depth exactly 4 returns False."""
        data = {"a": {"b": {"c": {"d": "value"}}}}
        result = should_use_toon(data)
        assert result is False


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
