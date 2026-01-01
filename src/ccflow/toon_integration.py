"""
TOON Integration - Token-optimized serialization for context injection.

Wraps the official toon-format library to provide:
- Automatic encoding of context objects
- Compression metric tracking
- Prompt formatting for CLI injection

See docs/TOON_RESEARCH.md for format specification.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from ccflow.exceptions import ToonEncodingError
from ccflow.types import ToonConfig

logger = structlog.get_logger(__name__)

# Try to import toon-format library
try:
    from toon_format import encode as toon_encode
    from toon_format import decode as toon_decode
    from toon_format import count_tokens

    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False
    toon_encode = None  # type: ignore
    toon_decode = None  # type: ignore
    count_tokens = None  # type: ignore
    logger.warning(
        "toon_format_not_installed",
        message="Install with: pip install git+https://github.com/toon-format/toon-python.git",
    )


class ToonSerializer:
    """Wrapper around toon-format library for middleware integration.

    Provides automatic TOON encoding with compression tracking
    and fallback to JSON when TOON is unavailable or disabled.

    Example:
        >>> config = ToonConfig(enabled=True, track_savings=True)
        >>> serializer = ToonSerializer(config)
        >>> toon_str = serializer.encode({"users": [{"id": 1}, {"id": 2}]})
        >>> print(f"Saved {config.last_compression_ratio:.1%} tokens")
    """

    def __init__(self, config: ToonConfig | None = None) -> None:
        """Initialize serializer.

        Args:
            config: TOON configuration options
        """
        self.config = config or ToonConfig()

    @property
    def is_available(self) -> bool:
        """Check if TOON library is installed."""
        return TOON_AVAILABLE

    def encode(self, data: Any) -> str:
        """Encode Python object to TOON format.

        Falls back to JSON if TOON is disabled or unavailable.

        Args:
            data: Python object to encode

        Returns:
            TOON-formatted string (or JSON fallback)

        Raises:
            ToonEncodingError: If encoding fails
        """
        # Fallback to JSON if disabled or unavailable
        if not self.config.enabled or not TOON_AVAILABLE:
            return self._encode_json(data)

        try:
            options = {
                "delimiter": self.config.delimiter,
                "indent": self.config.indent,
                "lengthMarker": "#" if self.config.length_marker else "",
            }

            toon_str = toon_encode(data, options)

            if self.config.track_savings:
                self._track_savings(data, toon_str)

            logger.debug(
                "toon_encoded",
                length=len(toon_str),
                savings_ratio=self.config.last_compression_ratio,
            )

            return toon_str

        except Exception as e:
            logger.warning("toon_encode_failed", error=str(e), fallback="json")
            # Fallback to JSON on encoding errors
            return self._encode_json(data)

    def decode(self, toon_str: str) -> Any:
        """Decode TOON format to Python object.

        Args:
            toon_str: TOON-formatted string

        Returns:
            Decoded Python object

        Raises:
            ToonEncodingError: If decoding fails
        """
        if not TOON_AVAILABLE:
            raise ToonEncodingError(
                "TOON library not installed. Cannot decode TOON format."
            )

        try:
            return toon_decode(toon_str, {"strict": True})
        except Exception as e:
            raise ToonEncodingError(f"TOON decode failed: {e}") from e

    def format_for_prompt(
        self,
        data: Any,
        label: str = "Context",
        include_format_hint: bool = True,
    ) -> str:
        """Format data for injection into system prompt.

        Creates a formatted block suitable for appending to system prompts,
        with optional format hint for the model.

        Args:
            data: Data to encode
            label: Label for the data block
            include_format_hint: Include hint about TOON format

        Returns:
            Formatted string for prompt injection
        """
        encoded = self.encode(data)
        format_type = "toon" if (self.config.enabled and TOON_AVAILABLE) else "json"

        parts = [f"\n[{label}]"]

        if include_format_hint and format_type == "toon":
            parts.append(
                "(Data encoded in TOON format for token efficiency. "
                "Arrays use [N]{fields}: header with comma-separated rows.)"
            )

        parts.append(f"```{format_type}")
        parts.append(encoded)
        parts.append("```")

        return "\n".join(parts)

    def _encode_json(self, data: Any) -> str:
        """Fallback JSON encoding."""
        try:
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            raise ToonEncodingError(f"JSON encode failed: {e}") from e

    def _track_savings(self, data: Any, toon_str: str) -> None:
        """Track token savings from TOON encoding."""
        if not TOON_AVAILABLE or count_tokens is None:
            return

        try:
            json_str = json.dumps(data, default=str)
            self.config._last_json_tokens = count_tokens(json_str)
            self.config._last_toon_tokens = count_tokens(toon_str)

            ratio = self.config.last_compression_ratio
            if ratio > 0:
                logger.info(
                    "toon_savings",
                    json_tokens=self.config._last_json_tokens,
                    toon_tokens=self.config._last_toon_tokens,
                    savings_percent=f"{ratio:.1%}",
                )
        except Exception as e:
            logger.debug("token_counting_failed", error=str(e))


def should_use_toon(data: Any) -> bool:
    """Heuristic for TOON applicability.

    TOON works best on uniform arrays of objects. This function
    checks if the data structure is suitable for TOON encoding.

    Args:
        data: Data to check

    Returns:
        True if TOON is likely beneficial
    """
    # Helper to calculate max nesting depth
    def max_depth(obj: Any, depth: int = 0) -> int:
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(max_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(max_depth(v, depth + 1) for v in obj)
        return depth

    # Check for uniform arrays (TOON's sweet spot)
    def has_uniform_array(obj: Any) -> bool:
        if isinstance(obj, list) and len(obj) > 3:
            if all(isinstance(item, dict) for item in obj):
                keys = [frozenset(item.keys()) for item in obj]
                if len(set(keys)) == 1:
                    return True
        if isinstance(obj, dict):
            for value in obj.values():
                if has_uniform_array(value):
                    return True
        return False

    # If data contains uniform arrays, TOON is beneficial
    if has_uniform_array(data):
        return True

    # For non-array data, only use TOON if shallow (depth <= 3)
    if isinstance(data, dict):
        depth = max_depth(data)
        if depth <= 3:
            return True

    return False


# Module-level convenience functions


def encode_context(
    data: Any,
    config: ToonConfig | None = None,
    label: str = "Context",
) -> str:
    """Encode data for prompt context injection.

    Convenience function for one-off encoding without managing
    a serializer instance.

    Args:
        data: Data to encode
        config: Optional TOON configuration
        label: Label for the context block

    Returns:
        Formatted string for prompt injection
    """
    serializer = ToonSerializer(config)
    return serializer.format_for_prompt(data, label)
