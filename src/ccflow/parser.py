"""
Stream Parser - Parse NDJSON stream into typed Message objects.

Handles real-time parsing of CLI stream-json output,
converting raw JSON events into strongly-typed Message objects.
"""

from __future__ import annotations

from typing import Any

import structlog

from ccflow.exceptions import ParseError
from ccflow.types import (
    AssistantMessage,
    ErrorMessage,
    HookMessage,
    InitMessage,
    Message,
    ResultMessage,
    StopMessage,
    TextMessage,
    ThinkingMessage,
    ToolResultMessage,
    ToolUseMessage,
    UnknownMessage,
    UnknownSystemMessage,
    UserMessage,
)

logger = structlog.get_logger(__name__)


class StreamParser:
    """Parses NDJSON stream into typed Message objects.

    Handles the stream-json output format from Claude CLI,
    discriminating between message types and validating structure.

    Example:
        >>> parser = StreamParser()
        >>> event = {"type": "system", "subtype": "init", "session_id": "abc123"}
        >>> msg = parser.parse_event(event)
        >>> assert isinstance(msg, InitMessage)
    """

    def parse_event(self, event: dict[str, Any]) -> Message:
        """Parse single event dictionary into typed Message.

        Args:
            event: Raw JSON event from CLI stream

        Returns:
            Typed Message object

        Raises:
            ParseError: If event structure is invalid (only for critical issues)
        """
        event_type = event.get("type")

        if event_type == "system":
            return self._parse_system_event(event)
        elif event_type == "message":
            return self._parse_message_event(event)
        elif event_type == "assistant":
            return self._parse_assistant_event(event)
        elif event_type == "user":
            return self._parse_user_event(event)
        elif event_type == "tool_use":
            return self._parse_tool_use_event(event)
        elif event_type == "tool_result":
            return self._parse_tool_result_event(event)
        elif event_type == "result":
            return self._parse_result_event(event)
        elif event_type == "error":
            return self._parse_error_event(event)
        elif event_type == "thinking":
            return self._parse_thinking_event(event)
        else:
            # Handle unknown event types gracefully for forward compatibility
            logger.debug("unknown_event_type", event_type=event_type, raw_event=event)
            return UnknownMessage(event_type=event_type or "unknown", raw_data=event)

    def _parse_system_event(
        self, event: dict[str, Any]
    ) -> InitMessage | StopMessage | HookMessage | UnknownSystemMessage:
        """Parse system event (init, stop, hook_response, etc.)."""
        subtype = event.get("subtype")
        session_id = event.get("session_id", "")

        if subtype == "init":
            return InitMessage(session_id=session_id)
        elif subtype == "stop":
            usage = event.get("usage", {})
            return StopMessage(session_id=session_id, usage=usage)
        elif subtype == "hook_response":
            hook_type = event.get("hook_type", "unknown")
            content = event.get("content")
            return HookMessage(hook_type=hook_type, content=content)
        else:
            # Handle unknown system subtypes gracefully for forward compatibility
            logger.debug("unknown_system_subtype", subtype=subtype, raw_event=event)
            return UnknownSystemMessage(subtype=subtype or "unknown", raw_data=event)

    def _parse_message_event(self, event: dict[str, Any]) -> TextMessage:
        """Parse message/text event."""
        content = event.get("content", "")
        delta_type = event.get("delta_type", "text_delta")

        return TextMessage(content=content, delta_type=delta_type)

    def _parse_assistant_event(self, event: dict[str, Any]) -> AssistantMessage:
        """Parse assistant response event (complete message)."""
        message = event.get("message", {})
        session_id = event.get("session_id", "")

        content = message.get("content", [])
        model = message.get("model", "")
        message_id = message.get("id", "")
        usage = message.get("usage", {})
        stop_reason = message.get("stop_reason")

        return AssistantMessage(
            content=content,
            model=model,
            message_id=message_id,
            session_id=session_id,
            usage=usage,
            stop_reason=stop_reason,
        )

    def _parse_tool_use_event(self, event: dict[str, Any]) -> ToolUseMessage:
        """Parse tool use event."""
        tool = event.get("tool", "")
        args = event.get("args", {})

        if not tool:
            raise ParseError("tool_use event missing tool name")

        return ToolUseMessage(tool=tool, args=args)

    def _parse_tool_result_event(self, event: dict[str, Any]) -> ToolResultMessage:
        """Parse tool result event."""
        tool = event.get("tool", "")
        result = event.get("content", event.get("result", ""))

        return ToolResultMessage(tool=tool, result=result)

    def _parse_error_event(self, event: dict[str, Any]) -> ErrorMessage:
        """Parse error event."""
        message = event.get("message", event.get("error", "Unknown error"))
        code = event.get("code")

        return ErrorMessage(message=message, code=code)

    def _parse_user_event(self, event: dict[str, Any]) -> UserMessage:
        """Parse user message event (typically tool results)."""
        message = event.get("message", {})
        session_id = event.get("session_id", "")

        content = message.get("content", [])
        tool_use_result = event.get("tool_use_result")

        return UserMessage(
            content=content,
            session_id=session_id,
            tool_use_result=tool_use_result,
        )

    def _parse_result_event(self, event: dict[str, Any]) -> ResultMessage:
        """Parse final result summary event."""
        result = event.get("result", "")
        session_id = event.get("session_id", "")
        duration_ms = event.get("duration_ms", 0)
        num_turns = event.get("num_turns", 0)
        total_cost_usd = event.get("total_cost_usd", 0.0)
        usage = event.get("usage", {})
        is_error = event.get("is_error", False)

        return ResultMessage(
            result=result,
            session_id=session_id,
            duration_ms=duration_ms,
            num_turns=num_turns,
            total_cost_usd=total_cost_usd,
            usage=usage,
            is_error=is_error,
        )

    def _parse_thinking_event(self, event: dict[str, Any]) -> ThinkingMessage:
        """Parse thinking/reasoning event from extended thinking mode.

        Thinking events contain the model's internal reasoning process
        when ultrathink mode is enabled.
        """
        content = event.get("content", event.get("thinking", ""))
        thinking_tokens = event.get("thinking_tokens", 0)

        return ThinkingMessage(
            content=content,
            thinking_tokens=thinking_tokens,
        )


# Convenience functions


def parse_event(event: dict[str, Any]) -> Message:
    """Parse single event using default parser.

    Args:
        event: Raw JSON event from CLI stream

    Returns:
        Typed Message object
    """
    return StreamParser().parse_event(event)


def extract_session_id(events: list[dict[str, Any]]) -> str | None:
    """Extract session ID from list of events.

    Looks for the init event which contains the session ID.

    Args:
        events: List of raw JSON events

    Returns:
        Session ID if found, None otherwise
    """
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            return event.get("session_id")
    return None


def extract_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    """Extract token usage from list of events.

    Looks for the stop event which contains usage statistics.

    Args:
        events: List of raw JSON events

    Returns:
        Usage dict with input_tokens and output_tokens
    """
    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "stop":
            return event.get("usage", {"input_tokens": 0, "output_tokens": 0})
    return {"input_tokens": 0, "output_tokens": 0}


def collect_text(messages: list[Message]) -> str:
    """Collect all text content from messages.

    Args:
        messages: List of Message objects

    Returns:
        Concatenated text content
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, TextMessage):
            parts.append(msg.content)
    return "".join(parts)


def collect_thinking(messages: list[Message]) -> str:
    """Collect all thinking content from messages.

    Args:
        messages: List of Message objects

    Returns:
        Concatenated thinking content
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ThinkingMessage):
            parts.append(msg.content)
    return "".join(parts)


def extract_thinking_from_assistant(msg: AssistantMessage) -> str:
    """Extract thinking content from AssistantMessage content blocks.

    Thinking content may be embedded in the content array as blocks
    with type "thinking".

    Args:
        msg: AssistantMessage to extract thinking from

    Returns:
        Concatenated thinking content from content blocks
    """
    parts: list[str] = []
    for block in msg.content:
        if block.get("type") == "thinking":
            parts.append(block.get("thinking", ""))
    return "".join(parts)


def extract_thinking_tokens(events: list[dict[str, Any]]) -> int:
    """Extract thinking token count from events.

    Args:
        events: List of raw JSON events

    Returns:
        Total thinking tokens used
    """
    total = 0
    for event in events:
        # Check for thinking events
        if event.get("type") == "thinking":
            total += event.get("thinking_tokens", 0)
        # Check for usage in stop/result events
        elif (event.get("type") == "system" and event.get("subtype") == "stop") or event.get(
            "type"
        ) == "result":
            usage = event.get("usage", {})
            total += usage.get("thinking_tokens", 0)
    return total
