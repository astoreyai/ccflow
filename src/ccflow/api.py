"""
SDK-Compatible API - High-level interface matching Agent SDK patterns.

Provides query(), batch_query(), and related functions that serve
as drop-in replacements for Claude Agent SDK usage.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import structlog

from ccflow.executor import CLIExecutor, get_executor
from ccflow.parser import StreamParser, collect_text
from ccflow.toon_integration import ToonSerializer
from ccflow.types import (
    CLIAgentOptions,
    InitMessage,
    Message,
    QueryResult,
    StopMessage,
)

logger = structlog.get_logger(__name__)


async def query(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> AsyncIterator[Message]:
    """Execute single query with streaming response.

    Drop-in replacement for claude_agent_sdk.query().
    Routes through CLI with streaming NDJSON parsing.

    Args:
        prompt: The prompt to send to Claude
        options: Configuration options

    Yields:
        Message objects (text, tool_use, tool_result, etc.)

    Example:
        >>> async for msg in query("Explain this code", CLIAgentOptions(model="sonnet")):
        ...     if isinstance(msg, TextMessage):
        ...         print(msg.content, end="")
    """
    opts = options or CLIAgentOptions()
    executor = get_executor()
    parser = StreamParser()
    toon = ToonSerializer(opts.toon)

    # Inject TOON-encoded context if provided
    if opts.context and opts.toon.encode_context:
        context_str = toon.format_for_prompt(opts.context, label="Context")
        existing = opts.append_system_prompt or ""
        opts.append_system_prompt = existing + context_str

    # Build CLI flags
    flags = executor.build_flags(opts)

    logger.info(
        "query_started",
        model=opts.model,
        has_context=opts.context is not None,
        toon_enabled=opts.toon.enabled,
    )

    # Execute and stream
    async for event in executor.execute(
        prompt,
        flags,
        timeout=opts.timeout,
        cwd=opts.cwd,
    ):
        msg = parser.parse_event(event)
        yield msg


async def query_simple(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> str:
    """Execute query and return text result.

    Convenience wrapper that collects all text content
    from the streaming response.

    Args:
        prompt: The prompt to send to Claude
        options: Configuration options

    Returns:
        Concatenated text response
    """
    messages: list[Message] = []
    async for msg in query(prompt, options):
        messages.append(msg)
    return collect_text(messages)


async def batch_query(
    prompts: list[str],
    options: CLIAgentOptions | None = None,
    concurrency: int = 5,
) -> list[QueryResult]:
    """Execute multiple independent queries concurrently.

    Runs prompts in parallel with configurable concurrency limit.
    Each prompt gets its own session.

    Args:
        prompts: List of prompts to process
        options: Shared configuration options
        concurrency: Maximum concurrent executions

    Returns:
        List of QueryResult objects with responses

    Example:
        >>> prompts = ["Review file A", "Review file B", "Review file C"]
        >>> results = await batch_query(prompts, CLIAgentOptions(model="haiku"))
        >>> for r in results:
        ...     print(f"{r.session_id}: {r.result[:100]}...")
    """
    opts = options or CLIAgentOptions()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[QueryResult] = []

    async def process_prompt(prompt: str, index: int) -> QueryResult:
        async with semaphore:
            start_time = time.monotonic()
            session_id = ""
            text_parts: list[str] = []
            input_tokens = 0
            output_tokens = 0
            error: str | None = None

            try:
                async for msg in query(prompt, opts):
                    if isinstance(msg, InitMessage):
                        session_id = msg.session_id
                    elif hasattr(msg, "content"):
                        text_parts.append(msg.content)
                    elif isinstance(msg, StopMessage):
                        input_tokens = msg.usage.get("input_tokens", 0)
                        output_tokens = msg.usage.get("output_tokens", 0)

            except Exception as e:
                error = str(e)
                logger.error("batch_query_error", index=index, error=error)

            duration = time.monotonic() - start_time

            return QueryResult(
                session_id=session_id,
                result="".join(text_parts),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=duration,
                toon_savings_ratio=opts.toon.last_compression_ratio,
                error=error,
            )

    logger.info(
        "batch_query_started",
        prompt_count=len(prompts),
        concurrency=concurrency,
    )

    # Execute all prompts concurrently
    tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    # Log summary
    successful = sum(1 for r in results if r.success)
    total_tokens = sum(r.total_tokens for r in results)
    total_duration = sum(r.duration_seconds for r in results)

    logger.info(
        "batch_query_complete",
        successful=successful,
        failed=len(results) - successful,
        total_tokens=total_tokens,
        total_duration=f"{total_duration:.2f}s",
    )

    return list(results)


async def stream_to_callback(
    prompt: str,
    callback: callable,
    options: CLIAgentOptions | None = None,
) -> QueryResult:
    """Execute query with callback for each message.

    Alternative streaming interface using callbacks instead
    of async iteration.

    Args:
        prompt: The prompt to send to Claude
        callback: Function called for each message
        options: Configuration options

    Returns:
        QueryResult with final statistics
    """
    opts = options or CLIAgentOptions()
    start_time = time.monotonic()
    session_id = ""
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0

    async for msg in query(prompt, opts):
        callback(msg)

        if isinstance(msg, InitMessage):
            session_id = msg.session_id
        elif hasattr(msg, "content"):
            text_parts.append(msg.content)
        elif isinstance(msg, StopMessage):
            input_tokens = msg.usage.get("input_tokens", 0)
            output_tokens = msg.usage.get("output_tokens", 0)

    duration = time.monotonic() - start_time

    return QueryResult(
        session_id=session_id,
        result="".join(text_parts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration,
        toon_savings_ratio=opts.toon.last_compression_ratio,
    )
