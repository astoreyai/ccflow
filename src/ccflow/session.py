"""
Session Manager - Stateful multi-turn conversation management.

Maps SDK Session interface to CLI --resume functionality,
tracking conversation state and usage statistics with optional persistence.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from ccflow.events import (
    Event,
    EventEmitter,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionLoadedEvent,
    SessionPersistedEvent,
    SessionResumedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
    get_emitter,
)
from ccflow.executor import CLIExecutor, get_executor
from ccflow.hooks import HookContext, HookEvent, HookRegistry, get_hook_registry
from ccflow.parser import StreamParser
from ccflow.store import SessionState, SessionStatus, SessionStore
from ccflow.toon_integration import ToonSerializer
from ccflow.types import (
    AssistantMessage,
    CLIAgentOptions,
    InitMessage,
    Message,
    ResultMessage,
    SessionStats,
    StopMessage,
    TextMessage,
    ToolResultMessage,
    ToolUseMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


class Session:
    """Stateful session matching Agent SDK Session interface.

    Manages multi-turn conversations via CLI --resume flag,
    tracking usage statistics and supporting session forking.
    Optionally persists state to a SessionStore.

    Example:
        >>> session = Session(options=CLIAgentOptions(model="opus"))
        >>> async for msg in session.send_message("Review this code"):
        ...     print(msg.content, end="")
        >>> async for msg in session.send_message("Focus on security"):
        ...     print(msg.content, end="")
        >>> stats = await session.close()
        >>> print(f"Total tokens: {stats.total_tokens}")

    With persistence:
        >>> from ccflow import SQLiteSessionStore
        >>> store = SQLiteSessionStore()
        >>> session = Session(options=opts, store=store)
        >>> # State auto-persisted after each turn
    """

    def __init__(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
        executor: CLIExecutor | None = None,
        store: SessionStore | None = None,
        emitter: EventEmitter | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        """Initialize session.

        Args:
            session_id: Specific session UUID (auto-generated if None)
            options: CLI agent options
            executor: CLI executor instance (uses default if None)
            store: Optional session store for persistence
            emitter: Optional event emitter (uses global if None)
            hooks: Optional hook registry (uses global if None)
        """
        self._session_id = session_id or str(uuid.uuid4())
        self._options = options or CLIAgentOptions()
        self._executor = executor or get_executor()
        self._parser = StreamParser()
        self._toon = ToonSerializer(self._options.toon)
        self._store = store
        self._emitter = emitter or get_emitter()
        self._hooks = hooks or get_hook_registry()

        # Statistics
        self._turn_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._start_time = time.monotonic()
        self._created_at = datetime.now()
        self._is_closed = False

        # Track conversation for integrity
        self._last_prompt = ""
        self._last_response = ""
        self._messages_hash = ""

        # Track if this is a fresh session or resumed
        self._is_first_message = True

        # Tags for organization
        self._tags: list[str] = []

        logger.info("session_created", session_id=self._session_id)

        # Emit session created event
        self._emit_sync(
            SessionCreatedEvent(
                session_id=self._session_id,
                model=self._options.model,
                tags=self._tags.copy(),
            )
        )

    def _emit_sync(self, event: Event) -> None:
        """Emit event synchronously."""
        if self._emitter is not None:
            self._emitter.emit_sync(event)

    async def _emit(self, event: Event) -> None:
        """Emit event asynchronously."""
        if self._emitter is not None:
            await self._emitter.emit(event)

    @property
    def session_id(self) -> str:
        """Get session UUID."""
        return self._session_id

    @property
    def options(self) -> CLIAgentOptions:
        """Get session options."""
        return self._options

    @property
    def created_at(self) -> datetime:
        """Get session creation time."""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """Get last update time (approximated as now for active sessions)."""
        return datetime.now()

    @property
    def turn_count(self) -> int:
        """Get number of conversation turns."""
        return self._turn_count

    @property
    def is_closed(self) -> bool:
        """Check if session is closed."""
        return self._is_closed

    @property
    def tags(self) -> list[str]:
        """Get session tags."""
        return self._tags.copy()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the session."""
        if tag not in self._tags:
            self._tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the session."""
        if tag in self._tags:
            self._tags.remove(tag)

    def to_state(self) -> SessionState:
        """Convert session to persistable state."""
        return SessionState(
            session_id=self._session_id,
            created_at=self._created_at,
            updated_at=datetime.now(),
            status=SessionStatus.CLOSED if self._is_closed else SessionStatus.ACTIVE,
            model=self._options.model or "sonnet",
            system_prompt=self._options.system_prompt,
            append_system_prompt=self._options.append_system_prompt,
            turn_count=self._turn_count,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_cost_usd=self._total_cost_usd,
            messages_hash=self._messages_hash,
            last_prompt=self._last_prompt,
            last_response=self._last_response,
            tags=self._tags.copy(),
            toon_savings_ratio=self._options.toon.last_compression_ratio,
        )

    async def _persist(self) -> None:
        """Persist current state to store if configured."""
        if self._store is not None:
            try:
                await self._store.save(self.to_state())
                logger.debug("session_persisted", session_id=self._session_id)
                await self._emit(SessionPersistedEvent(session_id=self._session_id))
            except Exception as e:
                logger.warning(
                    "session_persist_failed",
                    session_id=self._session_id,
                    error=str(e),
                )

    async def send_message(
        self,
        content: str,
        context: dict | None = None,
    ) -> AsyncIterator[Message]:
        """Send message and stream response.

        Args:
            content: Message content to send
            context: Optional additional context (TOON-encoded if enabled)

        Yields:
            Message objects from Claude's response

        Raises:
            RuntimeError: If session is closed
        """
        if self._is_closed:
            raise RuntimeError("Cannot send message on closed session")

        turn_start_time = time.monotonic()
        self._turn_count += 1
        self._last_prompt = content
        response_parts: list[str] = []
        turn_input_tokens = 0
        turn_output_tokens = 0

        # Run USER_PROMPT_SUBMIT hook
        # Include context in metadata so hooks can access it (e.g., for emotion adaptation)
        hook_metadata = {"context": context} if context else {}
        prompt_ctx = HookContext(
            session_id=self._session_id,
            hook_event=HookEvent.USER_PROMPT_SUBMIT,
            prompt=content,
            metadata=hook_metadata,
        )
        prompt_ctx = await self._hooks.run(HookEvent.USER_PROMPT_SUBMIT, prompt_ctx)

        # Check if hook blocked the prompt
        if prompt_ctx.metadata.get("blocked"):
            logger.info(
                "prompt_blocked_by_hook",
                session_id=self._session_id,
                reason=prompt_ctx.metadata.get("block_reason"),
            )
            return

        # Allow hooks to modify the prompt
        effective_prompt = prompt_ctx.prompt or content

        # Emit turn started event
        await self._emit(
            TurnStartedEvent(
                session_id=self._session_id,
                turn_number=self._turn_count,
                prompt=effective_prompt,
            )
        )

        # Build options for this turn
        turn_options = CLIAgentOptions(
            model=self._options.model,
            fallback_model=self._options.fallback_model,
            system_prompt=self._options.system_prompt,
            append_system_prompt=self._options.append_system_prompt,
            permission_mode=self._options.permission_mode,
            allowed_tools=self._options.allowed_tools,
            disallowed_tools=self._options.disallowed_tools,
            session_id=self._session_id,
            resume=not self._is_first_message,  # Resume after first message
            max_budget_usd=self._options.max_budget_usd,
            timeout=self._options.timeout,
            cwd=self._options.cwd,
            add_dirs=self._options.add_dirs,
            mcp_servers=self._options.mcp_servers,
            strict_mcp=self._options.strict_mcp,
            toon=self._options.toon,
            verbose=self._options.verbose,
            include_partial=self._options.include_partial,
            ultrathink=self._options.ultrathink,
        )

        # Apply ultrathink prefix if enabled
        effective_content = effective_prompt
        if self._options.ultrathink:
            effective_content = f"ultrathink {effective_prompt}"
            logger.debug("ultrathink_enabled", turn=self._turn_count)

        # Inject context if provided
        if context and self._options.toon.encode_context:
            context_str = self._toon.format_for_prompt(context, label="Turn Context")
            existing = turn_options.append_system_prompt or ""
            turn_options.append_system_prompt = existing + context_str

        # Build CLI flags
        flags = self._executor.build_flags(turn_options)

        # Execute and stream
        async for event in self._executor.execute(
            effective_content,
            flags,
            timeout=self._options.timeout,
            cwd=self._options.cwd,
        ):
            msg = self._parser.parse_event(event)

            # Update session ID from init message if needed
            if isinstance(msg, InitMessage) and self._is_first_message:
                self._session_id = msg.session_id
                logger.debug("session_id_updated", session_id=self._session_id)

            # Collect response text for integrity tracking
            if isinstance(msg, TextMessage):
                response_parts.append(msg.content)
            elif isinstance(msg, AssistantMessage):
                # Extract text from content blocks
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            response_parts.append(text)

            # Run PRE_TOOL_USE hook for tool calls
            if isinstance(msg, ToolUseMessage):
                tool_ctx = HookContext(
                    session_id=self._session_id,
                    hook_event=HookEvent.PRE_TOOL_USE,
                    tool_name=msg.tool,
                    tool_input=msg.args,
                    message=msg,
                )
                tool_ctx = await self._hooks.run(HookEvent.PRE_TOOL_USE, tool_ctx)

                # Check if hook blocked the tool
                if tool_ctx.metadata.get("blocked"):
                    logger.info(
                        "tool_blocked_by_hook",
                        tool=msg.tool,
                        reason=tool_ctx.metadata.get("block_reason"),
                    )
                    # Still yield the message but mark it as blocked
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["blocked"] = True
                    msg.metadata["block_reason"] = tool_ctx.metadata.get("block_reason")

                # Run can_use_tool callback if defined
                elif self._options.can_use_tool is not None:
                    import inspect

                    try:
                        callback = self._options.can_use_tool
                        result = callback(msg.tool, msg.args or {}, tool_ctx)
                        if inspect.iscoroutine(result):
                            result = await result

                        if result.behavior == "deny":
                            if msg.metadata is None:
                                msg.metadata = {}
                            msg.metadata["blocked"] = True
                            msg.metadata["block_reason"] = (
                                result.message or "Denied by permission callback"
                            )
                            logger.info(
                                "tool_blocked_by_callback",
                                tool=msg.tool,
                                reason=result.message,
                            )
                        elif result.updated_input is not None:
                            msg.args = result.updated_input
                    except Exception as e:
                        logger.warning(
                            "can_use_tool_callback_error",
                            tool=msg.tool,
                            error=str(e),
                        )

            # Run POST_TOOL_USE hook for tool results
            if isinstance(msg, ToolResultMessage):
                result_ctx = HookContext(
                    session_id=self._session_id,
                    hook_event=HookEvent.POST_TOOL_USE,
                    tool_name=msg.tool,
                    tool_result=msg.result,
                    message=msg,
                )
                await self._hooks.run(HookEvent.POST_TOOL_USE, result_ctx)

            # Track usage and run STOP hook from stop message
            if isinstance(msg, StopMessage):
                turn_input_tokens = msg.usage.get("input_tokens", 0)
                turn_output_tokens = msg.usage.get("output_tokens", 0)
                self._total_input_tokens += turn_input_tokens
                self._total_output_tokens += turn_output_tokens
                logger.debug(
                    "turn_complete",
                    turn=self._turn_count,
                    input_tokens=turn_input_tokens,
                    output_tokens=turn_output_tokens,
                )

                # Run STOP hook with accumulated response
                response_text = "".join(response_parts)
                stop_ctx = HookContext(
                    session_id=self._session_id,
                    hook_event=HookEvent.STOP,
                    stop_reason=msg.stop_reason,
                    message=msg,
                    metadata={
                        "last_prompt": content,
                        "last_response": response_text,
                    },
                )
                await self._hooks.run(HookEvent.STOP, stop_ctx)

            # Also run STOP hook on ResultMessage (CLI produces 'result' events)
            if isinstance(msg, ResultMessage):
                usage = getattr(msg, "usage", {})
                self._total_input_tokens += usage.get("input_tokens", 0)
                self._total_output_tokens += usage.get("output_tokens", 0)

                # Include accumulated response text in metadata
                response_text = "".join(response_parts)
                stop_ctx = HookContext(
                    session_id=self._session_id,
                    hook_event=HookEvent.STOP,
                    stop_reason="result",
                    message=msg,
                    metadata={
                        "last_prompt": content,
                        "last_response": response_text,
                        "usage": usage,
                        "total_cost_usd": getattr(msg, "total_cost_usd", 0.0),
                        "duration_ms": getattr(msg, "duration_ms", 0),
                    },
                )
                await self._hooks.run(HookEvent.STOP, stop_ctx)

            yield msg

        # Update conversation tracking
        self._last_response = "".join(response_parts)
        self._update_hash()
        self._is_first_message = False

        # Calculate turn duration
        turn_duration = time.monotonic() - turn_start_time

        # Emit turn completed event
        await self._emit(
            TurnCompletedEvent(
                session_id=self._session_id,
                turn_number=self._turn_count,
                prompt=content,
                response=self._last_response,
                input_tokens=turn_input_tokens,
                output_tokens=turn_output_tokens,
                duration_seconds=turn_duration,
                metadata={"model": self._options.model},
            )
        )

        # Persist state after turn completes
        await self._persist()

    def _update_hash(self) -> None:
        """Update messages hash with current turn."""
        import hashlib

        combined = f"{self._messages_hash}:{self._last_prompt}:{self._last_response}"
        self._messages_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def fork(self) -> Session:
        """Create new session branched from current state.

        The forked session starts with the conversation history
        from this session but gets a new session ID.

        Returns:
            New Session instance
        """
        fork_options = CLIAgentOptions(
            model=self._options.model,
            fallback_model=self._options.fallback_model,
            system_prompt=self._options.system_prompt,
            append_system_prompt=self._options.append_system_prompt,
            permission_mode=self._options.permission_mode,
            allowed_tools=self._options.allowed_tools,
            disallowed_tools=self._options.disallowed_tools,
            session_id=self._session_id,
            resume=True,
            fork_session=True,  # This creates new ID from current state
            max_budget_usd=self._options.max_budget_usd,
            timeout=self._options.timeout,
            cwd=self._options.cwd,
            add_dirs=self._options.add_dirs,
            toon=self._options.toon,
        )

        forked = Session(
            session_id=None,  # Will get new ID from CLI
            options=fork_options,
            executor=self._executor,
            store=self._store,  # Share store with forked session
        )

        logger.info(
            "session_forked",
            parent_session=self._session_id,
            child_session=forked.session_id,
        )

        return forked

    async def close(self) -> SessionStats:
        """Close session and return statistics.

        Persists final state with CLOSED status if store is configured.

        Returns:
            SessionStats with usage information
        """
        if self._is_closed:
            logger.warning("session_already_closed", session_id=self._session_id)

        self._is_closed = True
        duration = time.monotonic() - self._start_time

        # Persist final state with CLOSED status
        await self._persist()

        stats = SessionStats(
            session_id=self._session_id,
            total_turns=self._turn_count,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            duration_seconds=duration,
            toon_savings_ratio=self._options.toon.last_compression_ratio,
        )

        # Emit session closed event
        await self._emit(
            SessionClosedEvent(
                session_id=self._session_id,
                model=self._options.model,
                turn_count=self._turn_count,
                total_tokens=stats.total_tokens,
                duration_seconds=duration,
                total_cost_usd=self._total_cost_usd,
            )
        )

        logger.info(
            "session_closed",
            session_id=self._session_id,
            turns=stats.total_turns,
            total_tokens=stats.total_tokens,
            duration_seconds=f"{duration:.2f}",
        )

        return stats


async def resume_session(
    session_id: str,
    options: CLIAgentOptions | None = None,
    store: SessionStore | None = None,
    emitter: EventEmitter | None = None,
) -> Session:
    """Resume an existing session by ID.

    Args:
        session_id: Session UUID to resume
        options: Optional options override
        store: Optional session store for persistence
        emitter: Optional event emitter

    Returns:
        Session instance configured to resume
    """
    opts = options or CLIAgentOptions()
    opts.session_id = session_id
    opts.resume = True

    session = Session(session_id=session_id, options=opts, store=store, emitter=emitter)
    session._is_first_message = False  # Mark as resumed

    # Emit resumed event (override created event)
    emitter_instance = emitter or get_emitter()
    emitter_instance.emit_sync(
        SessionResumedEvent(
            session_id=session_id,
            previous_turn_count=0,  # Unknown without store
        )
    )

    logger.info("session_resumed", session_id=session_id)

    return session


async def load_session(
    session_id: str,
    store: SessionStore,
    options: CLIAgentOptions | None = None,
    emitter: EventEmitter | None = None,
) -> Session | None:
    """Load and resume a session from a store.

    Restores session state including statistics and conversation tracking
    from the persistent store.

    Args:
        session_id: Session UUID to load
        store: Session store to load from
        options: Optional options override (model, prompts, etc.)
        emitter: Optional event emitter

    Returns:
        Session instance with restored state, or None if not found

    Example:
        >>> store = SQLiteSessionStore()
        >>> session = await load_session("abc-123", store)
        >>> if session:
        ...     async for msg in session.send_message("Continue..."):
        ...         print(msg.content, end="")
    """
    state = await store.load(session_id)
    if state is None:
        logger.warning("session_not_found", session_id=session_id)
        return None

    # Build options from stored state or provided override
    opts = options or CLIAgentOptions()
    opts.session_id = session_id
    opts.resume = True

    # Restore model and prompts from state if not overridden
    if opts.model is None:
        opts.model = state.model
    if opts.system_prompt is None:
        opts.system_prompt = state.system_prompt
    if opts.append_system_prompt is None:
        opts.append_system_prompt = state.append_system_prompt

    # Create session with restored state
    session = Session(session_id=session_id, options=opts, store=store, emitter=emitter)

    # Restore statistics
    session._turn_count = state.turn_count
    session._total_input_tokens = state.total_input_tokens
    session._total_output_tokens = state.total_output_tokens
    session._total_cost_usd = state.total_cost_usd
    session._created_at = state.created_at
    session._messages_hash = state.messages_hash
    session._last_prompt = state.last_prompt
    session._last_response = state.last_response
    session._tags = state.tags.copy()
    session._is_first_message = False  # Mark as resumed

    # Emit loaded event
    emitter_instance = emitter or get_emitter()
    await emitter_instance.emit(
        SessionLoadedEvent(
            session_id=session_id,
            turn_count=state.turn_count,
            total_tokens=state.total_tokens,
        )
    )

    logger.info(
        "session_loaded",
        session_id=session_id,
        turn_count=state.turn_count,
        total_tokens=state.total_tokens,
    )

    return session
