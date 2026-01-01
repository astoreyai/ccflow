#!/usr/bin/env python3
"""
Event Handling Example

Demonstrates the event system for observability and custom integrations.
"""

import asyncio
from datetime import datetime

from ccflow import Session, CLIAgentOptions
from ccflow.events import (
    EventEmitter,
    EventType,
    Event,
    SessionCreatedEvent,
    SessionClosedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    TokensUsedEvent,
    CostIncurredEvent,
    ToolCalledEvent,
    get_emitter,
)


class MetricsCollector:
    """Custom metrics collector using events."""

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.sessions_created = 0
        self.turns_completed = 0
        self.tool_calls = []

    def on_tokens(self, event: TokensUsedEvent):
        self.total_tokens += event.input_tokens + event.output_tokens
        print(f"  [Metrics] Tokens: +{event.input_tokens + event.output_tokens} (total: {self.total_tokens})")

    def on_cost(self, event: CostIncurredEvent):
        self.total_cost += event.cost_usd
        print(f"  [Metrics] Cost: +${event.cost_usd:.4f} (total: ${self.total_cost:.4f})")

    def on_session_created(self, event: SessionCreatedEvent):
        self.sessions_created += 1
        print(f"  [Metrics] Session created: {event.session_id[:8]}...")

    def on_turn_completed(self, event: TurnCompletedEvent):
        self.turns_completed += 1
        print(f"  [Metrics] Turn {event.turn_number} completed in {event.duration_ms}ms")

    def on_tool_called(self, event: ToolCalledEvent):
        self.tool_calls.append(event.tool)
        print(f"  [Metrics] Tool called: {event.tool}")


class AuditLogger:
    """Audit logger for compliance tracking."""

    def __init__(self):
        self.log = []

    def on_session_event(self, event: Event):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event.type.value,
            "data": str(event),
        }
        self.log.append(entry)
        print(f"  [Audit] {event.type.value}")


async def demo_basic_events():
    """Basic event subscription and handling."""
    print("\n1. Basic Event Handling")
    print("-" * 40)

    emitter = get_emitter()

    # Simple handler
    def on_tokens(event: TokensUsedEvent):
        total = event.input_tokens + event.output_tokens
        print(f"  Tokens used: {total}")

    # Subscribe to event
    emitter.on(EventType.TOKENS_USED, on_tokens)

    # Emit test event
    await emitter.emit(TokensUsedEvent(
        input_tokens=100,
        output_tokens=50,
        model="sonnet",
    ))

    # Unsubscribe
    emitter.off(EventType.TOKENS_USED, on_tokens)


async def demo_multiple_handlers():
    """Multiple handlers for same event type."""
    print("\n2. Multiple Handlers")
    print("-" * 40)

    emitter = EventEmitter()

    handler1_called = False
    handler2_called = False

    def handler1(event):
        nonlocal handler1_called
        handler1_called = True
        print("  Handler 1: received event")

    def handler2(event):
        nonlocal handler2_called
        handler2_called = True
        print("  Handler 2: received event")

    emitter.on(EventType.SESSION_CREATED, handler1)
    emitter.on(EventType.SESSION_CREATED, handler2)

    await emitter.emit(SessionCreatedEvent(
        session_id="test-123",
        model="sonnet",
        tags=[],
    ))

    print(f"  Handler 1 called: {handler1_called}")
    print(f"  Handler 2 called: {handler2_called}")


async def demo_async_handlers():
    """Async event handlers."""
    print("\n3. Async Handlers")
    print("-" * 40)

    emitter = EventEmitter()

    async def async_handler(event: SessionClosedEvent):
        print(f"  Async handler: session {event.session_id[:8]}... closed")
        await asyncio.sleep(0.1)  # Simulate async work
        print("  Async handler: finished processing")

    emitter.on(EventType.SESSION_CLOSED, async_handler)

    await emitter.emit(SessionClosedEvent(
        session_id="test-456",
        total_tokens=1500,
        total_cost_usd=0.05,
    ))


async def demo_metrics_collector():
    """Custom metrics collection using events."""
    print("\n4. Custom Metrics Collector")
    print("-" * 40)

    emitter = EventEmitter()
    collector = MetricsCollector()

    # Register all handlers
    emitter.on(EventType.TOKENS_USED, collector.on_tokens)
    emitter.on(EventType.COST_INCURRED, collector.on_cost)
    emitter.on(EventType.SESSION_CREATED, collector.on_session_created)
    emitter.on(EventType.TURN_COMPLETED, collector.on_turn_completed)
    emitter.on(EventType.TOOL_CALLED, collector.on_tool_called)

    # Simulate session lifecycle
    await emitter.emit(SessionCreatedEvent(
        session_id="metrics-test",
        model="sonnet",
        tags=["demo"],
    ))

    await emitter.emit(TurnStartedEvent(
        session_id="metrics-test",
        turn_number=1,
        prompt="Hello",
    ))

    await emitter.emit(TokensUsedEvent(
        input_tokens=50,
        output_tokens=100,
        model="sonnet",
    ))

    await emitter.emit(CostIncurredEvent(
        cost_usd=0.0045,
        model="sonnet",
        input_tokens=50,
        output_tokens=100,
    ))

    await emitter.emit(TurnCompletedEvent(
        session_id="metrics-test",
        turn_number=1,
        input_tokens=50,
        output_tokens=100,
        duration_ms=1500,
    ))

    print(f"\n  Final metrics:")
    print(f"    Total tokens: {collector.total_tokens}")
    print(f"    Total cost: ${collector.total_cost:.4f}")
    print(f"    Sessions: {collector.sessions_created}")
    print(f"    Turns: {collector.turns_completed}")


async def demo_audit_logging():
    """Audit logging for all events."""
    print("\n5. Audit Logging")
    print("-" * 40)

    emitter = EventEmitter()
    audit = AuditLogger()

    # Subscribe to all event types
    for event_type in EventType:
        emitter.on(event_type, audit.on_session_event)

    # Emit various events
    await emitter.emit(SessionCreatedEvent(
        session_id="audit-test",
        model="opus",
        tags=["production"],
    ))

    await emitter.emit(TokensUsedEvent(
        input_tokens=200,
        output_tokens=500,
        model="opus",
    ))

    await emitter.emit(SessionClosedEvent(
        session_id="audit-test",
        total_tokens=700,
        total_cost_usd=0.15,
    ))

    print(f"\n  Audit log entries: {len(audit.log)}")


async def demo_event_filtering():
    """Filtering events by criteria."""
    print("\n6. Event Filtering")
    print("-" * 40)

    emitter = EventEmitter()
    high_cost_alerts = []

    def cost_alert_handler(event: CostIncurredEvent):
        if event.cost_usd > 0.01:
            high_cost_alerts.append(event)
            print(f"  ALERT: High cost ${event.cost_usd:.4f} for {event.model}")

    emitter.on(EventType.COST_INCURRED, cost_alert_handler)

    # Emit costs - some will trigger alert
    for cost in [0.005, 0.015, 0.002, 0.025]:
        await emitter.emit(CostIncurredEvent(
            cost_usd=cost,
            model="sonnet",
            input_tokens=100,
            output_tokens=200,
        ))

    print(f"\n  High cost events: {len(high_cost_alerts)}")


async def main():
    """Run all event handling demos."""
    print("=" * 60)
    print("Event Handling Examples")
    print("=" * 60)

    await demo_basic_events()
    await demo_multiple_handlers()
    await demo_async_handlers()
    await demo_metrics_collector()
    await demo_audit_logging()
    await demo_event_filtering()

    print("\n" + "=" * 60)
    print("All event demos complete!")


if __name__ == "__main__":
    asyncio.run(main())
