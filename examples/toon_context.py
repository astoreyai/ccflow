#!/usr/bin/env python3
"""
TOON Context Injection Example

Demonstrates TOON serialization for efficient structured data injection
into Claude prompts.
"""

import asyncio

from ccflow import query, CLIAgentOptions, TextMessage
from ccflow.types import ToonConfig
from ccflow.toon_integration import ToonSerializer, should_use_toon, TOON_AVAILABLE


async def main():
    """Demonstrate TOON context injection."""

    # Sample portfolio data (uniform array - ideal for TOON)
    portfolio_data = {
        "account": "U1234567",
        "positions": [
            {"symbol": "AAPL", "qty": 100, "avgCost": 150.25, "pnl": 1500.00},
            {"symbol": "GOOGL", "qty": 50, "avgCost": 2800.00, "pnl": -200.00},
            {"symbol": "MSFT", "qty": 75, "avgCost": 380.50, "pnl": 850.00},
            {"symbol": "TSLA", "qty": 25, "avgCost": 250.00, "pnl": -500.00},
            {"symbol": "AMZN", "qty": 30, "avgCost": 3400.00, "pnl": 1200.00},
        ],
        "cash": 50000.00,
        "margin_used": 0.35,
    }

    print("=" * 60)
    print("TOON Context Injection Demo")
    print("=" * 60)

    # Check TOON applicability
    print(f"\nTOON library available: {TOON_AVAILABLE}")
    print(f"Data suitable for TOON: {should_use_toon(portfolio_data)}")

    # Configure TOON
    toon_config = ToonConfig(
        enabled=TOON_AVAILABLE,  # Enable if library available
        delimiter="\t",          # Tab delimiter for extra savings
        track_savings=True,
    )

    # Show TOON encoding
    serializer = ToonSerializer(toon_config)
    toon_str = serializer.format_for_prompt(portfolio_data, label="Portfolio")

    print("\n" + "-" * 40)
    print("TOON-encoded context preview:")
    print("-" * 40)
    print(toon_str[:500] + "..." if len(toon_str) > 500 else toon_str)

    if toon_config.track_savings and TOON_AVAILABLE:
        ratio = toon_config.last_compression_ratio
        print(f"\nToken savings: {ratio:.1%}")
        print(f"  JSON tokens: {toon_config._last_json_tokens}")
        print(f"  TOON tokens: {toon_config._last_toon_tokens}")

    # Query with TOON context
    print("\n" + "-" * 40)
    print("Querying Claude with TOON context...")
    print("-" * 40)

    options = CLIAgentOptions(
        model="sonnet",
        context=portfolio_data,  # Auto-TOON encoded
        toon=toon_config,
        max_turns=5,
    )

    async for msg in query(
        "Analyze this portfolio's risk exposure. Which positions contribute most to risk?",
        options,
    ):
        if isinstance(msg, TextMessage):
            print(msg.content, end="", flush=True)

    print("\n" + "=" * 60)
    print(f"Final TOON savings: {options.toon.last_compression_ratio:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
