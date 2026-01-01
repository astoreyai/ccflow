#!/usr/bin/env python3
"""
Batch Processing Example

Demonstrates concurrent batch query processing for multiple prompts.
"""

import asyncio

from ccflow import batch_query, CLIAgentOptions


async def main():
    """Process multiple prompts in parallel."""

    # Define prompts to process
    prompts = [
        "What is the purpose of the README.md file?",
        "List the main Python files in this project",
        "What testing framework is used?",
        "Describe the project structure briefly",
    ]

    # Configure options (shared across all prompts)
    options = CLIAgentOptions(
        model="haiku",  # Use faster model for batch
        max_turns=3,
        allowed_tools=["Read", "Glob"],
    )

    print(f"Processing {len(prompts)} prompts with concurrency=3...")
    print("=" * 50)

    # Execute batch query
    results = await batch_query(
        prompts,
        options=options,
        concurrency=3,  # Process 3 at a time
    )

    # Display results
    for i, result in enumerate(results):
        print(f"\n[Prompt {i + 1}]")
        print(f"Status: {'Success' if result.success else 'Error'}")

        if result.success:
            # Truncate long results for display
            text = result.result
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"Response: {text}")
            print(f"Tokens: {result.total_tokens}")
            print(f"Duration: {result.duration_seconds:.2f}s")
        else:
            print(f"Error: {result.error}")

    # Summary
    print("\n" + "=" * 50)
    successful = sum(1 for r in results if r.success)
    total_tokens = sum(r.total_tokens for r in results)
    total_duration = sum(r.duration_seconds for r in results)

    print(f"Summary: {successful}/{len(results)} successful")
    print(f"Total tokens: {total_tokens}")
    print(f"Total duration: {total_duration:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
