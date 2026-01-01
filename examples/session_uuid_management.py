#!/usr/bin/env python3
"""
Session UUID Management Example

Demonstrates two powerful patterns:
1. Resuming conversations using session UUIDs
2. Consolidating context from multiple sessions into a new session

Use cases:
- Resume a research session days later
- Combine insights from parallel investigations
- Create summary sessions from multiple topic explorations
"""

import asyncio
from datetime import datetime
from pathlib import Path
import tempfile

from ccflow import Session, CLIAgentOptions, query, TextMessage, AssistantMessage
from ccflow.session import load_session
from ccflow.store import SessionState, SessionStatus
from ccflow.stores.sqlite import SQLiteSessionStore
from ccflow.stores.memory import MemorySessionStore


def extract_text(msg) -> str:
    """Extract text content from various message types."""
    if isinstance(msg, TextMessage):
        return msg.content
    elif isinstance(msg, AssistantMessage):
        return msg.text_content
    return ""


# =============================================================================
# Pattern 1: Resume Conversation by UUID
# =============================================================================

async def demo_session_resume():
    """Demonstrate resuming a conversation using its UUID.

    This pattern is useful when:
    - You need to continue a conversation after closing the terminal
    - You want to pick up where you left off in a research session
    - You're building a chatbot that maintains conversation history
    """
    print("\n" + "=" * 60)
    print("Pattern 1: Resume Conversation by UUID")
    print("=" * 60)

    # Use a temp database for this demo
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteSessionStore(db_path)

    # --- First Session: Start a conversation ---
    print("\n[Step 1] Starting initial conversation...")

    session1 = Session(
        options=CLIAgentOptions(model="haiku"),
        store=store,
    )

    # Save the UUID for later
    session_uuid = session1.session_id
    print(f"  Session UUID: {session_uuid}")

    # Have a conversation
    print("  Asking about Python decorators...")
    response1 = ""
    async for msg in session1.send_message(
        "Explain Python decorators in 2 sentences. Remember this for later."
    ):
        response1 += extract_text(msg)

    print(f"  Response: {response1[:100]}...")
    print(f"  Turns so far: {session1.turn_count}")

    # Close the session (simulating end of first interaction)
    await session1.close()
    print("  Session closed (simulating app restart)")

    # --- Later: Resume the conversation ---
    print("\n[Step 2] Resuming conversation using UUID...")
    print(f"  Loading session: {session_uuid}")

    # Load the session by UUID
    resumed_session = await load_session(session_uuid, store)

    if resumed_session:
        print(f"  Successfully loaded! Previous turns: {resumed_session.turn_count}")

        # Continue the conversation - Claude remembers context
        print("  Asking follow-up question...")
        response2 = ""
        async for msg in resumed_session.send_message(
            "Now give me a simple example of what you just explained."
        ):
            response2 += extract_text(msg)

        print(f"  Response: {response2[:100]}...")
        print(f"  Total turns: {resumed_session.turn_count}")

        await resumed_session.close()
    else:
        print("  ERROR: Session not found!")

    # Cleanup
    await store.close()
    Path(db_path).unlink()

    return session_uuid


# =============================================================================
# Pattern 2: Consolidate Multiple Sessions
# =============================================================================

async def demo_context_consolidation():
    """Demonstrate consolidating context from multiple sessions.

    This pattern is useful when:
    - You've researched multiple topics and want to synthesize findings
    - You have parallel investigation threads to merge
    - You need to create a summary from multiple conversations
    """
    print("\n" + "=" * 60)
    print("Pattern 2: Consolidate Context from Multiple Sessions")
    print("=" * 60)

    # Use memory store for this demo (faster, no cleanup needed)
    store = MemorySessionStore()

    # Track session UUIDs and their topics
    research_sessions: dict[str, dict] = {}

    # --- Create multiple research sessions ---
    print("\n[Step 1] Creating parallel research sessions...")

    topics = [
        ("benefits", "What are 3 key benefits of microservices architecture? Be concise."),
        ("challenges", "What are 3 main challenges of microservices? Be concise."),
        ("patterns", "Name 3 common microservices design patterns. Be concise."),
    ]

    for topic_name, question in topics:
        session = Session(
            options=CLIAgentOptions(model="haiku"),
            store=store,
        )

        session_uuid = session.session_id
        print(f"  [{topic_name}] Session UUID: {session_uuid[:8]}...")

        # Get response
        response = ""
        async for msg in session.send_message(question):
            response += extract_text(msg)

        # Store session info
        research_sessions[session_uuid] = {
            "topic": topic_name,
            "question": question,
            "response": response,
            "turns": session.turn_count,
        }

        print(f"      Response: {response[:60]}...")
        await session.close()

    # --- Consolidate into a summary session ---
    print("\n[Step 2] Creating consolidation session...")

    # Build context from all previous sessions
    context_parts = []
    for uuid, info in research_sessions.items():
        context_parts.append(f"""
### Research: {info['topic'].title()}
UUID: {uuid}
Question: {info['question']}
Findings: {info['response']}
""")

    consolidated_context = "\n".join(context_parts)

    # Create new session for synthesis
    synthesis_session = Session(
        options=CLIAgentOptions(model="haiku"),
        store=store,
    )

    print(f"  Synthesis Session UUID: {synthesis_session.session_id[:8]}...")
    print(f"  Consolidating {len(research_sessions)} research sessions...")

    # Ask Claude to synthesize
    synthesis_prompt = f"""I have conducted research on microservices across multiple sessions.
Please synthesize these findings into a brief executive summary (3-4 sentences):

{consolidated_context}

Provide a cohesive summary that integrates all the key points."""

    synthesis_response = ""
    async for msg in synthesis_session.send_message(synthesis_prompt):
        synthesis_response += extract_text(msg)

    print(f"\n  Synthesized Summary:")
    print(f"  {'-' * 50}")
    print(f"  {synthesis_response}")
    print(f"  {'-' * 50}")

    # Store metadata about source sessions
    await synthesis_session.close()

    await store.close()

    return list(research_sessions.keys()), synthesis_session.session_id


# =============================================================================
# Pattern 3: Session Registry for Long-Running Projects
# =============================================================================

async def demo_session_registry():
    """Demonstrate a session registry for project-based work.

    This pattern is useful when:
    - Working on a long-running project with multiple aspects
    - Need to track which sessions relate to which features
    - Want to resume any aspect of the project later
    """
    print("\n" + "=" * 60)
    print("Pattern 3: Session Registry for Project Management")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteSessionStore(db_path)

    # Create a registry mapping project aspects to session UUIDs
    project_registry: dict[str, str] = {}

    print("\n[Step 1] Creating sessions for different project aspects...")

    aspects = {
        "architecture": "You are helping design the architecture for a todo app.",
        "database": "You are helping design the database schema for a todo app.",
        "api": "You are helping design the REST API for a todo app.",
    }

    for aspect, system_prompt in aspects.items():
        session = Session(
            options=CLIAgentOptions(
                model="haiku",
                system_prompt=system_prompt,
            ),
            store=store,
        )

        # Register the session
        project_registry[aspect] = session.session_id
        print(f"  [{aspect}] Registered UUID: {session.session_id[:8]}...")

        # Quick initial message to establish context
        async for msg in session.send_message(f"Acknowledge that you're the {aspect} expert."):
            pass

        await session.close()

    # --- Later: Resume specific aspect ---
    print("\n[Step 2] Resuming specific project aspect...")

    aspect_to_resume = "database"
    uuid_to_resume = project_registry[aspect_to_resume]

    print(f"  Resuming '{aspect_to_resume}' session: {uuid_to_resume[:8]}...")

    resumed = await load_session(uuid_to_resume, store)
    if resumed:
        response = ""
        async for msg in resumed.send_message("What tables should we have? List 3."):
            response += extract_text(msg)

        print(f"  Response: {response[:100]}...")
        await resumed.close()

    # --- Show full registry ---
    print("\n[Step 3] Project Session Registry:")
    print(f"  {'-' * 50}")
    for aspect, uuid in project_registry.items():
        state = await store.load(uuid)
        status = state.status.value if state else "unknown"
        turns = state.turn_count if state else 0
        print(f"  {aspect:15} | {uuid[:8]}... | {status:8} | {turns} turns")
    print(f"  {'-' * 50}")

    await store.close()
    Path(db_path).unlink()

    return project_registry


# =============================================================================
# Pattern 4: Fork and Experiment
# =============================================================================

async def demo_fork_and_experiment():
    """Demonstrate forking a session to try different approaches.

    This pattern is useful when:
    - You want to explore alternative solutions
    - Need to A/B test different prompts
    - Want to preserve original context while experimenting
    """
    print("\n" + "=" * 60)
    print("Pattern 4: Fork Session for Experimentation")
    print("=" * 60)

    store = MemorySessionStore()

    # --- Create base session ---
    print("\n[Step 1] Creating base session with shared context...")

    base_session = Session(
        options=CLIAgentOptions(model="haiku"),
        store=store,
    )
    base_uuid = base_session.session_id

    # Establish context
    async for msg in base_session.send_message(
        "I'm building a Python web app. I need help choosing a framework. "
        "Remember: I prioritize simplicity and good documentation."
    ):
        pass

    print(f"  Base session: {base_uuid[:8]}...")
    base_state = await store.load(base_uuid)
    await base_session.close()

    # --- Fork into multiple experimental sessions ---
    print("\n[Step 2] Forking into experimental branches...")

    experiments = {}
    approaches = [
        ("flask_approach", "Recommend Flask and explain why it fits my criteria."),
        ("fastapi_approach", "Recommend FastAPI and explain why it fits my criteria."),
        ("django_approach", "Recommend Django and explain why it fits my criteria."),
    ]

    for exp_name, prompt in approaches:
        # Create new session that inherits context conceptually
        # (In practice, we include the original context in the prompt)
        exp_session = Session(
            options=CLIAgentOptions(model="haiku"),
            store=store,
        )

        # Reference the base session in metadata
        exp_uuid = exp_session.session_id
        experiments[exp_name] = {
            "uuid": exp_uuid,
            "forked_from": base_uuid,
        }

        # Include context from base session
        full_prompt = f"""Context from previous session ({base_uuid[:8]}...):
I'm building a Python web app. I need help choosing a framework.
I prioritize simplicity and good documentation.

Now, {prompt}"""

        response = ""
        async for msg in exp_session.send_message(full_prompt):
            response += extract_text(msg)

        experiments[exp_name]["response"] = response[:80]
        print(f"  [{exp_name}] UUID: {exp_uuid[:8]}... (forked from {base_uuid[:8]})")
        print(f"      Response: {response[:60]}...")

        await exp_session.close()

    # --- Show experiment tree ---
    print("\n[Step 3] Session Experiment Tree:")
    print(f"  Base: {base_uuid[:8]}...")
    for exp_name, info in experiments.items():
        print(f"    └─ {exp_name}: {info['uuid'][:8]}...")

    await store.close()

    return base_uuid, experiments


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all UUID management demos."""
    print("=" * 60)
    print("Session UUID Management Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to use session UUIDs for:")
    print("  1. Resuming conversations across app restarts")
    print("  2. Consolidating context from multiple sessions")
    print("  3. Managing project-based session registries")
    print("  4. Forking sessions for experimentation")

    await demo_session_resume()
    await demo_context_consolidation()
    await demo_session_registry()
    await demo_fork_and_experiment()

    print("\n" + "=" * 60)
    print("All UUID management demos complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  - Store session UUIDs to resume conversations later")
    print("  - Use load_session(uuid, store) to restore sessions")
    print("  - Consolidate by passing previous responses as context")
    print("  - Fork sessions by creating new ones with shared context")


if __name__ == "__main__":
    asyncio.run(main())
