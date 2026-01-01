#!/usr/bin/env python3
"""
HTTP Server Example

Demonstrates running ccflow as an HTTP service with REST and streaming endpoints.

Requirements:
    pip install ccflow[server]  # Installs FastAPI and uvicorn
"""

import asyncio
import sys

# Check for server dependencies
try:
    from fastapi import FastAPI
    import uvicorn
except ImportError:
    print("This example requires server extras:")
    print("  pip install ccflow[server]")
    sys.exit(1)

from ccflow import CLIAgentOptions
from ccflow.server import CCFlowServer


async def demo_server_setup():
    """Basic server setup and configuration."""
    print("\n1. Server Setup")
    print("-" * 40)

    # Create server with configuration
    server = CCFlowServer(
        host="0.0.0.0",
        port=8080,
        default_options=CLIAgentOptions(
            model="sonnet",
            allowed_tools=["Read", "Grep", "Glob"],
        ),
    )

    print(f"  Server configured:")
    print(f"    Host: {server._host}")
    print(f"    Port: {server._port}")
    print(f"    Default model: {server._default_options.model}")

    # Get FastAPI app for customization
    app = server.app
    print(f"  FastAPI app ready: {app is not None}")

    # Add custom routes
    @app.get("/custom/health")
    async def custom_health():
        return {"status": "healthy", "service": "ccflow"}

    print("  Added custom /custom/health endpoint")


async def demo_api_endpoints():
    """Available API endpoints."""
    print("\n2. Available Endpoints")
    print("-" * 40)

    endpoints = [
        ("POST", "/query", "Execute a query (non-streaming)"),
        ("POST", "/query/stream", "Execute with SSE streaming"),
        ("GET", "/sessions", "List all sessions"),
        ("GET", "/sessions/{id}", "Get session details"),
        ("DELETE", "/sessions/{id}", "Delete a session"),
        ("GET", "/health", "Health check"),
        ("GET", "/metrics", "Prometheus metrics"),
    ]

    for method, path, desc in endpoints:
        print(f"  {method:6} {path:25} - {desc}")


async def demo_request_examples():
    """Example requests to the server."""
    print("\n3. Request Examples")
    print("-" * 40)

    print("""
  # Query (non-streaming)
  curl -X POST http://localhost:8080/query \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "What is Python?", "model": "haiku"}'

  # Streaming query (SSE)
  curl -X POST http://localhost:8080/query/stream \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Tell me a story"}' \\
    --no-buffer

  # List sessions
  curl http://localhost:8080/sessions

  # Health check
  curl http://localhost:8080/health
    """)


async def demo_docker_usage():
    """Docker deployment pattern."""
    print("\n4. Docker Usage")
    print("-" * 40)

    print("""
  # Dockerfile is included in project
  docker build -t ccflow .

  # Run with Claude credentials mounted
  docker run -p 8080:8080 \\
    -v ~/.claude:/home/ccflow/.claude:ro \\
    ccflow

  # Or use docker-compose
  docker compose --profile prod up

  # Access server
  curl http://localhost:8080/health
    """)


async def demo_programmatic_client():
    """Using the server programmatically."""
    print("\n5. Programmatic Client")
    print("-" * 40)

    print("""
  import httpx

  async with httpx.AsyncClient() as client:
      # Non-streaming
      response = await client.post(
          "http://localhost:8080/query",
          json={"prompt": "Hello!", "model": "haiku"},
      )
      print(response.json())

      # Streaming with SSE
      async with client.stream(
          "POST",
          "http://localhost:8080/query/stream",
          json={"prompt": "Tell me about AI"},
      ) as response:
          async for line in response.aiter_lines():
              if line.startswith("data: "):
                  data = json.loads(line[6:])
                  print(data)
    """)


def run_server():
    """Actually run the server (if invoked with --run flag)."""
    print("\n" + "=" * 60)
    print("Starting ccflow HTTP Server...")
    print("=" * 60)

    server = CCFlowServer(
        host="0.0.0.0",
        port=8080,
        default_options=CLIAgentOptions(model="sonnet"),
    )

    # Start server (blocking)
    asyncio.run(server.start())


async def main():
    """Run all server demos."""
    print("=" * 60)
    print("HTTP Server Examples")
    print("=" * 60)

    await demo_server_setup()
    await demo_api_endpoints()
    await demo_request_examples()
    await demo_docker_usage()
    await demo_programmatic_client()

    print("\n" + "=" * 60)
    print("To start the server, run:")
    print("  python examples/http_server.py --run")
    print("=" * 60)


if __name__ == "__main__":
    if "--run" in sys.argv:
        run_server()
    else:
        asyncio.run(main())
