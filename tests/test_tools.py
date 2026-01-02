"""
Tests for custom tools system.

Tests @tool decorator, SdkMcpTool, and MCP server creation.
"""

from __future__ import annotations

import pytest

from ccflow.tools import (
    McpSdkServerConfig,
    SdkMcpServer,
    SdkMcpTool,
    TaskToolUse,
    clear_tools,
    create_sdk_mcp_server,
    get_tool,
    list_tools,
    parse_task_tool,
    register_tool,
    tool,
)


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_basic_tool(self):
        """Create tool with decorator."""

        @tool(description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, SdkMcpTool)
        assert add.name == "add"
        assert add.description == "Add two numbers"

    def test_tool_with_custom_name(self):
        """Create tool with custom name."""

        @tool(name="my_add", description="Add numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "my_add"

    def test_tool_schema_generation(self):
        """Schema is auto-generated from signature."""

        @tool(description="Test function")
        async def test_func(name: str, count: int, enabled: bool = True) -> str:
            return f"{name} x {count}"

        schema = test_func.input_schema

        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert "enabled" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["enabled"]["type"] == "boolean"
        assert "name" in schema["required"]
        assert "count" in schema["required"]
        assert "enabled" not in schema["required"]  # Has default

    def test_tool_with_explicit_schema(self):
        """Use explicit schema instead of auto-generation."""
        custom_schema = {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }

        @tool(description="Search", input_schema=custom_schema)
        async def search(query: str) -> list[str]:
            return []

        assert search.input_schema == custom_schema

    def test_tool_uses_docstring(self):
        """Description defaults to docstring."""

        @tool()
        async def my_function(x: int) -> int:
            """This is the docstring description."""
            return x * 2

        assert my_function.description == "This is the docstring description."

    def test_tool_default_description(self):
        """Default description when no docstring."""

        @tool()
        async def nodoc(x: int) -> int:
            return x

        assert "nodoc" in nodoc.description

    @pytest.mark.asyncio
    async def test_tool_is_callable(self):
        """Tool handler can be called."""

        @tool(description="Multiply")
        async def multiply(a: int, b: int) -> int:
            return a * b

        result = await multiply.handler(a=3, b=4)
        assert result == 12

    def test_sync_tool(self):
        """Sync functions work as tools."""

        @tool(description="Sync add")
        def sync_add(a: int, b: int) -> int:
            return a + b

        assert sync_add.name == "sync_add"
        assert sync_add.handler(a=2, b=3) == 5


class TestSdkMcpTool:
    """Tests for SdkMcpTool dataclass."""

    def test_create_tool(self):
        """Create tool manually."""

        async def handler(x: int) -> int:
            return x * 2

        tool = SdkMcpTool(
            name="double",
            description="Double a number",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=handler,
        )

        assert tool.name == "double"
        assert tool.description == "Double a number"

    def test_tool_metadata(self):
        """Tool can have metadata."""

        async def handler():
            pass

        tool = SdkMcpTool(
            name="test",
            description="Test",
            input_schema={},
            handler=handler,
            metadata={"version": "1.0", "author": "test"},
        )

        assert tool.metadata["version"] == "1.0"
        assert tool.metadata["author"] == "test"


class TestSdkMcpServer:
    """Tests for SdkMcpServer."""

    def test_create_server(self):
        """Create MCP server."""
        server = create_sdk_mcp_server(name="test-server", version="1.0.0")

        assert server.name == "test-server"
        assert server.version == "1.0.0"

    def test_server_with_tools(self):
        """Create server with tools."""

        @tool(description="Add")
        async def add(a: int, b: int) -> int:
            return a + b

        server = create_sdk_mcp_server(
            name="math-server",
            tools=[add],
        )

        assert len(server.list_tools()) == 1
        assert server.list_tools()[0]["name"] == "add"

    def test_list_tools(self):
        """List tools returns MCP format."""

        @tool(description="Test tool")
        async def test_tool(x: str) -> str:
            return x

        server = create_sdk_mcp_server(name="test", tools=[test_tool])
        tools = server.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "Test tool"
        assert "inputSchema" in tools[0]

    def test_get_tool(self):
        """Get tool by name."""

        @tool(description="Test")
        async def my_tool(x: int) -> int:
            return x

        server = create_sdk_mcp_server(name="test", tools=[my_tool])

        found = server.get_tool("my_tool")
        assert found is not None
        assert found.name == "my_tool"

        not_found = server.get_tool("nonexistent")
        assert not_found is None

    def test_add_tool(self):
        """Add tool to server."""
        server = create_sdk_mcp_server(name="test")

        @tool(description="Added later")
        async def late_tool(x: int) -> int:
            return x

        server.add_tool(late_tool)

        assert server.get_tool("late_tool") is not None

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Call tool through server."""

        @tool(description="Add")
        async def add(a: int, b: int) -> int:
            return a + b

        server = create_sdk_mcp_server(name="test", tools=[add])
        result = await server.call_tool("add", {"a": 3, "b": 4})

        assert result == 7

    @pytest.mark.asyncio
    async def test_call_sync_tool(self):
        """Call sync tool through server."""

        @tool(description="Sync multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        server = create_sdk_mcp_server(name="test", tools=[multiply])
        result = await server.call_tool("multiply", {"a": 3, "b": 4})

        assert result == 12

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self):
        """Calling nonexistent tool raises error."""
        server = create_sdk_mcp_server(name="test")

        with pytest.raises(ValueError, match="Tool not found"):
            await server.call_tool("nonexistent", {})

    def test_to_mcp_config(self):
        """Generate MCP config."""

        @tool(description="Test")
        async def test_tool(x: int) -> int:
            return x

        server = create_sdk_mcp_server(
            name="my-server",
            version="2.0.0",
            tools=[test_tool],
        )

        config = server.to_mcp_config()

        assert "mcpServers" in config
        assert "my-server" in config["mcpServers"]
        assert config["mcpServers"]["my-server"]["type"] == "sdk"
        assert config["mcpServers"]["my-server"]["version"] == "2.0.0"
        assert len(config["mcpServers"]["my-server"]["tools"]) == 1


class TestMcpSdkServerConfig:
    """Tests for McpSdkServerConfig."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = McpSdkServerConfig(name="test")

        assert config.name == "test"
        assert config.version == "1.0.0"
        assert config.tools == []

    def test_config_with_tools(self):
        """Config can include tools."""

        @tool(description="Test")
        async def test_tool(x: int) -> int:
            return x

        config = McpSdkServerConfig(
            name="test",
            version="2.0.0",
            tools=[test_tool],
        )

        assert len(config.tools) == 1
        assert config.tools[0].name == "test_tool"


class TestTaskToolUse:
    """Tests for TaskToolUse parsing."""

    def test_parse_task_tool(self):
        """Parse Task tool invocation."""
        args = {
            "subagent_type": "code-reviewer",
            "prompt": "Review this code",
            "description": "Code review task",
            "run_in_background": False,
        }

        result = parse_task_tool("Task", args)

        assert result is not None
        assert result.subagent_type == "code-reviewer"
        assert result.prompt == "Review this code"
        assert result.description == "Code review task"
        assert result.run_in_background is False

    def test_parse_task_tool_with_model(self):
        """Parse Task tool with model override."""
        args = {
            "subagent_type": "researcher",
            "prompt": "Research topic",
            "model": "opus",
        }

        result = parse_task_tool("Task", args)

        assert result is not None
        assert result.model == "opus"

    def test_parse_non_task_tool(self):
        """Non-Task tools return None."""
        result = parse_task_tool("Read", {"file_path": "/tmp/test"})
        assert result is None

    def test_parse_task_defaults(self):
        """Missing fields use defaults."""
        args = {
            "subagent_type": "test",
            "prompt": "Test",
        }

        result = parse_task_tool("Task", args)

        assert result.description is None
        assert result.run_in_background is False
        assert result.model is None


class TestGlobalToolRegistry:
    """Tests for global tool registry."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_tools()

    def test_register_and_get(self):
        """Register and retrieve tool."""

        @tool(description="Test")
        async def test_tool(x: int) -> int:
            return x

        register_tool(test_tool)

        found = get_tool("test_tool")
        assert found is not None
        assert found.name == "test_tool"

    def test_get_nonexistent(self):
        """Get nonexistent returns None."""
        result = get_tool("nonexistent")
        assert result is None

    def test_list_tools(self):
        """List registered tools."""

        @tool(description="Tool A")
        async def tool_a(x: int) -> int:
            return x

        @tool(description="Tool B")
        async def tool_b(x: int) -> int:
            return x

        register_tool(tool_a)
        register_tool(tool_b)

        names = list_tools()
        assert set(names) == {"tool_a", "tool_b"}

    def test_clear_tools(self):
        """Clear removes all tools."""

        @tool(description="Test")
        async def test_tool(x: int) -> int:
            return x

        register_tool(test_tool)
        assert len(list_tools()) == 1

        clear_tools()
        assert len(list_tools()) == 0


class TestSchemaGeneration:
    """Tests for automatic schema generation."""

    def test_string_type(self):
        """String type annotation."""

        @tool(description="Test")
        async def func(name: str) -> str:
            return name

        assert func.input_schema["properties"]["name"]["type"] == "string"

    def test_int_type(self):
        """Int type annotation."""

        @tool(description="Test")
        async def func(count: int) -> int:
            return count

        assert func.input_schema["properties"]["count"]["type"] == "integer"

    def test_float_type(self):
        """Float type annotation."""

        @tool(description="Test")
        async def func(value: float) -> float:
            return value

        assert func.input_schema["properties"]["value"]["type"] == "number"

    def test_bool_type(self):
        """Bool type annotation."""

        @tool(description="Test")
        async def func(enabled: bool) -> bool:
            return enabled

        assert func.input_schema["properties"]["enabled"]["type"] == "boolean"

    def test_list_type(self):
        """List type annotation."""

        @tool(description="Test")
        async def func(items: list) -> list:
            return items

        assert func.input_schema["properties"]["items"]["type"] == "array"

    def test_dict_type(self):
        """Dict type annotation."""

        @tool(description="Test")
        async def func(data: dict) -> dict:
            return data

        assert func.input_schema["properties"]["data"]["type"] == "object"

    def test_no_annotation(self):
        """No annotation defaults to string."""

        @tool(description="Test")
        async def func(x):
            return x

        assert func.input_schema["properties"]["x"]["type"] == "string"

    def test_required_vs_optional(self):
        """Required params have no default."""

        @tool(description="Test")
        async def func(required_param: str, optional_param: str = "default") -> str:
            return required_param + optional_param

        assert "required_param" in func.input_schema["required"]
        assert "optional_param" not in func.input_schema["required"]
