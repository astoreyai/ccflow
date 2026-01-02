"""
Custom tools for ccflow.

Provides @tool decorator for defining custom MCP tools and
create_sdk_mcp_server() for creating in-process MCP servers.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

# Type aliases
ToolHandler: TypeAlias = Callable[..., Any]


@dataclass
class SdkMcpTool:
    """Definition of an MCP tool.

    Matches Claude Agent SDK's tool definition format.

    Attributes:
        name: Tool name (unique identifier)
        description: What the tool does (for model context)
        input_schema: JSON schema for tool input parameters
        handler: Async function to execute when tool is called
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    metadata: dict[str, Any] = field(default_factory=dict)


def _generate_input_schema(func: Callable) -> dict[str, Any]:
    """Generate JSON schema from function signature.

    Args:
        func: Function to analyze

    Returns:
        JSON schema dict for the function's parameters
    """
    sig = inspect.signature(func)
    hints = getattr(func, "__annotations__", {})

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        prop: dict[str, Any] = {}

        # Get type from annotations
        hint = hints.get(name)
        if hint is not None:
            prop["type"] = _python_type_to_json_type(hint)
        else:
            prop["type"] = "string"  # Default

        # Check if parameter has description in docstring
        # (Would need docstring parsing, simplified for now)

        properties[name] = prop

        # Required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_type(python_type: type) -> str:
    """Convert Python type to JSON schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON schema type string
    """
    # Handle string representation of type
    type_str = str(python_type)

    # Handle basic types
    if python_type is str or "str" in type_str:
        return "string"
    if python_type is int or "int" in type_str:
        return "integer"
    if python_type is float or "float" in type_str:
        return "number"
    if python_type is bool or "bool" in type_str:
        return "boolean"
    if python_type is list or "list" in type_str.lower():
        return "array"
    if python_type is dict or "dict" in type_str.lower():
        return "object"

    # Default to string for unknown types
    return "string"


def tool(
    name: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Callable[[ToolHandler], SdkMcpTool]:
    """Decorator for defining custom MCP tools.

    Creates an SdkMcpTool from a function, using the function's
    signature and docstring for schema generation.

    Example:
        >>> @tool(description="Read a file from disk")
        ... async def read_file(path: str) -> str:
        ...     with open(path) as f:
        ...         return f.read()

        >>> @tool(
        ...     name="search",
        ...     description="Search for files",
        ...     input_schema={
        ...         "type": "object",
        ...         "properties": {"query": {"type": "string"}},
        ...         "required": ["query"],
        ...     }
        ... )
        ... async def search_files(query: str) -> list[str]:
        ...     return []

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        input_schema: JSON schema (auto-generated if not provided)

    Returns:
        Decorator that creates SdkMcpTool from function
    """

    def decorator(func: ToolHandler) -> SdkMcpTool:
        # Get name from function if not provided
        tool_name = name or func.__name__

        # Get description from docstring if not provided
        tool_description = description
        if tool_description is None:
            tool_description = inspect.getdoc(func) or f"Execute {tool_name}"

        # Generate schema if not provided
        schema = input_schema
        if schema is None:
            schema = _generate_input_schema(func)

        # Create and return the tool
        mcp_tool = SdkMcpTool(
            name=tool_name,
            description=tool_description,
            input_schema=schema,
            handler=func,
        )

        return mcp_tool

    return decorator


@dataclass
class McpSdkServerConfig:
    """Configuration for an in-process MCP server.

    Attributes:
        name: Server name for identification
        version: Server version string
        tools: List of tools provided by this server
    """

    name: str
    version: str = "1.0.0"
    tools: list[SdkMcpTool] = field(default_factory=list)


class SdkMcpServer:
    """In-process MCP server for custom tools.

    Provides a way to define and run custom tools that integrate
    with the Claude Code CLI.

    Example:
        >>> server = create_sdk_mcp_server(
        ...     name="my-tools",
        ...     version="1.0.0",
        ...     tools=[read_file_tool, search_tool],
        ... )
        >>> # Server config can be passed to CLI via MCP config
    """

    def __init__(self, config: McpSdkServerConfig) -> None:
        """Initialize MCP server.

        Args:
            config: Server configuration
        """
        self.config = config
        self._tools: dict[str, SdkMcpTool] = {t.name: t for t in config.tools}

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def version(self) -> str:
        """Get server version."""
        return self.config.version

    def list_tools(self) -> list[dict[str, Any]]:
        """List available tools in MCP format.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def get_tool(self, name: str) -> SdkMcpTool | None:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None if not found
        """
        return self._tools.get(name)

    def add_tool(self, tool: SdkMcpTool) -> None:
        """Add a tool to the server.

        Args:
            tool: Tool to add
        """
        self._tools[tool.name] = tool
        self.config.tools.append(tool)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool input arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}")

        # Call the handler
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(**arguments)
        return tool.handler(**arguments)

    def to_mcp_config(self) -> dict[str, Any]:
        """Generate MCP server configuration.

        Returns:
            Configuration dict for --mcp-config flag
        """
        return {
            "mcpServers": {
                self.name: {
                    "type": "sdk",
                    "name": self.name,
                    "version": self.version,
                    "tools": self.list_tools(),
                }
            }
        }


def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool] | None = None,
) -> SdkMcpServer:
    """Create an in-process MCP server.

    This is the main factory function for creating custom MCP servers
    that can be used with the Claude Code CLI.

    Example:
        >>> @tool(description="Add two numbers")
        ... async def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> server = create_sdk_mcp_server(
        ...     name="math-tools",
        ...     version="1.0.0",
        ...     tools=[add],
        ... )
        >>> config = server.to_mcp_config()

    Args:
        name: Server name
        version: Server version
        tools: List of tools to include

    Returns:
        Configured MCP server instance
    """
    config = McpSdkServerConfig(
        name=name,
        version=version,
        tools=tools or [],
    )
    return SdkMcpServer(config)


@dataclass
class TaskToolUse:
    """Represents a Task tool invocation for subagent spawning.

    Parsed from ToolUseMessage when the tool is "Task".

    Attributes:
        subagent_type: Name of the subagent to spawn
        prompt: Task prompt to send to the subagent
        description: Short description of the task
        run_in_background: Whether to run asynchronously
        model: Optional model override for the subagent
    """

    subagent_type: str
    prompt: str
    description: str | None = None
    run_in_background: bool = False
    model: str | None = None


def parse_task_tool(tool_name: str, args: dict[str, Any]) -> TaskToolUse | None:
    """Parse Task tool invocation from ToolUseMessage.

    Args:
        tool_name: Name of the tool
        args: Tool arguments

    Returns:
        TaskToolUse if this is a Task tool invocation, None otherwise
    """
    if tool_name != "Task":
        return None

    return TaskToolUse(
        subagent_type=args.get("subagent_type", ""),
        prompt=args.get("prompt", ""),
        description=args.get("description"),
        run_in_background=args.get("run_in_background", False),
        model=args.get("model"),
    )


# Global tool registry for convenience
_global_tools: dict[str, SdkMcpTool] = {}


def register_tool(tool: SdkMcpTool) -> None:
    """Register a tool in the global registry.

    Args:
        tool: Tool to register
    """
    _global_tools[tool.name] = tool


def get_tool(name: str) -> SdkMcpTool | None:
    """Get a tool from the global registry.

    Args:
        name: Tool name

    Returns:
        Tool or None if not found
    """
    return _global_tools.get(name)


def list_tools() -> list[str]:
    """List all registered tool names.

    Returns:
        List of tool names
    """
    return list(_global_tools.keys())


def clear_tools() -> None:
    """Clear all registered tools (for testing)."""
    _global_tools.clear()
