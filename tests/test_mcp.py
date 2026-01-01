"""Tests for MCP configuration module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ccflow.mcp import (
    MCPConfigManager,
    github_server,
    postgres_server,
    playwright_server,
    filesystem_server,
    custom_http_server,
)
from ccflow.types import MCPServerConfig
from ccflow.exceptions import MCPConfigError


class TestMCPConfigManager:
    """Tests for MCPConfigManager class."""

    def test_init_default_temp_dir(self):
        """Test manager uses system temp dir by default."""
        import tempfile

        manager = MCPConfigManager()
        assert manager._temp_dir == Path(tempfile.gettempdir())

    def test_init_custom_temp_dir(self, tmp_path):
        """Test manager uses provided temp dir."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        assert manager._temp_dir == tmp_path

    def test_init_string_temp_dir(self, tmp_path):
        """Test manager accepts string path."""
        manager = MCPConfigManager(temp_dir=str(tmp_path))
        assert manager._temp_dir == tmp_path


class TestCreateConfigFile:
    """Tests for create_config_file method."""

    def test_creates_json_file(self, tmp_path):
        """Test config file is created as JSON."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {
            "test": MCPServerConfig(command="echo", args=["hello"]),
        }

        config_path = manager.create_config_file(servers)

        assert config_path.exists()
        assert config_path.suffix == ".json"
        manager.cleanup()

    def test_config_content_structure(self, tmp_path):
        """Test generated config has correct structure."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {
            "myserver": MCPServerConfig(
                command="python",
                args=["-m", "myserver"],
                env={"API_KEY": "secret"},
            ),
        }

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        assert "mcpServers" in config
        assert "myserver" in config["mcpServers"]
        assert config["mcpServers"]["myserver"]["command"] == "python"
        assert config["mcpServers"]["myserver"]["args"] == ["-m", "myserver"]
        assert config["mcpServers"]["myserver"]["env"] == {"API_KEY": "secret"}

        manager.cleanup()

    def test_multiple_servers(self, tmp_path):
        """Test config with multiple servers."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {
            "server1": MCPServerConfig(command="cmd1"),
            "server2": MCPServerConfig(command="cmd2", args=["arg1"]),
            "server3": MCPServerConfig(command="cmd3", env={"KEY": "val"}),
        }

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        assert len(config["mcpServers"]) == 3
        assert "server1" in config["mcpServers"]
        assert "server2" in config["mcpServers"]
        assert "server3" in config["mcpServers"]

        manager.cleanup()

    def test_empty_servers_raises_error(self, tmp_path):
        """Test empty servers dict raises MCPConfigError."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        with pytest.raises(MCPConfigError) as exc_info:
            manager.create_config_file({})

        assert "No MCP servers provided" in str(exc_info.value)

    def test_tracks_created_files(self, tmp_path):
        """Test manager tracks created config files."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        path1 = manager.create_config_file({"s1": MCPServerConfig(command="c1")})
        path2 = manager.create_config_file({"s2": MCPServerConfig(command="c2")})

        assert len(manager._config_files) == 2
        assert path1 in manager._config_files
        assert path2 in manager._config_files

        manager.cleanup()

    def test_server_without_args(self, tmp_path):
        """Test server config without args."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {"simple": MCPServerConfig(command="simple-cmd")}

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        server_config = config["mcpServers"]["simple"]
        assert server_config["command"] == "simple-cmd"
        assert "args" not in server_config

        manager.cleanup()

    def test_server_without_env(self, tmp_path):
        """Test server config without env."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {"noenv": MCPServerConfig(command="cmd", args=["arg"])}

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        server_config = config["mcpServers"]["noenv"]
        assert "env" not in server_config

        manager.cleanup()


class TestSSEAndHTTPTransport:
    """Tests for SSE and HTTP transport configurations."""

    def test_http_transport(self, tmp_path):
        """Test HTTP transport configuration."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {
            "remote": MCPServerConfig(
                command="",
                transport="http",
                url="https://api.example.com/mcp",
            ),
        }

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        server_config = config["mcpServers"]["remote"]
        assert server_config["type"] == "http"
        assert server_config["url"] == "https://api.example.com/mcp"

        manager.cleanup()

    def test_sse_transport(self, tmp_path):
        """Test SSE transport configuration."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {
            "sse_server": MCPServerConfig(
                command="",
                transport="sse",
                url="https://stream.example.com/events",
            ),
        }

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        server_config = config["mcpServers"]["sse_server"]
        assert server_config["type"] == "sse"
        assert server_config["url"] == "https://stream.example.com/events"

        manager.cleanup()

    def test_http_transport_with_headers(self, tmp_path):
        """Test HTTP transport with auth headers."""
        manager = MCPConfigManager(temp_dir=tmp_path)
        servers = {
            "authed": MCPServerConfig(
                command="",
                transport="http",
                url="https://api.example.com/mcp",
                env={"Authorization": "Bearer token123"},
            ),
        }

        config_path = manager.create_config_file(servers)

        with open(config_path) as f:
            config = json.load(f)

        server_config = config["mcpServers"]["authed"]
        assert server_config["headers"] == {"Authorization": "Bearer token123"}

        manager.cleanup()


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_removes_files(self, tmp_path):
        """Test cleanup removes all created files."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        path1 = manager.create_config_file({"s1": MCPServerConfig(command="c1")})
        path2 = manager.create_config_file({"s2": MCPServerConfig(command="c2")})

        assert path1.exists()
        assert path2.exists()

        manager.cleanup()

        assert not path1.exists()
        assert not path2.exists()

    def test_cleanup_clears_tracking_list(self, tmp_path):
        """Test cleanup clears the tracked files list."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        manager.create_config_file({"s": MCPServerConfig(command="c")})
        assert len(manager._config_files) == 1

        manager.cleanup()
        assert len(manager._config_files) == 0

    def test_cleanup_handles_already_deleted_files(self, tmp_path):
        """Test cleanup handles files that were already deleted."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        path = manager.create_config_file({"s": MCPServerConfig(command="c")})

        # Manually delete the file
        path.unlink()

        # Cleanup should not raise
        manager.cleanup()

    def test_cleanup_logs_warning_on_error(self, tmp_path):
        """Test cleanup logs warning when file deletion fails."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        path = manager.create_config_file({"s": MCPServerConfig(command="c")})

        # Make unlink raise an unexpected error
        with patch.object(Path, 'unlink', side_effect=PermissionError("Access denied")):
            # Should not raise, just log warning
            manager.cleanup()


class TestContextManager:
    """Tests for context manager interface."""

    def test_context_manager_enter(self, tmp_path):
        """Test __enter__ returns manager."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        with manager as m:
            assert m is manager

    def test_context_manager_cleanup_on_exit(self, tmp_path):
        """Test __exit__ calls cleanup."""
        with MCPConfigManager(temp_dir=tmp_path) as manager:
            path = manager.create_config_file({"s": MCPServerConfig(command="c")})
            assert path.exists()

        # After exiting context, file should be cleaned up
        assert not path.exists()

    def test_context_manager_cleanup_on_exception(self, tmp_path):
        """Test cleanup happens even if exception is raised."""
        path = None
        try:
            with MCPConfigManager(temp_dir=tmp_path) as manager:
                path = manager.create_config_file({"s": MCPServerConfig(command="c")})
                raise ValueError("Test error")
        except ValueError:
            pass

        # File should still be cleaned up
        assert path is not None
        assert not path.exists()


class TestErrorHandling:
    """Tests for error handling."""

    def test_create_config_wraps_exceptions(self, tmp_path):
        """Test exceptions are wrapped in MCPConfigError."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        # Mock _write_config to raise an error
        with patch.object(manager, '_write_config', side_effect=IOError("Write failed")):
            with pytest.raises(MCPConfigError) as exc_info:
                manager.create_config_file({"s": MCPServerConfig(command="c")})

            assert "Failed to create MCP config" in str(exc_info.value)

    def test_write_config_cleans_up_on_error(self, tmp_path):
        """Test _write_config cleans up file on write error."""
        manager = MCPConfigManager(temp_dir=tmp_path)

        # Count files before
        files_before = list(tmp_path.glob("*.json"))

        # Mock json.dump to fail
        with patch('json.dump', side_effect=TypeError("Not serializable")):
            with pytest.raises(MCPConfigError):
                manager.create_config_file({"s": MCPServerConfig(command="c")})

        # No orphaned files should remain
        files_after = list(tmp_path.glob("ccflow_mcp_*.json"))
        assert len(files_after) == len(files_before)


class TestConvenienceFunctions:
    """Tests for convenience server factory functions."""

    def test_github_server_default(self):
        """Test github_server with default token env."""
        config = github_server()

        assert config.command == "npx"
        assert "@anthropic-ai/mcp-server-github" in config.args
        assert "GITHUB_TOKEN" in config.env

    def test_github_server_custom_token(self):
        """Test github_server with custom token env."""
        config = github_server(token_env="GH_TOKEN")

        assert "GH_TOKEN" in config.env

    def test_postgres_server_default(self):
        """Test postgres_server with default connection env."""
        config = postgres_server()

        assert config.command == "python"
        assert "-m" in config.args
        assert "mcp_postgres" in config.args
        assert "DATABASE_URL" in config.env

    def test_postgres_server_custom_connection(self):
        """Test postgres_server with custom connection env."""
        config = postgres_server(connection_string_env="PG_URL")

        assert "PG_URL" in config.env

    def test_playwright_server(self):
        """Test playwright_server configuration."""
        config = playwright_server()

        assert config.command == "npx"
        assert "@playwright/mcp@latest" in config.args

    def test_filesystem_server_no_paths(self):
        """Test filesystem_server without allowed paths."""
        config = filesystem_server()

        assert config.command == "npx"
        assert "@anthropic-ai/mcp-server-filesystem" in config.args
        assert len(config.args) == 1

    def test_filesystem_server_with_paths(self):
        """Test filesystem_server with allowed paths."""
        config = filesystem_server(allowed_paths=["/home", "/tmp", "/var/data"])

        assert config.command == "npx"
        assert "@anthropic-ai/mcp-server-filesystem" in config.args
        assert "/home" in config.args
        assert "/tmp" in config.args
        assert "/var/data" in config.args
        assert len(config.args) == 4

    def test_custom_http_server_basic(self):
        """Test custom_http_server with just URL."""
        config = custom_http_server("https://api.example.com/mcp")

        assert config.transport == "http"
        assert config.url == "https://api.example.com/mcp"
        assert config.env == {}

    def test_custom_http_server_with_headers(self):
        """Test custom_http_server with headers."""
        headers = {
            "Authorization": "Bearer token",
            "X-Custom-Header": "value",
        }
        config = custom_http_server("https://api.example.com/mcp", headers=headers)

        assert config.transport == "http"
        assert config.env == headers


class TestIntegration:
    """Integration tests for MCP configuration."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow with multiple server types."""
        with MCPConfigManager(temp_dir=tmp_path) as manager:
            servers = {
                "github": github_server(),
                "playwright": playwright_server(),
                "filesystem": filesystem_server(["/home/user"]),
                "custom": custom_http_server(
                    "https://api.example.com",
                    {"Authorization": "Bearer xyz"},
                ),
            }

            config_path = manager.create_config_file(servers)

            # Verify file exists and is valid JSON
            assert config_path.exists()

            with open(config_path) as f:
                config = json.load(f)

            # Verify all servers present
            assert len(config["mcpServers"]) == 4
            assert "github" in config["mcpServers"]
            assert "playwright" in config["mcpServers"]
            assert "filesystem" in config["mcpServers"]
            assert "custom" in config["mcpServers"]

            # Verify different transport types
            assert "command" in config["mcpServers"]["github"]
            assert config["mcpServers"]["custom"]["type"] == "http"

        # After context exit, file should be cleaned up
        assert not config_path.exists()
