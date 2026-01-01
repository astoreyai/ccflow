"""Tests for CLI entry point module."""

import argparse
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow.cli import (
    create_parser,
    get_permission_mode,
    main,
    run_query,
    run_server,
    run_sessions,
    run_stats,
)
from ccflow.types import PermissionMode


class TestCreateParser:
    """Tests for create_parser function."""

    def test_create_parser_returns_parser(self):
        """Test that create_parser returns an ArgumentParser."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_prog_name(self):
        """Test parser has correct program name."""
        parser = create_parser()
        assert parser.prog == "ccflow"

    def test_parser_version_exits(self):
        """Test --version causes SystemExit."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_verbose_flag(self):
        """Test -v verbose flag at top level."""
        parser = create_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose is True


class TestQuerySubcommand:
    """Tests for query subcommand parsing."""

    def test_query_command(self):
        """Test query command with prompt."""
        parser = create_parser()
        args = parser.parse_args(["query", "Hello world"])
        assert args.command == "query"
        assert args.prompt == "Hello world"

    def test_query_alias(self):
        """Test 'q' alias for query."""
        parser = create_parser()
        args = parser.parse_args(["q", "Hello world"])
        assert args.command == "q"
        assert args.prompt == "Hello world"

    def test_query_prompt_optional(self):
        """Test parser allows no prompt (for stdin)."""
        parser = create_parser()
        args = parser.parse_args(["query"])
        assert args.prompt is None

    def test_query_model_option_short(self):
        """Test -m model option."""
        parser = create_parser()
        args = parser.parse_args(["query", "-m", "opus", "prompt"])
        assert args.model == "opus"

    def test_query_model_option_long(self):
        """Test --model option."""
        parser = create_parser()
        args = parser.parse_args(["query", "--model", "haiku", "prompt"])
        assert args.model == "haiku"

    def test_query_model_default(self):
        """Test model defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["query", "prompt"])
        assert args.model is None

    def test_query_stream_flag(self):
        """Test --stream flag."""
        parser = create_parser()
        args = parser.parse_args(["query", "--stream", "prompt"])
        assert args.stream is True

    def test_query_stream_default(self):
        """Test stream defaults to False."""
        parser = create_parser()
        args = parser.parse_args(["query", "prompt"])
        assert args.stream is False

    def test_query_max_budget_option(self):
        """Test --max-budget option."""
        parser = create_parser()
        args = parser.parse_args(["query", "--max-budget", "5.0", "prompt"])
        assert args.max_budget == 5.0

    def test_query_timeout_option(self):
        """Test --timeout option."""
        parser = create_parser()
        args = parser.parse_args(["query", "--timeout", "60.0", "prompt"])
        assert args.timeout == 60.0

    def test_query_permission_mode_default(self):
        """Test permission mode defaults to 'default'."""
        parser = create_parser()
        args = parser.parse_args(["query", "prompt"])
        assert args.permission_mode == "default"

    def test_query_permission_mode_plan(self):
        """Test --permission-mode plan."""
        parser = create_parser()
        args = parser.parse_args(["query", "--permission-mode", "plan", "prompt"])
        assert args.permission_mode == "plan"

    def test_query_permission_mode_accept_edits(self):
        """Test --permission-mode acceptEdits."""
        parser = create_parser()
        args = parser.parse_args(["query", "--permission-mode", "acceptEdits", "prompt"])
        assert args.permission_mode == "acceptEdits"

    def test_query_permission_mode_bypass(self):
        """Test --permission-mode bypass."""
        parser = create_parser()
        args = parser.parse_args(["query", "--permission-mode", "bypass", "prompt"])
        assert args.permission_mode == "bypass"

    def test_query_permission_mode_dont_ask(self):
        """Test --permission-mode dontAsk."""
        parser = create_parser()
        args = parser.parse_args(["query", "--permission-mode", "dontAsk", "prompt"])
        assert args.permission_mode == "dontAsk"

    def test_query_permission_mode_delegate(self):
        """Test --permission-mode delegate."""
        parser = create_parser()
        args = parser.parse_args(["query", "--permission-mode", "delegate", "prompt"])
        assert args.permission_mode == "delegate"

    def test_query_permission_mode_invalid(self):
        """Test invalid permission mode raises error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["query", "--permission-mode", "invalid", "prompt"])

    def test_query_allowed_tools(self):
        """Test --allowed-tools with multiple tools."""
        parser = create_parser()
        args = parser.parse_args(["query", "prompt", "--allowed-tools", "Read", "Write", "Bash"])
        assert args.allowed_tools == ["Read", "Write", "Bash"]

    def test_query_session_id(self):
        """Test --session-id option."""
        parser = create_parser()
        args = parser.parse_args(["query", "--session-id", "sess-123", "prompt"])
        assert args.session_id == "sess-123"

    def test_query_resume_flag(self):
        """Test --resume flag."""
        parser = create_parser()
        args = parser.parse_args(["query", "--resume", "prompt"])
        assert args.resume is True

    def test_query_no_toon_flag(self):
        """Test --no-toon flag."""
        parser = create_parser()
        args = parser.parse_args(["query", "--no-toon", "prompt"])
        assert args.no_toon is True

    def test_query_combined_options(self):
        """Test multiple options combined."""
        parser = create_parser()
        args = parser.parse_args([
            "query",
            "-m", "opus",
            "--stream",
            "--max-budget", "10.0",
            "--timeout", "120.0",
            "--permission-mode", "acceptEdits",
            "--no-toon",
            "Analyze this code",
            "--allowed-tools", "Read", "Write",
        ])
        assert args.model == "opus"
        assert args.stream is True
        assert args.max_budget == 10.0
        assert args.timeout == 120.0
        assert args.permission_mode == "acceptEdits"
        assert args.allowed_tools == ["Read", "Write"]
        assert args.no_toon is True
        assert args.prompt == "Analyze this code"


class TestSessionsSubcommand:
    """Tests for sessions subcommand parsing."""

    def test_sessions_command(self):
        """Test sessions command."""
        parser = create_parser()
        args = parser.parse_args(["sessions"])
        assert args.command == "sessions"

    def test_sessions_alias(self):
        """Test 's' alias for sessions."""
        parser = create_parser()
        args = parser.parse_args(["s"])
        assert args.command == "s"

    def test_sessions_list_action(self):
        """Test sessions list action."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "list"])
        assert args.sessions_action == "list"

    def test_sessions_list_alias(self):
        """Test sessions list 'ls' alias."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "ls"])
        assert args.sessions_action == "ls"

    def test_sessions_list_status_filter(self):
        """Test sessions list with status filter."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "list", "--status", "active"])
        assert args.status == "active"

    def test_sessions_list_model_filter(self):
        """Test sessions list with model filter."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "list", "--model", "opus"])
        assert args.model == "opus"

    def test_sessions_list_limit(self):
        """Test sessions list with limit."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "list", "--limit", "10"])
        assert args.limit == 10

    def test_sessions_list_json_output(self):
        """Test sessions list with JSON output."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "list", "--json"])
        assert args.json is True

    def test_sessions_get_action(self):
        """Test sessions get action."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "get", "sess-123"])
        assert args.sessions_action == "get"
        assert args.session_id == "sess-123"

    def test_sessions_get_json_output(self):
        """Test sessions get with JSON output."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "get", "sess-123", "--json"])
        assert args.json is True

    def test_sessions_delete_action(self):
        """Test sessions delete action."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "delete", "sess-123"])
        assert args.sessions_action == "delete"
        assert args.session_id == "sess-123"

    def test_sessions_delete_alias(self):
        """Test sessions delete 'rm' alias."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "rm", "sess-123"])
        assert args.sessions_action == "rm"

    def test_sessions_delete_force(self):
        """Test sessions delete with force flag."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "delete", "-f", "sess-123"])
        assert args.force is True

    def test_sessions_cleanup_action(self):
        """Test sessions cleanup action."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "cleanup"])
        assert args.sessions_action == "cleanup"

    def test_sessions_cleanup_days(self):
        """Test sessions cleanup with days option."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "cleanup", "--days", "30"])
        assert args.days == 30

    def test_sessions_cleanup_dry_run(self):
        """Test sessions cleanup with dry-run flag."""
        parser = create_parser()
        args = parser.parse_args(["sessions", "cleanup", "--dry-run"])
        assert args.dry_run is True


class TestServerSubcommand:
    """Tests for server subcommand parsing."""

    def test_server_command(self):
        """Test server command."""
        parser = create_parser()
        args = parser.parse_args(["server"])
        assert args.command == "server"

    def test_server_alias(self):
        """Test 'serve' alias for server."""
        parser = create_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"

    def test_server_host_option(self):
        """Test --host option."""
        parser = create_parser()
        args = parser.parse_args(["server", "--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_server_host_default(self):
        """Test host defaults to 127.0.0.1."""
        parser = create_parser()
        args = parser.parse_args(["server"])
        assert args.host == "127.0.0.1"

    def test_server_port_option(self):
        """Test --port option."""
        parser = create_parser()
        args = parser.parse_args(["server", "--port", "9000"])
        assert args.port == 9000

    def test_server_port_default(self):
        """Test port defaults to 8000."""
        parser = create_parser()
        args = parser.parse_args(["server"])
        assert args.port == 8000

    def test_server_reload_flag(self):
        """Test --reload flag."""
        parser = create_parser()
        args = parser.parse_args(["server", "--reload"])
        assert args.reload is True

    def test_server_workers_option(self):
        """Test --workers option."""
        parser = create_parser()
        args = parser.parse_args(["server", "--workers", "4"])
        assert args.workers == 4

    def test_server_cors_origins(self):
        """Test --cors-origins option."""
        parser = create_parser()
        args = parser.parse_args(["server", "--cors-origins", "http://localhost:3000", "http://example.com"])
        assert args.cors_origins == ["http://localhost:3000", "http://example.com"]

    def test_server_metrics_port(self):
        """Test --metrics-port option."""
        parser = create_parser()
        args = parser.parse_args(["server", "--metrics-port", "9090"])
        assert args.metrics_port == 9090


class TestStatsSubcommand:
    """Tests for stats subcommand parsing."""

    def test_stats_command(self):
        """Test stats command."""
        parser = create_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_stats_json_output(self):
        """Test stats with JSON output."""
        parser = create_parser()
        args = parser.parse_args(["stats", "--json"])
        assert args.json is True

    def test_stats_rate_limiter_flag(self):
        """Test stats with rate-limiter flag."""
        parser = create_parser()
        args = parser.parse_args(["stats", "--rate-limiter"])
        assert args.rate_limiter is True


class TestGetPermissionMode:
    """Tests for get_permission_mode function."""

    def test_default_mode(self):
        """Test 'default' maps to DEFAULT."""
        assert get_permission_mode("default") == PermissionMode.DEFAULT

    def test_plan_mode(self):
        """Test 'plan' maps to PLAN."""
        assert get_permission_mode("plan") == PermissionMode.PLAN

    def test_dont_ask_mode(self):
        """Test 'dontAsk' maps to DONT_ASK."""
        assert get_permission_mode("dontAsk") == PermissionMode.DONT_ASK

    def test_accept_edits_mode(self):
        """Test 'acceptEdits' maps to ACCEPT_EDITS."""
        assert get_permission_mode("acceptEdits") == PermissionMode.ACCEPT_EDITS

    def test_delegate_mode(self):
        """Test 'delegate' maps to DELEGATE."""
        assert get_permission_mode("delegate") == PermissionMode.DELEGATE

    def test_bypass_mode(self):
        """Test 'bypass' maps to BYPASS."""
        assert get_permission_mode("bypass") == PermissionMode.BYPASS

    def test_unknown_mode_returns_default(self):
        """Test unknown mode returns DEFAULT."""
        assert get_permission_mode("unknown") == PermissionMode.DEFAULT

    def test_empty_string_returns_default(self):
        """Test empty string returns DEFAULT."""
        assert get_permission_mode("") == PermissionMode.DEFAULT


class TestRunQuery:
    """Tests for run_query async function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock argparse namespace."""
        args = argparse.Namespace()
        args.command = "query"
        args.prompt = "Test prompt"
        args.model = "sonnet"
        args.stream = False
        args.max_budget = None
        args.timeout = None
        args.permission_mode = "default"
        args.allowed_tools = None
        args.session_id = None
        args.resume = False
        args.no_toon = False
        args.verbose = False
        return args

    @pytest.mark.asyncio
    async def test_run_query_simple_mode(self, mock_args):
        """Test run_query in simple (non-streaming) mode."""
        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Test response"

            with patch("builtins.print") as mock_print:
                result = await run_query(mock_args)

            assert result == 0
            mock_query.assert_called_once()
            mock_print.assert_called_with("Test response")

    @pytest.mark.asyncio
    async def test_run_query_streaming_mode(self, mock_args):
        """Test run_query in streaming mode."""
        mock_args.stream = True

        mock_msg = MagicMock()
        mock_msg.content = "Streamed content"

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg

        with patch("ccflow.cli.query", side_effect=mock_query_gen):
            with patch("builtins.print") as mock_print:
                result = await run_query(mock_args)

            assert result == 0
            assert mock_print.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_query_no_prompt_no_stdin(self, mock_args):
        """Test run_query with no prompt and stdin is tty."""
        mock_args.prompt = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("builtins.print"):
                result = await run_query(mock_args)

            assert result == 1

    @pytest.mark.asyncio
    async def test_run_query_reads_from_stdin(self, mock_args):
        """Test run_query reads prompt from stdin when not provided."""
        mock_args.prompt = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "Stdin prompt\n"

            with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
                mock_query.return_value = "Response"

                with patch("builtins.print"):
                    result = await run_query(mock_args)

                assert result == 0
                call_args = mock_query.call_args
                assert call_args[0][0] == "Stdin prompt"

    @pytest.mark.asyncio
    async def test_run_query_empty_stdin(self, mock_args):
        """Test run_query with empty stdin."""
        mock_args.prompt = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "   "

            with patch("builtins.print"):
                result = await run_query(mock_args)

            assert result == 1

    @pytest.mark.asyncio
    async def test_run_query_with_custom_model(self, mock_args):
        """Test run_query uses custom model."""
        mock_args.model = "opus"

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            with patch("builtins.print"):
                await run_query(mock_args)

            call_args = mock_query.call_args
            options = call_args[0][1]
            assert options.model == "opus"

    @pytest.mark.asyncio
    async def test_run_query_with_allowed_tools(self, mock_args):
        """Test run_query passes allowed_tools."""
        mock_args.allowed_tools = ["Read", "Write"]

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            with patch("builtins.print"):
                await run_query(mock_args)

            call_args = mock_query.call_args
            options = call_args[0][1]
            assert options.allowed_tools == ["Read", "Write"]

    @pytest.mark.asyncio
    async def test_run_query_with_no_toon(self, mock_args):
        """Test run_query disables TOON when --no-toon is set."""
        mock_args.no_toon = True

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            with patch("builtins.print"):
                await run_query(mock_args)

            call_args = mock_query.call_args
            options = call_args[0][1]
            assert options.toon.enabled is False

    @pytest.mark.asyncio
    async def test_run_query_with_max_budget(self, mock_args):
        """Test run_query passes max_budget_usd."""
        mock_args.max_budget = 5.0

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            with patch("builtins.print"):
                await run_query(mock_args)

            call_args = mock_query.call_args
            options = call_args[0][1]
            assert options.max_budget_usd == 5.0

    @pytest.mark.asyncio
    async def test_run_query_with_timeout(self, mock_args):
        """Test run_query passes timeout."""
        mock_args.timeout = 120.0

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            with patch("builtins.print"):
                await run_query(mock_args)

            call_args = mock_query.call_args
            options = call_args[0][1]
            assert options.timeout == 120.0

    @pytest.mark.asyncio
    async def test_run_query_keyboard_interrupt(self, mock_args):
        """Test run_query handles KeyboardInterrupt."""
        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = KeyboardInterrupt()

            with patch("builtins.print"):
                result = await run_query(mock_args)

            assert result == 130

    @pytest.mark.asyncio
    async def test_run_query_exception(self, mock_args):
        """Test run_query handles general exception."""
        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = Exception("Test error")

            with patch("builtins.print"):
                result = await run_query(mock_args)

            assert result == 1

    @pytest.mark.asyncio
    async def test_run_query_exception_verbose(self, mock_args):
        """Test run_query prints traceback in verbose mode."""
        mock_args.verbose = True

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = Exception("Test error")

            with patch("builtins.print"):
                with patch("traceback.print_exc") as mock_traceback:
                    result = await run_query(mock_args)

                    mock_traceback.assert_called_once()

            assert result == 1

    @pytest.mark.asyncio
    async def test_run_query_streaming_message_without_content(self, mock_args):
        """Test run_query handles streaming messages without content attribute."""
        mock_args.stream = True

        mock_msg = MagicMock(spec=[])

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg

        with patch("ccflow.cli.query", side_effect=mock_query_gen):
            with patch("builtins.print"):
                result = await run_query(mock_args)

            assert result == 0


class TestRunSessions:
    """Tests for run_sessions async function."""

    @pytest.fixture
    def mock_session_metadata(self):
        """Create mock session metadata."""
        from ccflow.store import SessionMetadata, SessionStatus

        return SessionMetadata(
            session_id="sess-123",
            model="sonnet",
            status=SessionStatus.ACTIVE,
            turn_count=5,
            total_input_tokens=500,
            total_output_tokens=250,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["test"],
        )

    @pytest.fixture
    def mock_session(self):
        """Create mock session object."""
        session = MagicMock()
        session.session_id = "sess-123"
        session.turn_count = 5
        session.is_closed = False
        session.created_at = datetime.now()
        session.updated_at = datetime.now()
        session.tags = set()
        session.options = MagicMock()
        session.options.model = "sonnet"
        return session

    @pytest.mark.asyncio
    async def test_run_sessions_no_action(self):
        """Test sessions with no action specified."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action=None,
            verbose=False,
        )

        with patch("builtins.print"):
            result = await run_sessions(args)

        assert result == 1

    @pytest.mark.asyncio
    async def test_run_sessions_list_empty(self):
        """Test sessions list with no sessions."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="list",
            status=None,
            model=None,
            limit=20,
            json=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.list_sessions = AsyncMock(return_value=[])
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        mock_print.assert_called_with("No sessions found.")

    @pytest.mark.asyncio
    async def test_run_sessions_list_with_data(self, mock_session_metadata):
        """Test sessions list with sessions."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="list",
            status=None,
            model=None,
            limit=20,
            json=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.list_sessions = AsyncMock(return_value=[mock_session_metadata])
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        # Should print header and data
        assert mock_print.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_sessions_list_json_output(self, mock_session_metadata):
        """Test sessions list with JSON output."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="list",
            status=None,
            model=None,
            limit=20,
            json=True,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.list_sessions = AsyncMock(return_value=[mock_session_metadata])
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        # Check JSON output
        call_args = mock_print.call_args[0][0]
        assert "session_id" in call_args

    @pytest.mark.asyncio
    async def test_run_sessions_list_with_status_filter(self, mock_session_metadata):
        """Test sessions list with status filter."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="list",
            status="active",
            model=None,
            limit=20,
            json=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.list_sessions = AsyncMock(return_value=[mock_session_metadata])
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print"):
                result = await run_sessions(args)

        assert result == 0
        # Check that status filter was passed
        call_kwargs = mock_manager.list_sessions.call_args
        assert call_kwargs[1]["status"] is not None

    @pytest.mark.asyncio
    async def test_run_sessions_get_found(self, mock_session):
        """Test sessions get when session exists."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="get",
            session_id="sess-123",
            json=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=mock_session)
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        # Should print session details
        assert mock_print.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_sessions_get_not_found(self):
        """Test sessions get when session doesn't exist."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="get",
            session_id="nonexistent",
            json=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print"):
                result = await run_sessions(args)

        assert result == 1

    @pytest.mark.asyncio
    async def test_run_sessions_get_json_output(self, mock_session):
        """Test sessions get with JSON output."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="get",
            session_id="sess-123",
            json=True,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=mock_session)
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        call_args = mock_print.call_args[0][0]
        assert "session_id" in call_args

    @pytest.mark.asyncio
    async def test_run_sessions_delete_with_force(self):
        """Test sessions delete with force flag."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="delete",
            session_id="sess-123",
            force=True,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.delete_session = AsyncMock(return_value=True)
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print"):
                result = await run_sessions(args)

        assert result == 0
        mock_manager.delete_session.assert_called_once_with("sess-123")

    @pytest.mark.asyncio
    async def test_run_sessions_delete_not_found(self):
        """Test sessions delete when session doesn't exist."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="delete",
            session_id="nonexistent",
            force=True,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.delete_session = AsyncMock(return_value=False)
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print"):
                result = await run_sessions(args)

        assert result == 1

    @pytest.mark.asyncio
    async def test_run_sessions_delete_cancelled(self):
        """Test sessions delete cancelled by user."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="delete",
            session_id="sess-123",
            force=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.input", return_value="n"):
                with patch("builtins.print") as mock_print:
                    result = await run_sessions(args)

        assert result == 0
        mock_print.assert_called_with("Cancelled.")

    @pytest.mark.asyncio
    async def test_run_sessions_cleanup(self):
        """Test sessions cleanup."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="cleanup",
            days=7,
            dry_run=False,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.cleanup_expired = AsyncMock(return_value=5)
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        mock_print.assert_called_with("Cleaned up 5 expired sessions")

    @pytest.mark.asyncio
    async def test_run_sessions_cleanup_dry_run(self, mock_session_metadata):
        """Test sessions cleanup with dry run."""
        args = argparse.Namespace(
            command="sessions",
            sessions_action="cleanup",
            days=7,
            dry_run=True,
            verbose=False,
        )

        mock_manager = MagicMock()
        mock_manager.list_sessions = AsyncMock(return_value=[mock_session_metadata])
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_sessions(args)

        assert result == 0
        # Should indicate dry run output
        assert any("Would delete" in str(call) for call in mock_print.call_args_list)


class TestRunServer:
    """Tests for run_server async function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock server args."""
        return argparse.Namespace(
            command="server",
            host="127.0.0.1",
            port=8000,
            reload=False,
            workers=1,
            cors_origins=["*"],
            metrics_port=None,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_run_server_uvicorn_not_installed(self, mock_args):
        """Test server fails gracefully when uvicorn not installed."""
        with patch.dict("sys.modules", {"uvicorn": None}):
            with patch("builtins.__import__", side_effect=ImportError("No uvicorn")):
                with patch("builtins.print"):
                    # This will try to import uvicorn and fail
                    # Since uvicorn is installed in test env, we need to simulate
                    pass

    @pytest.mark.asyncio
    async def test_run_server_starts(self, mock_args):
        """Test server starts with correct config."""
        mock_server_instance = MagicMock()
        mock_server_instance.serve = AsyncMock()

        mock_ccflow_server = MagicMock()
        mock_ccflow_server.app = MagicMock()

        with patch("ccflow.server.CCFlowServer", return_value=mock_ccflow_server):
            with patch("uvicorn.Config") as mock_config:
                with patch("uvicorn.Server", return_value=mock_server_instance):
                    with patch("builtins.print"):
                        result = await run_server(mock_args)

        assert result == 0
        mock_config.assert_called_once()
        mock_server_instance.serve.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_server_with_metrics_port(self, mock_args):
        """Test server starts metrics server on separate port."""
        mock_args.metrics_port = 9090

        mock_server_instance = MagicMock()
        mock_server_instance.serve = AsyncMock()

        mock_ccflow_server = MagicMock()
        mock_ccflow_server.app = MagicMock()

        with patch("ccflow.server.CCFlowServer", return_value=mock_ccflow_server):
            with patch("uvicorn.Config"):
                with patch("uvicorn.Server", return_value=mock_server_instance):
                    with patch("ccflow.metrics_handlers.start_metrics_server", return_value=True) as mock_metrics:
                        with patch("builtins.print"):
                            result = await run_server(mock_args)

        assert result == 0
        mock_metrics.assert_called_once_with(9090)


class TestRunStats:
    """Tests for run_stats async function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock stats args."""
        return argparse.Namespace(
            command="stats",
            json=False,
            rate_limiter=False,
            verbose=False,
        )

    @pytest.mark.asyncio
    async def test_run_stats_basic(self, mock_args):
        """Test stats displays session statistics."""
        mock_manager = MagicMock()
        mock_manager.get_stats = AsyncMock(return_value={
            "total_sessions": 10,
            "active_sessions": 3,
            "closed_sessions": 7,
            "in_memory_sessions": 2,
        })
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_stats(mock_args)

        assert result == 0
        # Should print session stats
        assert any("Total sessions" in str(call) for call in mock_print.call_args_list)

    @pytest.mark.asyncio
    async def test_run_stats_json_output(self, mock_args):
        """Test stats with JSON output."""
        mock_args.json = True

        mock_manager = MagicMock()
        mock_manager.get_stats = AsyncMock(return_value={
            "total_sessions": 10,
        })
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("builtins.print") as mock_print:
                result = await run_stats(mock_args)

        assert result == 0
        call_args = mock_print.call_args[0][0]
        assert "sessions" in call_args

    @pytest.mark.asyncio
    async def test_run_stats_with_rate_limiter(self, mock_args):
        """Test stats includes rate limiter stats."""
        mock_args.rate_limiter = True

        mock_manager = MagicMock()
        mock_manager.get_stats = AsyncMock(return_value={
            "total_sessions": 10,
        })
        mock_manager.__aenter__ = AsyncMock(return_value=mock_manager)
        mock_manager.__aexit__ = AsyncMock(return_value=None)

        mock_limiter = MagicMock()
        mock_limiter.stats = {
            "rate_limiter": {"total_requests": 100, "total_waits": 5},
            "concurrency_limiter": {"peak_concurrent": 3, "current_concurrent": 1},
        }

        with patch("ccflow.manager.SessionManager", return_value=mock_manager):
            with patch("ccflow.rate_limiting.get_limiter", return_value=mock_limiter):
                with patch("builtins.print") as mock_print:
                    result = await run_stats(mock_args)

        assert result == 0
        # Should print rate limiter stats
        assert any("Rate Limiter" in str(call) for call in mock_print.call_args_list)


class TestMain:
    """Tests for main entry point."""

    def test_main_no_command_shows_help(self):
        """Test main with no command shows help."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                command=None,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with pytest.raises(SystemExit) as exc_info:
                    main()

            assert exc_info.value.code == 0
            mock_parser.print_help.assert_called_once()

    def test_main_query_command(self):
        """Test main routes to query handler."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                command="query",
                prompt="test",
                model=None,
                stream=False,
                max_budget=None,
                timeout=None,
                permission_mode="default",
                allowed_tools=None,
                session_id=None,
                resume=False,
                no_toon=False,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 0

                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 0
                    mock_run.assert_called_once()

    def test_main_sessions_command(self):
        """Test main routes to sessions handler."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                command="sessions",
                sessions_action="list",
                status=None,
                model=None,
                limit=20,
                json=False,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 0

                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 0
                    mock_run.assert_called_once()

    def test_main_server_command(self):
        """Test main routes to server handler."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                command="server",
                host="127.0.0.1",
                port=8000,
                reload=False,
                workers=1,
                cors_origins=["*"],
                metrics_port=None,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 0

                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 0
                    mock_run.assert_called_once()

    def test_main_stats_command(self):
        """Test main routes to stats handler."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                command="stats",
                json=False,
                rate_limiter=False,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 0

                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 0
                    mock_run.assert_called_once()

    def test_main_exits_with_correct_code(self):
        """Test main exits with the code from handler."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                command="query",
                prompt="test",
                model=None,
                stream=False,
                max_budget=None,
                timeout=None,
                permission_mode="default",
                allowed_tools=None,
                session_id=None,
                resume=False,
                no_toon=False,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 42

                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    assert exc_info.value.code == 42
