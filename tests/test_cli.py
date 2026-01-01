"""Tests for CLI entry point module."""

import argparse
import asyncio
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow.cli import create_parser, get_permission_mode, run_query, main
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

    def test_parser_accepts_prompt(self):
        """Test parser accepts positional prompt argument."""
        parser = create_parser()
        args = parser.parse_args(["Hello world"])
        assert args.prompt == "Hello world"

    def test_parser_prompt_optional(self):
        """Test parser allows no prompt (for stdin)."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.prompt is None

    def test_parser_model_option_short(self):
        """Test -m model option."""
        parser = create_parser()
        args = parser.parse_args(["-m", "opus", "prompt"])
        assert args.model == "opus"

    def test_parser_model_option_long(self):
        """Test --model option."""
        parser = create_parser()
        args = parser.parse_args(["--model", "haiku", "prompt"])
        assert args.model == "haiku"

    def test_parser_model_default(self):
        """Test model defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.model is None

    def test_parser_stream_flag(self):
        """Test --stream flag."""
        parser = create_parser()
        args = parser.parse_args(["--stream", "prompt"])
        assert args.stream is True

    def test_parser_stream_default(self):
        """Test stream defaults to False."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.stream is False

    def test_parser_max_budget_option(self):
        """Test --max-budget option."""
        parser = create_parser()
        args = parser.parse_args(["--max-budget", "5.0", "prompt"])
        assert args.max_budget == 5.0

    def test_parser_max_budget_default(self):
        """Test max_budget defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.max_budget is None

    def test_parser_timeout_option(self):
        """Test --timeout option."""
        parser = create_parser()
        args = parser.parse_args(["--timeout", "60.0", "prompt"])
        assert args.timeout == 60.0

    def test_parser_timeout_default(self):
        """Test timeout defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.timeout is None

    def test_parser_permission_mode_default(self):
        """Test permission mode defaults to 'default'."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.permission_mode == "default"

    def test_parser_permission_mode_plan(self):
        """Test --permission-mode plan."""
        parser = create_parser()
        args = parser.parse_args(["--permission-mode", "plan", "prompt"])
        assert args.permission_mode == "plan"

    def test_parser_permission_mode_accept_edits(self):
        """Test --permission-mode acceptEdits."""
        parser = create_parser()
        args = parser.parse_args(["--permission-mode", "acceptEdits", "prompt"])
        assert args.permission_mode == "acceptEdits"

    def test_parser_permission_mode_bypass(self):
        """Test --permission-mode bypass."""
        parser = create_parser()
        args = parser.parse_args(["--permission-mode", "bypass", "prompt"])
        assert args.permission_mode == "bypass"

    def test_parser_permission_mode_dont_ask(self):
        """Test --permission-mode dontAsk."""
        parser = create_parser()
        args = parser.parse_args(["--permission-mode", "dontAsk", "prompt"])
        assert args.permission_mode == "dontAsk"

    def test_parser_permission_mode_delegate(self):
        """Test --permission-mode delegate."""
        parser = create_parser()
        args = parser.parse_args(["--permission-mode", "delegate", "prompt"])
        assert args.permission_mode == "delegate"

    def test_parser_permission_mode_invalid(self):
        """Test invalid permission mode raises error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--permission-mode", "invalid", "prompt"])

    def test_parser_allowed_tools(self):
        """Test --allowed-tools with multiple tools."""
        parser = create_parser()
        # Prompt must come before --allowed-tools since nargs="+" consumes rest
        args = parser.parse_args(["prompt", "--allowed-tools", "Read", "Write", "Bash"])
        assert args.allowed_tools == ["Read", "Write", "Bash"]

    def test_parser_allowed_tools_default(self):
        """Test allowed_tools defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.allowed_tools is None

    def test_parser_no_toon_flag(self):
        """Test --no-toon flag."""
        parser = create_parser()
        args = parser.parse_args(["--no-toon", "prompt"])
        assert args.no_toon is True

    def test_parser_no_toon_default(self):
        """Test no_toon defaults to False."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.no_toon is False

    def test_parser_verbose_flag_short(self):
        """Test -v verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["-v", "prompt"])
        assert args.verbose is True

    def test_parser_verbose_flag_long(self):
        """Test --verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "prompt"])
        assert args.verbose is True

    def test_parser_verbose_default(self):
        """Test verbose defaults to False."""
        parser = create_parser()
        args = parser.parse_args(["prompt"])
        assert args.verbose is False

    def test_parser_version_exits(self):
        """Test --version causes SystemExit."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_combined_options(self):
        """Test multiple options combined."""
        parser = create_parser()
        # Prompt before --allowed-tools since nargs="+" consumes rest
        args = parser.parse_args([
            "-m", "opus",
            "--stream",
            "--max-budget", "10.0",
            "--timeout", "120.0",
            "--permission-mode", "acceptEdits",
            "--no-toon",
            "-v",
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
        assert args.verbose is True
        assert args.prompt == "Analyze this code"


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
        args.prompt = "Test prompt"
        args.model = "sonnet"
        args.stream = False
        args.max_budget = None
        args.timeout = None
        args.permission_mode = "default"
        args.allowed_tools = None
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

        # Create mock message with content attribute
        mock_msg = MagicMock()
        mock_msg.content = "Streamed content"

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg

        with patch("ccflow.cli.query", side_effect=mock_query_gen):
            with patch("builtins.print") as mock_print:
                result = await run_query(mock_args)

            assert result == 0
            # Should print content and final newline
            assert mock_print.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_query_no_prompt_no_stdin(self, mock_args):
        """Test run_query with no prompt and stdin is tty."""
        mock_args.prompt = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
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
                # Check that the prompt from stdin was used
                call_args = mock_query.call_args
                assert call_args[0][0] == "Stdin prompt"

    @pytest.mark.asyncio
    async def test_run_query_empty_stdin(self, mock_args):
        """Test run_query with empty stdin."""
        mock_args.prompt = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.read.return_value = "   "  # Whitespace only

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
    async def test_run_query_with_verbose(self, mock_args):
        """Test run_query sets verbose option."""
        mock_args.verbose = True

        with patch("ccflow.cli.query_simple", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            with patch("builtins.print"):
                await run_query(mock_args)

            call_args = mock_query.call_args
            options = call_args[0][1]
            assert options.verbose is True

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

        # Create mock message without content attribute
        mock_msg = MagicMock(spec=[])  # No attributes

        async def mock_query_gen(*args, **kwargs):
            yield mock_msg

        with patch("ccflow.cli.query", side_effect=mock_query_gen):
            with patch("builtins.print") as mock_print:
                result = await run_query(mock_args)

            assert result == 0


class TestMain:
    """Tests for main entry point."""

    def test_main_calls_parser(self):
        """Test main creates parser and parses args."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                prompt="test",
                model=None,
                stream=False,
                max_budget=None,
                timeout=None,
                permission_mode="default",
                allowed_tools=None,
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

            mock_create.assert_called_once()

    def test_main_configures_logging(self):
        """Test main calls configure_logging."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                prompt="test",
                model=None,
                stream=False,
                max_budget=None,
                timeout=None,
                permission_mode="default",
                allowed_tools=None,
                no_toon=False,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging") as mock_logging:
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 0

                    with pytest.raises(SystemExit):
                        main()

                    mock_logging.assert_called_once()

    def test_main_runs_async(self):
        """Test main calls asyncio.run with run_query."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                prompt="test",
                model=None,
                stream=False,
                max_budget=None,
                timeout=None,
                permission_mode="default",
                allowed_tools=None,
                no_toon=False,
                verbose=False,
            )
            mock_parser.parse_args.return_value = mock_args
            mock_create.return_value = mock_parser

            with patch("ccflow.cli.configure_logging"):
                with patch("ccflow.cli.asyncio.run") as mock_run:
                    mock_run.return_value = 0

                    with pytest.raises(SystemExit):
                        main()

                    mock_run.assert_called_once()

    def test_main_exits_with_correct_code(self):
        """Test main exits with the code from run_query."""
        with patch("ccflow.cli.create_parser") as mock_create:
            mock_parser = MagicMock()
            mock_args = argparse.Namespace(
                prompt="test",
                model=None,
                stream=False,
                max_budget=None,
                timeout=None,
                permission_mode="default",
                allowed_tools=None,
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
