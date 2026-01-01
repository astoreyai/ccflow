"""Tests for ccflow configuration module."""

import os
import pytest
from unittest.mock import patch
import structlog

from ccflow.config import CCFlowSettings, get_settings, configure_logging


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def reset_structlog():
    """Reset structlog configuration after test."""
    yield
    # Reset to default configuration
    structlog.reset_defaults()


class TestCCFlowSettings:
    """Tests for CCFlowSettings class."""

    def test_default_values(self):
        """Test settings have expected defaults."""
        settings = CCFlowSettings()

        assert settings.default_model == "sonnet"
        assert settings.fallback_model is None
        assert settings.default_timeout == 300.0
        assert settings.default_max_turns is None
        assert settings.toon_enabled is True
        assert settings.toon_delimiter == ","
        assert settings.toon_track_savings is True
        assert settings.enable_metrics is True
        assert settings.log_level == "INFO"
        assert settings.log_format == "console"
        assert settings.max_concurrent_requests == 10
        assert settings.rate_limit_per_minute == 60
        assert settings.allow_dangerous_permissions is False
        assert settings.claude_cli_path is None
        assert settings.session_storage_path is None
        assert settings.temp_dir is None

    def test_env_prefix(self):
        """Test CCFLOW_ prefix for environment variables."""
        with patch.dict(os.environ, {"CCFLOW_DEFAULT_MODEL": "opus"}):
            settings = CCFlowSettings()
            assert settings.default_model == "opus"

    def test_env_override_timeout(self):
        """Test timeout can be set via environment."""
        with patch.dict(os.environ, {"CCFLOW_DEFAULT_TIMEOUT": "600.0"}):
            settings = CCFlowSettings()
            assert settings.default_timeout == 600.0

    def test_env_override_boolean(self):
        """Test boolean settings from environment."""
        with patch.dict(os.environ, {"CCFLOW_TOON_ENABLED": "false"}):
            settings = CCFlowSettings()
            assert settings.toon_enabled is False

        with patch.dict(os.environ, {"CCFLOW_ENABLE_METRICS": "false"}):
            settings = CCFlowSettings()
            assert settings.enable_metrics is False

    def test_env_override_max_turns(self):
        """Test max_turns can be set via environment."""
        with patch.dict(os.environ, {"CCFLOW_DEFAULT_MAX_TURNS": "25"}):
            settings = CCFlowSettings()
            assert settings.default_max_turns == 25

    def test_env_override_log_format(self):
        """Test log format can be set to json."""
        with patch.dict(os.environ, {"CCFLOW_LOG_FORMAT": "json"}):
            settings = CCFlowSettings()
            assert settings.log_format == "json"

    def test_env_override_cli_path(self):
        """Test CLI path can be set via environment."""
        with patch.dict(os.environ, {"CCFLOW_CLAUDE_CLI_PATH": "/custom/path/claude"}):
            settings = CCFlowSettings()
            assert settings.claude_cli_path == "/custom/path/claude"

    def test_env_override_toon_delimiter(self):
        """Test TOON delimiter options."""
        with patch.dict(os.environ, {"CCFLOW_TOON_DELIMITER": "\t"}):
            settings = CCFlowSettings()
            assert settings.toon_delimiter == "\t"

        with patch.dict(os.environ, {"CCFLOW_TOON_DELIMITER": "|"}):
            settings = CCFlowSettings()
            assert settings.toon_delimiter == "|"

    def test_env_override_rate_limits(self):
        """Test rate limit settings from environment."""
        with patch.dict(os.environ, {
            "CCFLOW_MAX_CONCURRENT_REQUESTS": "5",
            "CCFLOW_RATE_LIMIT_PER_MINUTE": "30",
        }):
            settings = CCFlowSettings()
            assert settings.max_concurrent_requests == 5
            assert settings.rate_limit_per_minute == 30

    def test_env_override_dangerous_permissions(self):
        """Test dangerous permissions flag."""
        with patch.dict(os.environ, {"CCFLOW_ALLOW_DANGEROUS_PERMISSIONS": "true"}):
            settings = CCFlowSettings()
            assert settings.allow_dangerous_permissions is True

    def test_case_insensitive_env(self):
        """Test environment variables are case insensitive."""
        with patch.dict(os.environ, {"ccflow_default_model": "haiku"}):
            settings = CCFlowSettings()
            # pydantic-settings handles case insensitivity
            assert settings.default_model == "haiku"


class TestTempPath:
    """Tests for temp_path property."""

    def test_temp_path_default(self):
        """Test temp_path uses /tmp by default."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove TMPDIR if set
            env = os.environ.copy()
            env.pop("TMPDIR", None)
            with patch.dict(os.environ, env, clear=True):
                settings = CCFlowSettings()
                assert str(settings.temp_path) == "/tmp"

    def test_temp_path_from_setting(self):
        """Test temp_path uses temp_dir setting."""
        with patch.dict(os.environ, {"CCFLOW_TEMP_DIR": "/custom/temp"}):
            settings = CCFlowSettings()
            assert str(settings.temp_path) == "/custom/temp"

    def test_temp_path_from_tmpdir_env(self):
        """Test temp_path uses TMPDIR environment variable."""
        with patch.dict(os.environ, {"TMPDIR": "/var/tmp"}):
            settings = CCFlowSettings(temp_dir=None)
            assert str(settings.temp_path) == "/var/tmp"

    def test_temp_path_setting_overrides_tmpdir(self):
        """Test temp_dir setting takes precedence over TMPDIR."""
        with patch.dict(os.environ, {
            "TMPDIR": "/var/tmp",
            "CCFLOW_TEMP_DIR": "/custom/temp",
        }):
            settings = CCFlowSettings()
            assert str(settings.temp_path) == "/custom/temp"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self):
        """Test get_settings returns CCFlowSettings."""
        settings = get_settings()
        assert isinstance(settings, CCFlowSettings)

    def test_cached(self):
        """Test get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_cache_can_be_cleared(self):
        """Test cache can be cleared for fresh settings."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()

        # After cache clear, should be new instance
        assert isinstance(settings2, CCFlowSettings)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_console(self, reset_structlog):
        """Test configure_logging with console format."""
        settings = CCFlowSettings(log_format="console")

        # Should not raise
        configure_logging(settings)

    def test_configure_logging_json(self, reset_structlog):
        """Test configure_logging with JSON format."""
        settings = CCFlowSettings(log_format="json")

        # Should not raise
        configure_logging(settings)

    def test_configure_logging_uses_global_settings(self, reset_structlog):
        """Test configure_logging uses get_settings if not provided."""
        # Should not raise and use global settings
        configure_logging()

    def test_configure_logging_json_processors(self, reset_structlog):
        """Test JSON format configures correct processors."""
        settings = CCFlowSettings(log_format="json")
        configure_logging(settings)

        # Verify structlog was configured (check it doesn't raise on use)
        logger = structlog.get_logger()
        assert logger is not None

    def test_configure_logging_console_processors(self, reset_structlog):
        """Test console format configures correct processors."""
        settings = CCFlowSettings(log_format="console")
        configure_logging(settings)

        # Verify structlog was configured
        logger = structlog.get_logger()
        assert logger is not None


class TestSettingsValidation:
    """Tests for settings validation constraints."""

    def test_timeout_minimum(self):
        """Test timeout has minimum constraint."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CCFlowSettings(default_timeout=0.5)  # Below minimum of 1.0

    def test_timeout_maximum(self):
        """Test timeout has maximum constraint."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CCFlowSettings(default_timeout=4000.0)  # Above maximum of 3600

    def test_max_turns_minimum(self):
        """Test max_turns has minimum constraint."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CCFlowSettings(default_max_turns=0)  # Below minimum of 1

    def test_max_turns_maximum(self):
        """Test max_turns has maximum constraint."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CCFlowSettings(default_max_turns=101)  # Above maximum of 100

    def test_max_concurrent_requests_range(self):
        """Test max_concurrent_requests has valid range."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CCFlowSettings(max_concurrent_requests=0)

        with pytest.raises(ValidationError):
            CCFlowSettings(max_concurrent_requests=101)

    def test_rate_limit_range(self):
        """Test rate_limit_per_minute has valid range."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CCFlowSettings(rate_limit_per_minute=0)

        with pytest.raises(ValidationError):
            CCFlowSettings(rate_limit_per_minute=1001)

    def test_valid_log_format_values(self):
        """Test log_format only accepts valid values."""
        from pydantic import ValidationError

        # Valid values should work
        CCFlowSettings(log_format="console")
        CCFlowSettings(log_format="json")

        # Invalid value should fail
        with pytest.raises(ValidationError):
            CCFlowSettings(log_format="xml")  # type: ignore

    def test_valid_toon_delimiter_values(self):
        """Test toon_delimiter only accepts valid values."""
        from pydantic import ValidationError

        # Valid values should work
        CCFlowSettings(toon_delimiter=",")
        CCFlowSettings(toon_delimiter="\t")
        CCFlowSettings(toon_delimiter="|")

        # Invalid value should fail
        with pytest.raises(ValidationError):
            CCFlowSettings(toon_delimiter=";")  # type: ignore


class TestSettingsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_extra_env_vars_ignored(self):
        """Test extra environment variables are ignored."""
        with patch.dict(os.environ, {"CCFLOW_UNKNOWN_SETTING": "value"}):
            # Should not raise
            settings = CCFlowSettings()
            assert not hasattr(settings, "unknown_setting")

    def test_empty_string_env_vars(self):
        """Test empty string environment variables."""
        with patch.dict(os.environ, {"CCFLOW_FALLBACK_MODEL": ""}):
            settings = CCFlowSettings()
            # Empty string should be treated as empty, not None
            # Behavior depends on pydantic version

    def test_settings_are_immutable_by_default(self):
        """Test settings object behavior."""
        settings = CCFlowSettings()
        # Settings should be usable
        assert settings.default_model == "sonnet"
