"""Tests for ccflow metrics module."""

from unittest.mock import MagicMock, patch

import pytest


class TestMetricsDisabled:
    """Tests when metrics are disabled or prometheus unavailable."""

    def test_record_request_noop_when_disabled(self):
        """Test record_request does nothing when disabled."""
        from ccflow.metrics import record_request

        # Should not raise even with invalid data
        record_request(
            model="test",
            status="success",
            duration=1.0,
            input_tokens=100,
            output_tokens=50,
        )

    def test_record_toon_savings_noop_when_disabled(self):
        """Test record_toon_savings does nothing when disabled."""
        from ccflow.metrics import record_toon_savings

        record_toon_savings(json_tokens=100, toon_tokens=60)

    def test_record_error_noop_when_disabled(self):
        """Test record_error does nothing when disabled."""
        from ccflow.metrics import record_error

        record_error("TestError")

    def test_track_session_noop_when_disabled(self):
        """Test track_session does nothing when disabled."""
        from ccflow.metrics import track_session

        track_session(increment=True)
        track_session(increment=False)

    def test_record_session_complete_noop_when_disabled(self):
        """Test record_session_complete does nothing when disabled."""
        from ccflow.metrics import record_session_complete

        record_session_complete(turns=5)


class TestMetricsEnabled:
    """Tests when metrics are enabled with mocked prometheus."""

    @pytest.fixture
    def mock_prometheus(self):
        """Mock prometheus client and enable metrics."""
        # Create mock metric objects
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_gauge = MagicMock()
        mock_info = MagicMock()

        # Labels returns the same mock for chaining
        mock_counter.labels.return_value = mock_counter
        mock_histogram.labels.return_value = mock_histogram

        return {
            "counter": mock_counter,
            "histogram": mock_histogram,
            "gauge": mock_gauge,
            "info": mock_info,
        }

    @pytest.fixture
    def enable_metrics(self, mock_prometheus):
        """Enable metrics with mocked settings and prometheus."""
        mock_settings = MagicMock()
        mock_settings.enable_metrics = True

        with patch("ccflow.metrics.get_settings", return_value=mock_settings), \
             patch("ccflow.metrics.PROMETHEUS_AVAILABLE", True), \
             patch("ccflow.metrics.REQUESTS_TOTAL", mock_prometheus["counter"]), \
             patch("ccflow.metrics.REQUEST_DURATION", mock_prometheus["histogram"]), \
             patch("ccflow.metrics.TOKENS_INPUT", mock_prometheus["counter"]), \
             patch("ccflow.metrics.TOKENS_OUTPUT", mock_prometheus["counter"]), \
             patch("ccflow.metrics.TOON_SAVINGS_RATIO", mock_prometheus["histogram"]), \
             patch("ccflow.metrics.TOON_TOKENS_SAVED", mock_prometheus["counter"]), \
             patch("ccflow.metrics.ACTIVE_SESSIONS", mock_prometheus["gauge"]), \
             patch("ccflow.metrics.SESSION_TURNS", mock_prometheus["histogram"]), \
             patch("ccflow.metrics.ERRORS_TOTAL", mock_prometheus["counter"]), \
             patch("ccflow.metrics.BUILD_INFO", mock_prometheus["info"]):
            yield mock_prometheus

    def test_record_request_increments_counters(self, enable_metrics):
        """Test record_request increments prometheus counters."""
        from ccflow.metrics import record_request

        record_request(
            model="sonnet",
            status="success",
            duration=1.5,
            input_tokens=100,
            output_tokens=50,
        )

        # Verify labels were called correctly
        enable_metrics["counter"].labels.assert_any_call(model="sonnet", status="success")
        enable_metrics["counter"].inc.assert_called()
        enable_metrics["histogram"].labels.assert_any_call(model="sonnet")
        enable_metrics["histogram"].observe.assert_called_with(1.5)

    def test_record_request_skips_zero_tokens(self, enable_metrics):
        """Test record_request doesn't increment token counters for zero."""
        from ccflow.metrics import record_request

        # Reset mocks
        enable_metrics["counter"].reset_mock()

        record_request(
            model="sonnet",
            status="success",
            duration=1.0,
            input_tokens=0,
            output_tokens=0,
        )

        # Token counters should not have inc called with token amounts
        calls = [str(c) for c in enable_metrics["counter"].inc.call_args_list]
        # Only the request counter increment, not token counters
        assert len(calls) == 1

    def test_record_toon_savings_calculates_ratio(self, enable_metrics):
        """Test record_toon_savings calculates correct ratio."""
        from ccflow.metrics import record_toon_savings

        record_toon_savings(json_tokens=100, toon_tokens=60)

        # Ratio should be 1 - (60/100) = 0.4
        enable_metrics["histogram"].observe.assert_called_with(0.4)
        # Saved should be 100 - 60 = 40
        enable_metrics["counter"].inc.assert_called_with(40)

    def test_record_toon_savings_handles_zero_json_tokens(self, enable_metrics):
        """Test record_toon_savings handles zero json tokens."""
        from ccflow.metrics import record_toon_savings

        enable_metrics["histogram"].reset_mock()
        enable_metrics["counter"].reset_mock()

        record_toon_savings(json_tokens=0, toon_tokens=0)

        # Should not record anything
        enable_metrics["histogram"].observe.assert_not_called()

    def test_record_error_increments_counter(self, enable_metrics):
        """Test record_error increments error counter."""
        from ccflow.metrics import record_error

        record_error("CLIExecutionError")

        enable_metrics["counter"].labels.assert_called_with(error_type="CLIExecutionError")
        enable_metrics["counter"].inc.assert_called()

    def test_track_session_increment(self, enable_metrics):
        """Test track_session increments gauge."""
        from ccflow.metrics import track_session

        track_session(increment=True)

        enable_metrics["gauge"].inc.assert_called_once()

    def test_track_session_decrement(self, enable_metrics):
        """Test track_session decrements gauge."""
        from ccflow.metrics import track_session

        track_session(increment=False)

        enable_metrics["gauge"].dec.assert_called_once()

    def test_record_session_complete(self, enable_metrics):
        """Test record_session_complete records turns and decrements."""
        from ccflow.metrics import record_session_complete

        record_session_complete(turns=5)

        enable_metrics["histogram"].observe.assert_called_with(5)
        enable_metrics["gauge"].dec.assert_called_once()


class TestTimedOperation:
    """Tests for timed_operation context manager."""

    def test_timed_operation_success(self):
        """Test timed_operation with successful operation."""
        from ccflow.metrics import timed_operation

        with patch("ccflow.metrics.record_request") as mock_record:
            with timed_operation("sonnet", "test_op") as result:
                result["input_tokens"] = 100
                result["output_tokens"] = 50

            # Should have recorded success
            mock_record.assert_called_once()
            call_kwargs = mock_record.call_args.kwargs
            assert call_kwargs["model"] == "sonnet"
            assert call_kwargs["status"] == "success"
            assert call_kwargs["input_tokens"] == 100
            assert call_kwargs["output_tokens"] == 50
            assert call_kwargs["duration"] > 0

    def test_timed_operation_error(self):
        """Test timed_operation records error on exception."""
        from ccflow.metrics import timed_operation

        with patch("ccflow.metrics.record_request") as mock_record, \
             patch("ccflow.metrics.record_error") as mock_error:
            with pytest.raises(ValueError), timed_operation("sonnet", "test_op"):
                raise ValueError("Test error")

            # Should have recorded error
            mock_error.assert_called_once_with("ValueError")
            mock_record.assert_called_once()
            assert mock_record.call_args.kwargs["status"] == "error"

    def test_timed_operation_measures_duration(self):
        """Test timed_operation accurately measures duration."""
        import time

        from ccflow.metrics import timed_operation

        with patch("ccflow.metrics.record_request") as mock_record:
            with timed_operation("sonnet"):
                time.sleep(0.01)  # 10ms

            duration = mock_record.call_args.kwargs["duration"]
            assert duration >= 0.01


class TestSetBuildInfo:
    """Tests for set_build_info function."""

    def test_set_build_info_with_prometheus(self):
        """Test set_build_info sets info metric."""
        mock_info = MagicMock()

        with patch("ccflow.metrics.PROMETHEUS_AVAILABLE", True), \
             patch("ccflow.metrics.BUILD_INFO", mock_info):
            from ccflow.metrics import set_build_info

            set_build_info("1.0.0", commit="abc123", branch="main")

            mock_info.info.assert_called_once_with({
                "version": "1.0.0",
                "commit": "abc123",
                "branch": "main",
            })

    def test_set_build_info_without_prometheus(self):
        """Test set_build_info is noop without prometheus."""
        with patch("ccflow.metrics.PROMETHEUS_AVAILABLE", False), \
             patch("ccflow.metrics.BUILD_INFO", None):
            from ccflow.metrics import set_build_info

            # Should not raise
            set_build_info("1.0.0")


class TestPrometheusAvailability:
    """Tests for prometheus availability detection."""

    def test_prometheus_available_flag(self):
        """Test PROMETHEUS_AVAILABLE flag is set correctly."""
        from ccflow import metrics

        # Just verify the flag exists and is boolean
        assert isinstance(metrics.PROMETHEUS_AVAILABLE, bool)

    def test_metrics_noop_when_prometheus_unavailable(self):
        """Test all metrics are noop when prometheus unavailable."""
        with patch("ccflow.metrics.PROMETHEUS_AVAILABLE", False):
            from ccflow.metrics import (
                record_error,
                record_request,
                record_session_complete,
                record_toon_savings,
                track_session,
            )

            # None of these should raise
            record_request("test", "success", 1.0, 10, 5)
            record_toon_savings(100, 60)
            record_error("TestError")
            track_session(True)
            track_session(False)
            record_session_complete(5)
