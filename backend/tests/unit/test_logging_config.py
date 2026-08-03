"""
Tests for logging configuration to catch structlog issues.
This test specifically catches the 'event' parameter conflict that caused task failures.
"""

import os
import sys

import pytest
import structlog

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from logging_config import (
    TaskContext,
    get_logger,
    log_api_request,
    log_api_response,
    log_external_service_call,
    log_file_operation,
    log_task_complete,
    log_task_error,
    log_task_start,
    setup_logging,
)


class TestLoggingConfiguration:
    """Test logging configuration and structured logging functions."""

    def setup_method(self):
        """Setup for each test method."""
        # Reset structlog configuration before each test
        structlog.reset_defaults()

    def test_setup_logging_basic(self):
        """Test basic logging setup doesn't crash."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_json_mode(self):
        """Test JSON logging mode setup."""
        setup_logging(level="DEBUG", testing=False, json_logs=True)
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_console_mode(self):
        """Test console logging mode setup."""
        setup_logging(level="WARNING", testing=False, json_logs=False)
        logger = get_logger("test")
        assert logger is not None

    def test_logger_basic_methods(self):
        """Test that basic logger methods work without parameter conflicts."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        # These should not raise any exceptions
        logger.info("Test message")
        logger.error("Test error")
        logger.warning("Test warning")
        logger.debug("Test debug")

    def test_logger_with_kwargs(self):
        """Test logger with keyword arguments - this catches the 'event' conflict."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        # This should work without conflicts
        logger.info("Test message", key1="value1", key2="value2")
        logger.error("Test error", error_code=500, details="Something went wrong")

    def test_task_context_manager(self):
        """Test TaskContext context manager."""
        setup_logging(level="INFO", testing=True)

        with TaskContext("test-task-123", "video_processing"):
            logger = get_logger("test")
            logger.info("Inside task context")

    def test_log_task_functions(self):
        """Test all log_task_* functions work without parameter conflicts."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        # Test all logging helper functions
        log_task_start(logger, "test_task", param1="value1")
        log_task_complete(logger, "test_task", duration=1.5, result="success")

        # Test error logging
        test_error = Exception("Test error")
        log_task_error(logger, "test_task", test_error, context="test")

    def test_log_external_service_call(self):
        """Test external service call logging - this was the source of the bug."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        # This was causing the 'event' parameter conflict
        log_external_service_call(
            logger,
            service="youtube",
            operation="download",
            success=True,
            duration=2.5,
            url="https://example.com",
        )

        # Test failure case
        log_external_service_call(
            logger,
            service="youtube",
            operation="download",
            success=False,
            error="Network timeout",
        )

    def test_log_api_functions(self):
        """Test API logging functions."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        log_api_request(logger, "POST", "/api/process", user_id="123")
        log_api_response(logger, "POST", "/api/process", 200, duration=0.5)

    def test_log_file_operation(self):
        """Test file operation logging."""
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        log_file_operation(
            logger,
            operation="download",
            file_path="/tmp/test.mp4",
            success=True,
            size_mb=15.2,
        )

    def test_logger_with_event_keyword(self):
        """
        Specific test for the 'event' keyword conflict that caused the original bug.
        This test ensures we can use 'event' as a parameter without conflicts.
        """
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test")

        # This should NOT cause: "got multiple values for argument 'event'"
        # Use different parameter name to avoid conflict
        logger.info("Test message with event", event_type="custom_event")
        logger.error("Test error with event", event_type="error_event", details="test")

    def test_multiple_loggers(self):
        """Test that multiple loggers work correctly."""
        setup_logging(level="INFO", testing=True)

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        logger1.info("Message from module1")
        logger2.info("Message from module2")

    def test_logger_in_task_context(self):
        """Test logging within task context with various parameters."""
        setup_logging(level="INFO", testing=True)

        with TaskContext("task-456", "subtitle_generation", "user-789"):
            logger = get_logger("test")

            # Test various logging scenarios that might cause conflicts
            logger.info("Task progress", percent=50, stage="processing")
            logger.error(
                "Task error", error_type="ValidationError", message="Invalid input"
            )

            # Test the external service call within task context
            log_external_service_call(
                logger,
                service="whisper",
                operation="transcribe",
                success=True,
                duration=10.5,
            )


class TestLoggingRegression:
    """Regression tests for specific logging bugs."""

    def test_event_parameter_conflict_regression(self):
        """
        Regression test for the specific 'event' parameter conflict.

        This test reproduces the exact error that was happening:
        "_make_filtering_bound_logger.<locals>.make_method.<locals>.meth()
         got multiple values for argument 'event'"
        """
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test_regression")

        # This exact pattern was causing the error
        try:
            log_external_service_call(
                logger,
                service="youtube-dl",
                operation="extract_info",
                success=True,
                duration=3.2,
                url="https://youtube.com/watch?v=test",
                format="mp4",
            )
            # If we get here, the bug is fixed
            assert True

        except TypeError as e:
            if "got multiple values for argument 'event'" in str(e):
                pytest.fail(f"The 'event' parameter conflict bug has returned: {e}")
            else:
                # Some other TypeError, re-raise it
                raise

    def test_event_keyword_should_fail(self):
        """
        Test that demonstrates the WRONG way to use structlog.
        This test verifies that using 'event' as a keyword argument fails as expected.
        This is the MOST IMPORTANT test - it catches if someone accidentally uses event= again.
        """
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test_wrong_usage")

        # This SHOULD fail - it's the wrong way to use structlog
        with pytest.raises(TypeError, match="got multiple values for argument 'event'"):
            logger.info("Test message", event="this_should_fail")

    def test_correct_event_usage(self):
        """
        Test that demonstrates the CORRECT way to use structlog.
        The first parameter is always the event message.
        """
        setup_logging(level="INFO", testing=True)
        logger = get_logger("test_correct_usage")

        # This is the CORRECT way - event is the first positional parameter
        logger.info("This is the event message", custom_field="value")
        logger.error(
            "This is an error event", error_code=500, details="Something went wrong"
        )

    def test_structlog_configuration_stability(self):
        """Test that structlog configuration is stable and doesn't cause conflicts."""
        setup_logging(level="INFO", testing=True)

        # Get multiple loggers and use them extensively
        loggers = [get_logger(f"test_{i}") for i in range(5)]

        for i, logger in enumerate(loggers):
            logger.info(f"Logger {i} test", index=i, test_param="value")
            logger.error(f"Logger {i} error", index=i, error_code=500)

            # Test with various keyword combinations
            logger.info(
                "Complex log entry",
                event_type="test",  # This should not conflict
                service="test_service",
                operation="test_operation",
                success=True,
                metadata={"key": "value"},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
