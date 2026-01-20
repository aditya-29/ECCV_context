"""
Base Model Module
Defines abstract base class for all inference models with common functionality:
- Exponential backoff retry mechanism
- Provider-specific error handling via abstract methods
- Common configuration management
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Callable
from enum import Enum


class RetryableErrorType(Enum):
    """Enum for different types of retryable errors."""
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RateLimitException(Exception):
    """Exception for rate limit errors that should be queued."""
    pass


class PayloadTooLargeException(Exception):
    """Exception for payload size errors."""
    pass


class BaseModel(ABC):
    """
    Abstract base class for all model inference implementations.

    Provides:
    - Generic exponential backoff retry logic
    - Provider-specific error handling via abstract methods
    - Configuration management
    - Logging setup
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0
    ):
        """
        Initialize base model.

        Args:
            model_name: Name of the model
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.logger = None  # Should be set by subclass

    @abstractmethod
    def _classify_error(self, error: Exception) -> RetryableErrorType:
        """
        Classify an error to determine retry behavior.

        This method should be implemented by each provider-specific subclass
        to handle their specific error types (e.g., OpenAI errors, Anthropic errors).

        Args:
            error: The exception that was raised

        Returns:
            RetryableErrorType indicating the type of error
        """
        pass

    @abstractmethod
    def _should_retry(self, error_type: RetryableErrorType, attempt: int) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error_type: Type of error that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False otherwise
        """
        pass

    def _handle_error(
        self,
        error: Exception,
        error_type: RetryableErrorType,
        attempt: int
    ) -> None:
        """
        Handle error before retry or final failure.

        Can be overridden by subclasses for custom error handling.

        Args:
            error: The exception that was raised
            error_type: Classified error type
            attempt: Current attempt number (0-indexed)
        """
        # Default implementation - can be overridden
        if self.logger:
            if error_type == RetryableErrorType.RATE_LIMIT:
                self.logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self.max_retries})"
                )
            elif error_type == RetryableErrorType.API_ERROR:
                self.logger.warning(
                    f"API error (attempt {attempt + 1}/{self.max_retries}): {error}"
                )

    def _exponential_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry logic.

        This is a generic retry mechanism that delegates error classification
        and retry decisions to provider-specific implementations.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If all retries are exhausted or non-retryable error occurs
        """
        backoff = self.initial_backoff
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_error = e

                # Classify the error using provider-specific logic
                error_type = self._classify_error(e)

                # Check if we should retry
                if not self._should_retry(error_type, attempt):
                    # Non-retryable error or max retries reached
                    if error_type == RetryableErrorType.RATE_LIMIT:
                        raise RateLimitException(f"Rate limit exceeded: {e}")
                    elif error_type == RetryableErrorType.PAYLOAD_TOO_LARGE:
                        raise PayloadTooLargeException(f"Payload too large: {e}")
                    else:
                        raise

                # Handle error (logging, etc.)
                self._handle_error(e, error_type, attempt)

                # Sleep with exponential backoff
                if self.logger:
                    self.logger.info(f"Retrying in {backoff:.2f}s...")
                time.sleep(backoff)
                backoff = min(backoff * self.backoff_multiplier, self.max_backoff)

        # All retries exhausted
        raise Exception(
            f"Failed after {self.max_retries} retries. Last error: {last_error}"
        )

    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        """
        Perform inference. Must be implemented by subclasses.

        Returns:
            Model output
        """
        pass

    @abstractmethod
    def batch_infer(self, *args, **kwargs) -> Any:
        """
        Perform batch inference. Must be implemented by subclasses.

        Returns:
            Model output
        """
        pass