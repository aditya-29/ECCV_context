"""
Google Gemini Inference Module
Performs inference on Gemini models (Gemini 2.0 Flash, Gemini 2.0 Pro, Gemini 2.0 Thinking) with support for:
- Text-only inputs
- Image + text inputs
- Video + text inputs (native video support)
- Exponential backoff retry logic
- Request queueing for rate limits
- Batch processing with parallel execution
- Result caching
- Token usage and cost tracking
"""

import yaml
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions
from tqdm import tqdm

from context_evaluator.models.model_config import (
    supports_vision, calculate_cost
)
from context_evaluator.logger import setup_logging
from context_evaluator.models.cache_manager import CacheManager
from context_evaluator.models import (
    BaseModel,
    RateLimitException,
    PayloadTooLargeException,
    RetryableErrorType
)


class GeminiInference(BaseModel):
    """
    Main class for performing inference on Google Gemini models.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Gemini inference client.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Retry configuration
        retry_config = self.config.get('retry', {})

        # Initialize base class with retry configuration
        super().__init__(
            model_name=self.config['defaults'].get('model', 'gemini-2.0-flash-exp'),
            max_retries=retry_config.get('max_retries', 5),
            initial_backoff=retry_config.get('initial_backoff', 1.0),
            max_backoff=retry_config.get('max_backoff', 60.0),
            backoff_multiplier=retry_config.get('backoff_multiplier', 2.0)
        )

        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing Gemini Inference Client")

        # Initialize Gemini client
        api_key = self.config['api']['gemini_api_key']
        if not api_key or api_key == "your-gemini-api-key-here":
            raise ValueError("Please set your Gemini API key in config.yaml")

        genai.configure(api_key=api_key)

        # Initialize cache
        cache_config = self.config.get('cache', {})
        self.cache_enabled = cache_config.get('enabled', True)
        if self.cache_enabled:
            self.cache = CacheManager(
                cache_file=cache_config.get('cache_file', 'inference_cache.json')
            )
        else:
            self.cache = None

        # Queue for rate-limited requests
        self.request_queue = Queue(
            maxsize=self.config.get('queue', {}).get('max_queue_size', 100)
        )

        # Token tracking
        self.track_token_usage = self.config.get('logging', {}).get('track_token_usage', True)
        self.track_costs = self.config.get('logging', {}).get('track_costs', True)
        self.total_tokens = {'input': 0, 'output': 0, 'cached': 0}
        self.total_cost = 0.0

        # Safety settings (permissive for research)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.logger.info("Gemini Inference Client initialized successfully")

    def _classify_error(self, error: Exception) -> RetryableErrorType:
        """
        Classify Gemini-specific errors.

        Args:
            error: The exception that was raised

        Returns:
            RetryableErrorType indicating the type of error
        """
        # Check for rate limit errors
        if isinstance(error, google_exceptions.ResourceExhausted):
            return RetryableErrorType.RATE_LIMIT

        # Check for quota/resource errors
        if isinstance(error, (google_exceptions.TooManyRequests,
                             google_exceptions.ServiceUnavailable)):
            return RetryableErrorType.RATE_LIMIT

        # Check for API errors
        if isinstance(error, google_exceptions.GoogleAPIError):
            error_str = str(error).lower()

            # Payload too large
            if "payload too large" in error_str or "request too large" in error_str:
                return RetryableErrorType.PAYLOAD_TOO_LARGE

            # Timeout errors
            if "timeout" in error_str or "deadline exceeded" in error_str:
                return RetryableErrorType.TIMEOUT

            return RetryableErrorType.API_ERROR

        # Check for generic Google API errors
        if "quota" in str(error).lower() or "rate" in str(error).lower():
            return RetryableErrorType.RATE_LIMIT

        return RetryableErrorType.UNKNOWN

    def _should_retry(self, error_type: RetryableErrorType, attempt: int) -> bool:
        """
        Determine if a Gemini error should be retried.

        Args:
            error_type: Type of error that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False otherwise
        """
        # Never retry payload too large errors
        if error_type == RetryableErrorType.PAYLOAD_TOO_LARGE:
            return False

        # Don't retry on last attempt
        if attempt >= self.max_retries - 1:
            return False

        # Retry rate limits, API errors, and timeouts
        if error_type in (RetryableErrorType.RATE_LIMIT,
                         RetryableErrorType.API_ERROR,
                         RetryableErrorType.TIMEOUT):
            return True

        # Don't retry unknown errors
        return False

    def _prepare_content(
        self,
        text_prompt: str,
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Prepare content for Gemini API call.

        Args:
            text_prompt: Text prompt
            video_path: Optional path to video file
            image_paths: Optional list of image paths

        Returns:
            List of content parts for Gemini API
        """
        content_parts = []

        # Add video if provided (Gemini has native video support)
        if video_path:
            self.logger.info(f"Uploading video: {video_path}")

            # Upload video file to Gemini
            video_file = genai.upload_file(path=video_path)
            content_parts.append(video_file)

            self.logger.info(f"Video uploaded successfully: {video_file.name}")

        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                self.logger.info(f"Uploading image: {img_path}")
                img_file = genai.upload_file(path=img_path)
                content_parts.append(img_file)

        # Add text prompt
        content_parts.append(text_prompt)

        return content_parts

    def infer(
        self,
        text_prompt: str,
        video_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
        **model_params
    ) -> str:
        """
        Perform inference on Gemini model.

        Args:
            text_prompt: Text prompt
            video_path: Optional path to video file (native video support)
            image_paths: Optional list of image file paths
            model: Model name (default from config)
            use_cache: Whether to use cache
            **model_params: Additional model parameters

        Returns:
            Model output text

        Raises:
            ValueError: If model doesn't support vision but media is provided
            PayloadTooLargeException: If payload is too large
            RateLimitException: If rate limit is exceeded
        """
        # Use default model if not specified
        if model is None:
            model = self.config['defaults'].get('model', 'gemini-2.0-flash-exp')

        # Validate model supports vision if media provided
        if (video_path or image_paths) and not supports_vision(model):
            raise ValueError(f"Model '{model}' does not support vision/video inputs")

        # Check cache first
        cache_key_params = {**model_params}
        if image_paths:
            cache_key_params['image_paths'] = image_paths

        if use_cache and self.cache:
            cached_result = self.cache.get(video_path, text_prompt, model, cache_key_params)
            if cached_result:
                self.logger.info("Using cached result")
                return cached_result['output_text']

        # Set default parameters
        generation_config = {
            'temperature': model_params.get('temperature',
                                          self.config['defaults'].get('temperature', 0.7)),
            'max_output_tokens': model_params.get('max_tokens',
                                                  self.config['defaults'].get('max_tokens', 8192)),
        }

        # Add model-specific parameters for Gemini 2.0 Thinking
        if 'thinking' in model.lower():
            # Thinking models have different configuration
            if 'thinking_mode' in model_params:
                generation_config['thinking_mode'] = model_params['thinking_mode']

        # Prepare content
        content = self._prepare_content(text_prompt, video_path, image_paths)

        # Create model instance
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )

        # Make API call with retry logic
        self.logger.info(f"Making inference request with model: {model}")

        try:
            response = self._exponential_backoff(
                gemini_model.generate_content,
                content
            )
        except PayloadTooLargeException:
            self.logger.error("Payload too large - cannot proceed")
            raise
        except RateLimitException:
            self.logger.warning("Rate limit exceeded - adding to queue")
            raise

        # Extract output text
        output_text = response.text

        # Track tokens and costs
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            input_tokens = getattr(usage, 'prompt_token_count', 0)
            output_tokens = getattr(usage, 'candidates_token_count', 0)
            cached_tokens = getattr(usage, 'cached_content_token_count', 0)

            if self.track_token_usage:
                self.total_tokens['input'] += input_tokens
                self.total_tokens['output'] += output_tokens
                self.total_tokens['cached'] += cached_tokens

                self.logger.info(
                    f"Token usage - Input: {input_tokens}, Output: {output_tokens}, "
                    f"Cached: {cached_tokens}"
                )

            if self.track_costs:
                cost = calculate_cost(model, input_tokens, output_tokens, cached_tokens)
                self.total_cost += cost
                self.logger.info(f"Inference cost: ${cost:.6f}")
        else:
            input_tokens = output_tokens = cached_tokens = 0

        # Cache result
        if use_cache and self.cache:
            result = {
                'output_text': output_text,
                'usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cached_tokens': cached_tokens,
                },
                'model': model
            }
            self.cache.set(video_path, text_prompt, model, cache_key_params, result)

        return output_text

    def batch_infer(
        self,
        requests: List[Dict[str, Any]],
        num_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform batch inference with parallel processing.

        Args:
            requests: List of inference requests, each containing:
                - text_prompt: Text prompt
                - video_path: Optional video path
                - image_paths: Optional list of image paths
                - model: Optional model name
                - Other model parameters
            num_workers: Number of parallel workers (default from config)
            show_progress: Show progress bar

        Returns:
            List of results with outputs and metadata
        """
        if num_workers is None:
            num_workers = self.config.get('queue', {}).get('worker_threads', 4)

        self.logger.info(
            f"Starting batch inference: {len(requests)} requests, "
            f"{num_workers} workers"
        )

        results = [None] * len(requests)

        def process_request(idx: int, request: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single request."""
            try:
                output = self.infer(**request)
                return {
                    'index': idx,
                    'success': True,
                    'output': output,
                    'request': request
                }
            except Exception as e:
                self.logger.error(f"Error processing request {idx}: {e}")
                return {
                    'index': idx,
                    'success': False,
                    'error': str(e),
                    'request': request
                }

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_request, idx, req): idx
                for idx, req in enumerate(requests)
            }

            # Progress bar
            if show_progress:
                pbar = tqdm(total=len(requests), desc="Batch Inference")

            for future in as_completed(futures):
                result = future.result()
                results[result['index']] = result

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        # Calculate statistics
        successful = sum(1 for r in results if r and r['success'])
        failed = len(results) - successful

        self.logger.info(
            f"Batch inference completed: {successful} successful, {failed} failed"
        )

        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with token usage and cost stats
        """
        return {
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'cache_stats': self.cache.get_stats() if self.cache else None
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.total_tokens = {'input': 0, 'output': 0, 'cached': 0}
        self.total_cost = 0.0
        self.logger.info("Usage statistics reset")
