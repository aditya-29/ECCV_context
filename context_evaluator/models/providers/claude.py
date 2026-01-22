"""
Anthropic Claude Inference Module
Performs inference on Claude models (Claude Opus 4.5, Claude Sonnet 4.5) with support for:
- Text-only inputs
- Image + text inputs (via base64 encoding)
- Video + text inputs (videos converted to frames)
- Exponential backoff retry logic
- Request queueing for rate limits
- Batch processing with parallel execution
- Result caching
- Token usage and cost tracking
"""

import yaml
import base64
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import anthropic
from anthropic import Anthropic, RateLimitError, APIError, APIStatusError
from tqdm import tqdm

from context_evaluator.models.model_config import (
    supports_vision, calculate_cost
)
from context_evaluator.logger import setup_logging
from context_evaluator.models.video_utils import VideoProcessor, SamplingStrategy
from context_evaluator.models.cache_manager import CacheManager
from context_evaluator.models import (
    BaseModel,
    RateLimitException,
    PayloadTooLargeException,
    RetryableErrorType
)


class ClaudeInference(BaseModel):
    """
    Main class for performing inference on Anthropic Claude models.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Claude inference client.

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
            model_name=self.config['defaults'].get('model', 'claude-sonnet-4-5-20250929'),
            max_retries=retry_config.get('max_retries', 5),
            initial_backoff=retry_config.get('initial_backoff', 1.0),
            max_backoff=retry_config.get('max_backoff', 60.0),
            backoff_multiplier=retry_config.get('backoff_multiplier', 2.0)
        )

        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing Claude Inference Client")

        # Initialize Anthropic client
        api_key = self.config['api']['anthropic_api_key']
        if not api_key or api_key == "your-anthropic-api-key-here":
            raise ValueError("Please set your Anthropic API key in config.yaml")

        self.client = Anthropic(api_key=api_key)

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

        self.logger.info("Claude Inference Client initialized successfully")

    def _classify_error(self, error: Exception) -> RetryableErrorType:
        """
        Classify Claude-specific errors.

        Args:
            error: The exception that was raised

        Returns:
            RetryableErrorType indicating the type of error
        """
        # Check for rate limit errors
        if isinstance(error, RateLimitError):
            return RetryableErrorType.RATE_LIMIT

        # Check for API status errors
        if isinstance(error, APIStatusError):
            status_code = getattr(error, 'status_code', None)
            error_str = str(error).lower()

            # Rate limiting (429)
            if status_code == 429:
                return RetryableErrorType.RATE_LIMIT

            # Payload too large (413)
            if status_code == 413 or "request too large" in error_str:
                return RetryableErrorType.PAYLOAD_TOO_LARGE

            # Service unavailable, timeout (503, 504)
            if status_code in (503, 504) or "timeout" in error_str:
                return RetryableErrorType.TIMEOUT

            # Other API errors (500, 502, etc.)
            if status_code and 500 <= status_code < 600:
                return RetryableErrorType.API_ERROR

        # Check for general API errors
        if isinstance(error, APIError):
            error_str = str(error).lower()

            if "overloaded" in error_str or "capacity" in error_str:
                return RetryableErrorType.RATE_LIMIT

            if "too large" in error_str:
                return RetryableErrorType.PAYLOAD_TOO_LARGE

            return RetryableErrorType.API_ERROR

        # Check for string-based error detection
        error_str = str(error).lower()
        if "rate limit" in error_str or "quota" in error_str:
            return RetryableErrorType.RATE_LIMIT

        return RetryableErrorType.UNKNOWN

    def _should_retry(self, error_type: RetryableErrorType, attempt: int) -> bool:
        """
        Determine if a Claude error should be retried.

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
        video_processor: Optional[VideoProcessor] = None,
        image_paths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare content for Claude API call.

        Args:
            text_prompt: Text prompt
            video_path: Optional path to video file
            video_processor: Optional VideoProcessor instance
            image_paths: Optional list of image paths

        Returns:
            List of content blocks for Claude API
        """
        content_blocks = []

        # Add video frames if provided (Claude doesn't have native video support)
        if video_path:
            self.logger.info(f"Processing video: {video_path}")

            # Use provided processor or create default one
            if video_processor is None:
                video_config = self.config.get('video_processing', {})
                video_processor = VideoProcessor(
                    sampling_strategy=SamplingStrategy.UNIFORM,
                    target_fps=video_config.get('default_fps', 1.0),
                    max_frames=video_config.get('max_frames', 100)
                )

            # Extract frames as base64
            frames = video_processor.process_video(video_path, output_format="base64")
            self.logger.info(f"Extracted {len(frames)} frames from video")

            # Add frames to content
            for i, frame_base64 in enumerate(frames):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame_base64
                    }
                })

        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                self.logger.info(f"Adding image: {img_path}")
                with open(img_path, 'rb') as img_file:
                    img_data = base64.standard_b64encode(img_file.read()).decode('utf-8')

                # Detect image type
                media_type = "image/jpeg"
                if img_path.lower().endswith('.png'):
                    media_type = "image/png"
                elif img_path.lower().endswith('.gif'):
                    media_type = "image/gif"
                elif img_path.lower().endswith('.webp'):
                    media_type = "image/webp"

                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data
                    }
                })

        # Add text prompt
        content_blocks.append({
            "type": "text",
            "text": text_prompt
        })

        return content_blocks

    def infer(
        self,
        text_prompt: str,
        video_path: Optional[str] = None,
        video_processor: Optional[VideoProcessor] = None,
        image_paths: Optional[List[str]] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
        **model_params
    ) -> str:
        """
        Perform inference on Claude model.

        Args:
            text_prompt: Text prompt
            video_path: Optional path to video file (converted to frames)
            video_processor: Custom video processor
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
            model = self.config['defaults'].get('model', 'claude-sonnet-4-5-20250929')

        # Validate model supports vision if media provided
        if (video_path or image_paths) and not supports_vision(model):
            raise ValueError(f"Model '{model}' does not support vision/image inputs")

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
        api_params = {
            'model': model,
            'max_tokens': model_params.get('max_tokens',
                                          self.config['defaults'].get('max_tokens', 8192)),
            'temperature': model_params.get('temperature',
                                          self.config['defaults'].get('temperature', 0.7)),
        }

        # Add system message if provided
        if 'system' in model_params:
            api_params['system'] = model_params['system']

        # Prepare content
        content = self._prepare_content(
            text_prompt,
            video_path,
            video_processor,
            image_paths
        )

        api_params['messages'] = [
            {
                "role": "user",
                "content": content
            }
        ]

        # Make API call with retry logic
        self.logger.info(f"Making inference request with model: {model}")

        try:
            response = self._exponential_backoff(
                self.client.messages.create,
                **api_params
            )
        except PayloadTooLargeException:
            self.logger.error("Payload too large - cannot proceed")
            raise
        except RateLimitException:
            self.logger.warning("Rate limit exceeded - adding to queue")
            raise

        # Extract output text
        output_text = response.content[0].text

        # Track tokens and costs
        if hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            # Claude has cache_read_input_tokens and cache_creation_input_tokens
            cached_tokens = getattr(usage, 'cache_read_input_tokens', 0)

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
