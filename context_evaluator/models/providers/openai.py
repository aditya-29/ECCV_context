"""
OpenAI Inference Module
Performs inference on OpenAI models (GPT-5, GPT-5.2) with support for:
- Text-only inputs
- Video + text inputs (videos are converted to frames)
- Exponential backoff retry logic
- Request queueing for rate limits
- Batch processing with parallel execution
- Result caching
- Token usage and cost tracking
"""

import os
import json
import tempfile
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import time

from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

from context_evaluator.models.model_config import (
    get_model_config, supports_vision, get_api_type, calculate_cost
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

class OpenAIInference(BaseModel):
    """
    Main class for performing inference on OpenAI models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the inference client.

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
            model_name=self.config['defaults'].get('model', 'gpt-5.2'),
            max_retries=retry_config.get('max_retries', 5),
            initial_backoff=retry_config.get('initial_backoff', 1.0),
            max_backoff=retry_config.get('max_backoff', 60.0),
            backoff_multiplier=retry_config.get('backoff_multiplier', 2.0)
        )

        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing OpenAI Inference Client")

        # Initialize OpenAI client
        api_key = self.config['api']['openai_api_key']
        if not api_key or api_key == "your-openai-api-key-here":
            raise ValueError("Please set your OpenAI API key in config.yaml")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config['api'].get('base_url')
        )

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

        self.logger.info("OpenAI Inference Client initialized successfully")
    
    def _classify_error(self, error: Exception) -> RetryableErrorType:
        """
        Classify OpenAI-specific errors.

        Args:
            error: The exception that was raised

        Returns:
            RetryableErrorType indicating the type of error
        """
        if isinstance(error, RateLimitError):
            return RetryableErrorType.RATE_LIMIT

        if isinstance(error, APIError):
            error_str = str(error).lower()
            if "payload" in error_str and "too large" in error_str:
                return RetryableErrorType.PAYLOAD_TOO_LARGE
            return RetryableErrorType.API_ERROR

        return RetryableErrorType.UNKNOWN

    def _should_retry(self, error_type: RetryableErrorType, attempt: int) -> bool:
        """
        Determine if an OpenAI error should be retried.

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

        # Retry rate limits and API errors
        if error_type in (RetryableErrorType.RATE_LIMIT, RetryableErrorType.API_ERROR):
            return True

        # Don't retry unknown errors
        return False
    
    def _prepare_input(
        self,
        text_prompt: str,
        video_path: Optional[str] = None,
        video_processor: Optional[VideoProcessor] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare input for the API call.
        
        Args:
            text_prompt: Text prompt
            video_path: Optional path to video file
            video_processor: Optional VideoProcessor instance
            
        Returns:
            Formatted input list
        """
        content = []
        
        # Add video frames if provided
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
            
            # Extract frames
            frames = video_processor.process_video(video_path, output_format="base64")
            self.logger.info(f"Extracted {len(frames)} frames from video")
            
            # Add frames to content
            for i, frame_base64 in enumerate(frames):
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{frame_base64}"
                })
        
        # Add text prompt
        content.append({
            "type": "input_text",
            "text": text_prompt
        })
        
        return [{
            "role": "user",
            "content": content
        }]
    
    def infer(
        self,
        text_prompt: str,
        video_path: Optional[str] = None,
        model: Optional[str] = None,
        video_processor: Optional[VideoProcessor] = None,
        use_cache: bool = True,
        **model_params
    ) -> str:
        """
        Perform inference on OpenAI model.
        
        Args:
            text_prompt: Text prompt
            video_path: Optional path to video file
            model: Model name (default from config)
            video_processor: Optional VideoProcessor for custom video processing
            use_cache: Whether to use cache
            **model_params: Additional model parameters
            
        Returns:
            Model output text
            
        Raises:
            ValueError: If model doesn't support vision but video is provided
            PayloadTooLargeException: If payload is too large
            RateLimitException: If rate limit is exceeded
        """
        # Use default model if not specified
        if model is None:
            model = self.config['defaults'].get('model', 'gpt-5.2')
        
        # Validate model supports vision if video provided
        if video_path and not supports_vision(model):
            raise ValueError(f"Model '{model}' does not support vision/video inputs")
        
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get(video_path, text_prompt, model, model_params)
            if cached_result:
                self.logger.info("Using cached result")
                return cached_result['output_text']
        
        # Prepare model parameters
        model_config = get_model_config(model)
        api_type = get_api_type(model)
        
        # Set default parameters
        params = {
            'model': model,
            'temperature': model_params.get('temperature', 
                                          self.config['defaults'].get('temperature', 0.7)),
            'max_tokens': model_params.get('max_tokens',
                                          self.config['defaults'].get('max_tokens', 4096)),
        }
        
        # Add model-specific parameters
        if 'reasoning_effort' in model_config:
            params['reasoning'] = {
                'effort': model_params.get('reasoning_effort', 
                                          model_config.get('default_reasoning_effort', 'medium'))
            }
        
        if 'verbosity_levels' in model_config:
            params['text'] = {
                'verbosity': model_params.get('verbosity',
                                            model_config.get('default_verbosity', 'medium'))
            }
        
        # Prepare input
        input_data = self._prepare_input(text_prompt, video_path, video_processor)
        params['input'] = input_data
        
        # Make API call with retry logic
        self.logger.info(f"Making inference request with model: {model}")
        
        try:
            response = self._exponential_backoff(
                self.client.responses.create,
                **params
            )
        except PayloadTooLargeException:
            self.logger.error("Payload too large - cannot proceed")
            raise
        except RateLimitException:
            self.logger.warning("Rate limit exceeded - adding to queue")
            raise
        
        # Extract output text
        output_text = response.output_text
        
        # Track tokens and costs
        if hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            cached_tokens = getattr(usage, 'cached_tokens', 0)
            
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
        
        # Cache result
        if use_cache and self.cache:
            result = {
                'output_text': output_text,
                'usage': {
                    'input_tokens': input_tokens if hasattr(response, 'usage') else 0,
                    'output_tokens': output_tokens if hasattr(response, 'usage') else 0,
                    'cached_tokens': cached_tokens if hasattr(response, 'usage') else 0,
                },
                'model': model
            }
            self.cache.set(video_path, text_prompt, model, model_params, result)
        
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