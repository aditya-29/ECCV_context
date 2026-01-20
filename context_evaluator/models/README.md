# Models Module Documentation

The `models` module provides a flexible, extensible framework for interfacing with different LLM providers. It includes abstract base classes, provider-specific implementations, caching, video processing utilities, and model configurations.

## Table of Contents

- [How to Call](#how-to-call)
  - [Basic Inference](#basic-inference)
  - [Video + Text Inference](#video--text-inference)
  - [Batch Inference](#batch-inference)
  - [Using Cache](#using-cache)
- [Module Structure](#module-structure)
- [File Documentation](#file-documentation)
  - [base.py](#basepy)
  - [providers/openai.py](#providersopenaipy)
  - [model_config.py](#model_configpy)
  - [cache_manager.py](#cache_managerpy)
  - [video_utils.py](#video_utilspy)
- [Adding a New Provider](#adding-a-new-provider)
  - [Checklist](#checklist)
  - [Step-by-Step Guide](#step-by-step-guide)
  - [Example Implementation](#example-implementation)

---

## How to Call

### Basic Inference

```python
from context_evaluator.models import OpenAIInference

# Initialize the inference client
client = OpenAIInference(config_path="config.yaml")

# Perform text-only inference
response = client.infer(
    text_prompt="Explain what a neural network is",
    model="gpt-5.2",
    temperature=0.7,
    max_tokens=1000
)

print(response)  # String output from the model
```

### Video + Text Inference

```python
from context_evaluator.models import OpenAIInference
from context_evaluator.models.video_utils import VideoProcessor, SamplingStrategy

# Initialize the client
client = OpenAIInference(config_path="config.yaml")

# Create a video processor with custom settings
video_processor = VideoProcessor(
    sampling_strategy=SamplingStrategy.UNIFORM,
    target_fps=1.0,  # Extract 1 frame per second
    max_frames=100    # Limit to 100 frames
)

# Perform inference with video + text
response = client.infer(
    text_prompt="What is happening in this video?",
    video_path="/path/to/video.mp4",
    video_processor=video_processor,
    model="gpt-5.2"
)

print(response)
```

### Batch Inference

```python
from context_evaluator.models import OpenAIInference

client = OpenAIInference(config_path="config.yaml")

# Prepare multiple requests
requests = [
    {"text_prompt": "What is AI?", "model": "gpt-5.2"},
    {"text_prompt": "Explain machine learning", "model": "gpt-5.2"},
    {"text_prompt": "What is deep learning?", "model": "gpt-5-mini"}
]

# Run batch inference (parallel processing)
results = client.batch_infer(
    requests=requests,
    num_workers=4,
    show_progress=True
)

# Process results
for result in results:
    if result['success']:
        print(f"Output: {result['output']}")
    else:
        print(f"Error: {result['error']}")
```

### Using Cache

```python
from context_evaluator.models import OpenAIInference

# Cache is enabled by default via config.yaml
client = OpenAIInference(config_path="config.yaml")

# First call - hits the API
response1 = client.infer(
    text_prompt="What is AI?",
    model="gpt-5.2"
)

# Second call with same params - returns cached result
response2 = client.infer(
    text_prompt="What is AI?",
    model="gpt-5.2"
)

# Get cache statistics
stats = client.get_usage_stats()
print(f"Cache stats: {stats['cache_stats']}")

# Clear cache if needed
if client.cache:
    client.cache.clear()
```

### Working with Model Configurations

```python
from context_evaluator.models.model_config import (
    get_model_config,
    supports_vision,
    calculate_cost,
    list_supported_models
)

# List all supported models
models = list_supported_models()
print(f"Available models: {models}")

# Check if a model supports vision
has_vision = supports_vision("gpt-5.2")
print(f"Supports vision: {has_vision}")

# Get full model configuration
config = get_model_config("gpt-5.2")
print(f"Max tokens: {config['max_tokens']}")
print(f"Context window: {config['context_window']}")

# Calculate cost for inference
cost = calculate_cost(
    model_name="gpt-5.2",
    input_tokens=1000,
    output_tokens=500,
    cached_tokens=200
)
print(f"Estimated cost: ${cost:.6f}")
```

---

## Module Structure

```
models/
├── __init__.py              # Module exports
├── base.py                  # Abstract base class for all providers
├── model_config.py          # Model configuration loader
├── cache_manager.py         # Result caching system
├── video_utils.py           # Video processing utilities
└── providers/               # Provider-specific implementations
    ├── __init__.py
    └── openai.py            # OpenAI implementation
```

---

## File Documentation

### base.py

**Purpose**: Defines the abstract base class for all model providers with common retry logic and error handling.

#### Classes

##### `RetryableErrorType` (Enum)
Enumeration of different error types for retry logic.

**Values**:
- `RATE_LIMIT`: Rate limit exceeded
- `API_ERROR`: General API error
- `PAYLOAD_TOO_LARGE`: Request payload too large
- `TIMEOUT`: Request timeout
- `UNKNOWN`: Unknown error type

##### `RateLimitException` (Exception)
Custom exception raised when rate limits are exceeded.

##### `PayloadTooLargeException` (Exception)
Custom exception raised when request payload is too large.

##### `BaseModel` (ABC)
Abstract base class for all model inference implementations.

**Constructor Parameters**:
- `model_name` (str): Name of the model
- `max_retries` (int): Maximum retry attempts (default: 5)
- `initial_backoff` (float): Initial backoff time in seconds (default: 1.0)
- `max_backoff` (float): Maximum backoff time in seconds (default: 60.0)
- `backoff_multiplier` (float): Backoff multiplier (default: 2.0)

**Abstract Methods** (must be implemented by subclasses):

- **`_classify_error(error: Exception) -> RetryableErrorType`**
  - Classifies provider-specific errors into standardized types
  - Used by retry logic to determine error handling strategy

- **`_should_retry(error_type: RetryableErrorType, attempt: int) -> bool`**
  - Determines if an error should trigger a retry
  - Provider-specific retry policies

- **`infer(*args, **kwargs) -> Any`**
  - Main inference method
  - Must be implemented for single inference requests

- **`batch_infer(*args, **kwargs) -> Any`**
  - Batch inference method
  - Must be implemented for processing multiple requests

**Protected Methods**:

- **`_handle_error(error: Exception, error_type: RetryableErrorType, attempt: int) -> None`**
  - Handles errors before retry or final failure
  - Can be overridden for custom error handling
  - Default: logs warnings for rate limits and API errors

- **`_exponential_backoff(func: Callable, *args, **kwargs) -> Any`**
  - Executes a function with exponential backoff retry logic
  - Generic retry mechanism used by all providers
  - Delegates error classification to provider-specific methods

---

### providers/openai.py

**Purpose**: OpenAI-specific implementation for GPT models with support for text, images, and video inputs.

#### Classes

##### `OpenAIInference` (extends `BaseModel`)
Main class for performing inference on OpenAI models.

**Constructor Parameters**:
- `config_path` (str): Path to YAML configuration file (default: "config.yaml")

**Key Attributes**:
- `client`: OpenAI API client
- `cache`: CacheManager instance (if enabled)
- `config`: Loaded configuration dictionary
- `total_tokens`: Token usage tracking
- `total_cost`: Cost tracking

**Public Methods**:

- **`infer(text_prompt: str, video_path: Optional[str] = None, model: Optional[str] = None, video_processor: Optional[VideoProcessor] = None, use_cache: bool = True, **model_params) -> str`**

  Performs inference on OpenAI model.

  **Parameters**:
  - `text_prompt`: Text prompt/question
  - `video_path`: Optional path to video file
  - `model`: Model name (uses config default if None)
  - `video_processor`: Custom video processor (creates default if None)
  - `use_cache`: Whether to use cached results
  - `**model_params`: Additional parameters (temperature, max_tokens, etc.)

  **Returns**: Model output text (str)

  **Raises**:
  - `ValueError`: If model doesn't support vision but video is provided
  - `PayloadTooLargeException`: If payload exceeds limits
  - `RateLimitException`: If rate limit exceeded

- **`batch_infer(requests: List[Dict[str, Any]], num_workers: Optional[int] = None, show_progress: bool = True) -> List[Dict[str, Any]]`**

  Performs batch inference with parallel processing.

  **Parameters**:
  - `requests`: List of inference request dicts (each with text_prompt, video_path, model, etc.)
  - `num_workers`: Number of parallel workers (uses config default if None)
  - `show_progress`: Show tqdm progress bar

  **Returns**: List of result dicts with 'success', 'output'/'error', 'request', 'index'

- **`get_usage_stats() -> Dict[str, Any]`**

  Returns token usage and cost statistics.

  **Returns**: Dictionary with 'total_tokens', 'total_cost', 'cache_stats'

- **`reset_stats()`**

  Resets usage statistics to zero.

**Protected Methods**:

- **`_classify_error(error: Exception) -> RetryableErrorType`**

  Classifies OpenAI-specific errors.

  Maps `RateLimitError` → RATE_LIMIT, `APIError` → API_ERROR/PAYLOAD_TOO_LARGE

- **`_should_retry(error_type: RetryableErrorType, attempt: int) -> bool`**

  OpenAI-specific retry policy.

  - Never retries: PAYLOAD_TOO_LARGE, UNKNOWN
  - Retries: RATE_LIMIT, API_ERROR (until max_retries)

- **`_prepare_input(text_prompt: str, video_path: Optional[str] = None, video_processor: Optional[VideoProcessor] = None) -> List[Dict[str, Any]]`**

  Prepares input for OpenAI API call.

  Processes video into frames (if provided) and formats content for API.

---

### model_config.py

**Purpose**: Loads and manages model configurations from YAML file.

#### Functions

- **`get_model_config(model_name: str) -> Dict[str, Any]`**

  Gets configuration for a specific model.

  **Parameters**: `model_name` - Name of the model (e.g., "gpt-5.2")

  **Returns**: Dictionary with model configuration

  **Raises**: `ValueError` if model not supported

- **`supports_vision(model_name: str) -> bool`**

  Checks if a model supports vision/image inputs.

  **Returns**: True if model supports vision

- **`supports_video(model_name: str) -> bool`**

  Checks if a model natively supports video inputs.

  **Returns**: True if model supports native video (most return False - videos converted to frames)

- **`get_api_type(model_name: str) -> str`**

  Gets the API type for a model.

  **Returns**: "responses" or "chat"

- **`calculate_cost(model_name: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float`**

  Calculates cost for model inference.

  **Parameters**:
  - `model_name`: Model to calculate cost for
  - `input_tokens`: Number of input tokens
  - `output_tokens`: Number of output tokens
  - `cached_tokens`: Number of cached input tokens (get discount)

  **Returns**: Total cost in USD

- **`list_supported_models() -> list`**

  Gets list of all supported model names.

  **Returns**: List of model names

- **`get_vision_models() -> list`**

  Gets list of models that support vision.

  **Returns**: List of vision-capable model names

#### Configuration File

Model configurations are loaded from `configs/MODEL_CONFIGS.yaml`. See that file for the full structure including pricing, context windows, reasoning effort levels, etc.

---

### cache_manager.py

**Purpose**: Manages caching of inference results to avoid redundant API calls.

#### Classes

##### `CacheManager`
Manages caching of inference results with file-based persistence.

**Constructor Parameters**:
- `cache_file` (str): Path to cache file (default: "inference_cache.json")

**Public Methods**:

- **`get(video_path: Optional[str], text_prompt: str, model: str, model_params: Dict[str, Any]) -> Optional[Dict[str, Any]]`**

  Retrieves cached result if available.

  **Parameters**:
  - `video_path`: Path to video file (None for text-only)
  - `text_prompt`: Text prompt
  - `model`: Model name
  - `model_params`: Model parameters dict

  **Returns**: Cached result dict or None if not found

- **`set(video_path: Optional[str], text_prompt: str, model: str, model_params: Dict[str, Any], result: Dict[str, Any])`**

  Stores result in cache.

  **Parameters**:
  - `video_path`: Path to video file
  - `text_prompt`: Text prompt
  - `model`: Model name
  - `model_params`: Model parameters
  - `result`: Inference result to cache

- **`clear()`**

  Clears all cache entries.

- **`get_stats() -> Dict[str, Any]`**

  Gets cache statistics.

  **Returns**: Dictionary with 'total_entries', 'cache_file'

**Protected Methods**:

- **`_generate_key(video_path: Optional[str], text_prompt: str, model: str, model_params: Dict[str, Any]) -> str`**

  Generates MD5 hash key from input parameters.

- **`_load_cache() -> Dict[str, Any]`**

  Loads cache from JSON file.

- **`_save_cache()`**

  Saves cache to JSON file.

---

### video_utils.py

**Purpose**: Utilities for processing video files into frames for model inference.

#### Classes

##### `SamplingStrategy` (Enum)
Video frame sampling strategies.

**Values**:
- `UNIFORM`: Evenly spaced frames
- `ADAPTIVE`: Content-aware sampling (future)
- `KEYFRAME`: Extract only keyframes (future)

##### `VideoProcessor`
Processes videos into frames for vision models.

**Constructor Parameters**:
- `sampling_strategy` (SamplingStrategy): Frame sampling strategy (default: UNIFORM)
- `target_fps` (float): Target frames per second (default: 1.0)
- `max_frames` (int): Maximum frames to extract (default: 100)
- `resize_dimensions` (Optional[Tuple[int, int]]): Resize dimensions (default: None)

**Public Methods**:

- **`process_video(video_path: str, output_format: str = "base64") -> Union[List[str], List[np.ndarray]]`**

  Processes video into frames.

  **Parameters**:
  - `video_path`: Path to video file
  - `output_format`: "base64" for API, "numpy" for processing

  **Returns**: List of base64-encoded strings or numpy arrays

  **Raises**:
  - `FileNotFoundError`: If video file not found
  - `ValueError`: If video cannot be opened or invalid format

- **`get_video_info(video_path: str) -> Dict[str, Any]`**

  Gets video metadata.

  **Returns**: Dictionary with 'fps', 'frame_count', 'duration', 'width', 'height'

**Protected Methods**:

- **`_uniform_sampling(video_path: str) -> List[np.ndarray]`**

  Extracts frames using uniform sampling.

- **`_encode_frame_base64(frame: np.ndarray) -> str`**

  Encodes numpy frame to base64 JPEG string.

---

## Adding a New Provider

### Checklist

To add a new provider (e.g., Anthropic Claude, Google Gemini), follow this checklist:

- [ ] **1. Create provider file**: Create `providers/<provider_name>.py`
- [ ] **2. Create provider class**: Implement class extending `BaseModel`
- [ ] **3. Implement abstract methods**:
  - [ ] `_classify_error()` - Map provider errors to `RetryableErrorType`
  - [ ] `_should_retry()` - Define retry policy
  - [ ] `infer()` - Single inference implementation
  - [ ] `batch_infer()` - Batch inference implementation
- [ ] **4. Add model configurations**: Update `configs/MODEL_CONFIGS.yaml` with new models
- [ ] **5. Update provider __init__.py**: Export new class from `providers/__init__.py`
- [ ] **6. Update models __init__.py**: Export new class from `models/__init__.py`
- [ ] **7. Add configuration template**: Create example config in `configs/example_config.yaml`
- [ ] **8. Install dependencies**: Add provider SDK to `pyproject.toml`
- [ ] **9. Write tests**: Create tests for the new provider
- [ ] **10. Update documentation**: Add provider to this README

### Step-by-Step Guide

#### Step 1: Create Provider File

Create `models/providers/anthropic.py` (example):

```python
"""
Anthropic Claude Inference Module
"""

import anthropic
from typing import Optional, List, Dict, Any

from context_evaluator.models import (
    BaseModel,
    RateLimitException,
    PayloadTooLargeException,
    RetryableErrorType
)
```

#### Step 2: Implement Provider Class

```python
class AnthropicInference(BaseModel):
    """
    Main class for performing inference on Anthropic Claude models.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize base class
        retry_config = self.config.get('retry', {})
        super().__init__(
            model_name=self.config['defaults'].get('model', 'claude-3-5-sonnet'),
            max_retries=retry_config.get('max_retries', 5),
            initial_backoff=retry_config.get('initial_backoff', 1.0),
            max_backoff=retry_config.get('max_backoff', 60.0),
            backoff_multiplier=retry_config.get('backoff_multiplier', 2.0)
        )

        # Setup logging
        self.logger = setup_logging(self.config)

        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=self.config['api']['anthropic_api_key']
        )
```

#### Step 3: Implement Abstract Methods

```python
    def _classify_error(self, error: Exception) -> RetryableErrorType:
        """Classify Anthropic-specific errors."""
        if isinstance(error, anthropic.RateLimitError):
            return RetryableErrorType.RATE_LIMIT

        if isinstance(error, anthropic.APIError):
            if "overloaded" in str(error).lower():
                return RetryableErrorType.API_ERROR
            if "request too large" in str(error).lower():
                return RetryableErrorType.PAYLOAD_TOO_LARGE
            return RetryableErrorType.API_ERROR

        return RetryableErrorType.UNKNOWN

    def _should_retry(self, error_type: RetryableErrorType, attempt: int) -> bool:
        """Anthropic-specific retry policy."""
        # Never retry payload errors
        if error_type == RetryableErrorType.PAYLOAD_TOO_LARGE:
            return False

        # Don't retry on last attempt
        if attempt >= self.max_retries - 1:
            return False

        # Retry rate limits and API errors
        return error_type in (RetryableErrorType.RATE_LIMIT, RetryableErrorType.API_ERROR)

    def infer(self, text_prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Perform inference on Claude model."""
        model = model or self.model_name

        # Prepare request
        response = self._exponential_backoff(
            self.client.messages.create,
            model=model,
            messages=[{"role": "user", "content": text_prompt}],
            max_tokens=kwargs.get('max_tokens', 4096)
        )

        return response.content[0].text

    def batch_infer(self, requests: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Implement batch inference for Claude."""
        # Implementation here
        pass
```

#### Step 4: Add Model Configurations

Update `configs/MODEL_CONFIGS.yaml`:

```yaml
# Anthropic Claude Models
claude-3-5-sonnet:
  supports_vision: true
  supports_video: false
  api_type: messages
  max_tokens: 8192
  context_window: 200000
  pricing:
    input_tokens_per_million: 3.0
    output_tokens_per_million: 15.0
    cached_input_discount: 0.90
  description: Anthropic's most capable model

claude-3-5-haiku:
  supports_vision: true
  supports_video: false
  api_type: messages
  max_tokens: 8192
  context_window: 200000
  pricing:
    input_tokens_per_million: 0.8
    output_tokens_per_million: 4.0
    cached_input_discount: 0.90
  description: Fast and efficient Claude model
```

#### Step 5: Update Exports

Update `models/providers/__init__.py`:

```python
from context_evaluator.models.providers.openai import OpenAIInference
from context_evaluator.models.providers.anthropic import AnthropicInference

__all__ = ['OpenAIInference', 'AnthropicInference']
```

Update `models/__init__.py`:

```python
from context_evaluator.models.providers import OpenAIInference, AnthropicInference

__all__ = [
    'BaseModel',
    'RateLimitException',
    'PayloadTooLargeException',
    'RetryableErrorType',
    'OpenAIInference',
    'AnthropicInference'  # Add new provider
]
```

#### Step 6: Add Dependencies

Update `pyproject.toml`:

```toml
dependencies = [
    "openai>=1.50.0",
    "anthropic>=0.18.0",  # Add new provider SDK
    "opencv-python>=4.8.0",
    # ... other dependencies
]
```

#### Step 7: Update Configuration

Add to `configs/example_config.yaml`:

```yaml
api:
  anthropic_api_key: "your-anthropic-api-key-here"
  # ... other API keys

defaults:
  model: "claude-3-5-sonnet"  # Can change default
```

### Example Implementation

See `models/providers/openai.py` for a complete reference implementation.

---

## Best Practices

1. **Error Handling**: Always use the `_exponential_backoff` method for API calls
2. **Logging**: Use `self.logger` for consistent logging
3. **Configuration**: Load all settings from YAML config files
4. **Caching**: Integrate with CacheManager for consistent caching behavior
5. **Type Hints**: Use type hints for all function parameters and returns
6. **Documentation**: Add docstrings to all public methods
7. **Testing**: Write unit tests for error classification and retry logic

---

## Common Issues

### Import Errors
- Ensure `__init__.py` files are updated with new exports
- Check Python path includes project root

### Configuration Not Loading
- Verify YAML syntax in config files
- Check file paths are correct (absolute or relative to working directory)
- Ensure `yaml` package is installed

### Cache Not Working
- Check cache directory is writable
- Verify cache is enabled in config
- Use `cache.get_stats()` to debug

### Video Processing Fails
- Ensure OpenCV is properly installed (`opencv-python`)
- Check video file format is supported
- Verify video file path is correct

---

## License

MIT License - See project root LICENSE file.
