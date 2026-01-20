"""
Model Configuration Module
Loads model-specific information from YAML configuration file.
Includes support for vision, pricing, and API details.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Path to model configs YAML file
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
_MODEL_CONFIGS_PATH = _CONFIGS_DIR / "MODEL_CONFIGS.yaml"

# Global variable to store loaded configs
_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = None


def _load_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Load model configurations from YAML file.

    Returns:
        Dictionary of model configurations

    Raises:
        FileNotFoundError: If MODEL_CONFIGS.yaml is not found
        yaml.YAMLError: If YAML file is malformed
    """
    if not _MODEL_CONFIGS_PATH.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {_MODEL_CONFIGS_PATH}"
        )

    with open(_MODEL_CONFIGS_PATH, 'r') as f:
        configs = yaml.safe_load(f)

    return configs


def _get_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get model configurations, loading them lazily on first access.

    Returns:
        Dictionary of model configurations
    """
    global _MODEL_CONFIGS

    if _MODEL_CONFIGS is None:
        _MODEL_CONFIGS = _load_model_configs()

    return _MODEL_CONFIGS


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If model is not supported
    """
    configs = _get_configs()
    if model_name not in configs:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Supported models: {list(configs.keys())}"
        )
    return configs[model_name]


def supports_vision(model_name: str) -> bool:
    """Check if a model supports vision/image inputs."""
    config = get_model_config(model_name)
    return config.get("supports_vision", False)


def supports_video(model_name: str) -> bool:
    """Check if a model natively supports video inputs."""
    config = get_model_config(model_name)
    return config.get("supports_video", False)


def get_api_type(model_name: str) -> str:
    """Get the API type for a model (responses or chat)."""
    config = get_model_config(model_name)
    return config.get("api_type", "responses")


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int, 
                   cached_tokens: int = 0) -> float:
    """
    Calculate the cost for a model inference.
    
    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens
        
    Returns:
        Total cost in USD
    """
    config = get_model_config(model_name)
    pricing = config.get("pricing", {})
    
    # Calculate input cost
    regular_input_tokens = input_tokens - cached_tokens
    input_cost = (regular_input_tokens / 1_000_000) * pricing.get("input_tokens_per_million", 0)
    
    # Apply cache discount if available
    if cached_tokens > 0 and "cached_input_discount" in pricing:
        discount = pricing["cached_input_discount"]
        cached_cost = (cached_tokens / 1_000_000) * pricing.get("input_tokens_per_million", 0)
        cached_cost *= (1 - discount)
        input_cost += cached_cost
    
    # Calculate output cost
    output_cost = (output_tokens / 1_000_000) * pricing.get("output_tokens_per_million", 0)
    
    return input_cost + output_cost


def list_supported_models() -> list:
    """Get list of all supported model names."""
    return list(_get_configs().keys())


def get_vision_models() -> list:
    """Get list of models that support vision."""
    configs = _get_configs()
    return [name for name, config in configs.items()
            if config.get("supports_vision", False)]