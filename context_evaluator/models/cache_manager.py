"""
Cache Manager for OpenAI Inference
Stores inference results keyed by video path and model parameters.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of inference results.
    Cache key is generated from video path and model parameters.
    """

    def __init__(self, cache_file: str = "inference_cache.json"):
        """
        Initialize cache manager.

        Args:
            cache_file: Path to cache file
        """
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        logger.info(f"CacheManager initialized: cache_file={cache_file}")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
            except json.JSONDecodeError:
                logger.warning(f"Cache file {self.cache_file} is corrupted, starting fresh")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.debug(f"Cache saved with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _generate_key(
        self,
        video_path: Optional[str],
        text_prompt: str,
        model: str,
        model_params: Dict[str, Any]
    ) -> str:
        """
        Generate cache key from inputs.
        
        Args:
            video_path: Path to video file (None for text-only)
            text_prompt: Text prompt
            model: Model name
            model_params: Model parameters
            
        Returns:
            MD5 hash as cache key
        """
        # Create a dict with all parameters
        key_data = {
            "video_path": str(video_path) if video_path else None,
            "text_prompt": text_prompt,
            "model": model,
            "model_params": model_params
        }
        
        # Convert to JSON and hash
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return key_hash
    
    def get(
        self,
        video_path: Optional[str],
        text_prompt: str,
        model: str,
        model_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available.

        Args:
            video_path: Path to video file (None for text-only)
            text_prompt: Text prompt
            model: Model name
            model_params: Model parameters

        Returns:
            Cached result or None if not found
        """
        key = self._generate_key(video_path, text_prompt, model, model_params)

        if key not in self.cache:
            logger.debug(f"Cache miss for key: {key}")
            return None

        entry = self.cache[key]
        logger.info(f"Cache hit for key: {key}")
        return entry['result']
    
    def set(
        self,
        video_path: Optional[str],
        text_prompt: str,
        model: str,
        model_params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Store result in cache.

        Args:
            video_path: Path to video file (None for text-only)
            text_prompt: Text prompt
            model: Model name
            model_params: Model parameters
            result: Inference result to cache
        """
        key = self._generate_key(video_path, text_prompt, model, model_params)

        self.cache[key] = {
            "video_path": str(video_path) if video_path else None,
            "text_prompt": text_prompt,
            "model": model,
            "model_params": model_params,
            "result": result
        }

        self._save_cache()
        logger.info(f"Cached result for key: {key}")
    
    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self.cache)

        return {
            "total_entries": total_entries,
            "cache_file": str(self.cache_file)
        }