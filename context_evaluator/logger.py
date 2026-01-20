import logging
from typing import Optional, List, Dict, Any, Union

# Setup logging
def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('log_file', 'openai_inference.log')
    log_format = log_config.get('log_format', 
                                 '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)