"""
Utility functions for configuration loading, logging, and paths.
Beginner-friendly and well-documented.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Returns the absolute Path to the project root directory."""
    return Path(__file__).resolve().parent.parent

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file safely.
    
    Args:
        config_path: Relative or absolute path to YAML config file.
        
    Returns:
        dict containing configuration parameters.
    """
    root = get_project_root()
    resolved_path = root / config_path if not os.path.isabs(config_path) else Path(config_path)
    
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")
        
    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

def setup_logger(name: str = "MLOps", level: int = logging.INFO) -> logging.Logger:
    """
    Create a clean, standardized console logger for MLOps pipelines.
    
    Args:
        name: Name of the logger.
        level: Logging level (default: INFO).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
