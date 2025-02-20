import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_data_dir() -> Path:
    """Get the configured data directory, failing if not set"""
    data_dir_str = os.getenv('DATA_DIR')
    if not data_dir_str:
        raise ValueError("DATA_DIR environment variable must be set")
    data_dir = Path(data_dir_str)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_cache_dir() -> Path:
    """Get the configured cache directory, failing if not set"""
    cache_dir_str = os.getenv('CACHE_DIR')
    if not cache_dir_str:
        raise ValueError("CACHE_DIR environment variable must be set")
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir 