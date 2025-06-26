import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

def get_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    """
    Load .env file and return the value of the specified environment variable.
    Raises an error if not found.
    """
    load_dotenv()
    key = os.getenv(env_var)
    if not key:
        raise ValueError(f"Environment variable {env_var} not set.")
    return key


def clean_text(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces and newlines into single spaces.
    """
    return " ".join(text.split())


def load_json(path: Path) -> dict:
    """
    Read a JSON file from disk and return the parsed object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    """
    Pretty-print a Python object to JSON and save to the given path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with a standard format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger