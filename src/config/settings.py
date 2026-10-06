"""
config/settings.py

Global configuration loader with environment variable substitution.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_cwd() -> None:
    """Re-anchor CWD to PROJECT_ROOT if the current directory is stale.

    The /mnt/ssd mount can remount during long operations, leaving
    os.getcwd() pointing at a deleted inode.  Libraries like `datasets`
    call Path().resolve() internally, which crashes in that state.
    """
    try:
        os.getcwd()
    except (FileNotFoundError, OSError):
        os.chdir(PROJECT_ROOT)


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        pattern = re.compile(r'\$\{(\w+)\}')
        def replacer(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                import warnings
                warnings.warn(f"Environment variable {var_name} is not set; leaving placeholder")
                return f"${{{var_name}}}"
            return env_val
        return pattern.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(config_path: str, substitute_env: bool = True) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.
        substitute_env: If True, replace ${VAR} with environment variable values.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if substitute_env:
        config = _substitute_env_vars(config)

    # Resolve pipeline.domain or top-level domain, defaulting to "offsec".
    if "domain" not in config:
        pipeline = config.get("pipeline", {})
        if isinstance(pipeline, dict) and "domain" in pipeline:
            config["domain"] = pipeline["domain"]

    return config
