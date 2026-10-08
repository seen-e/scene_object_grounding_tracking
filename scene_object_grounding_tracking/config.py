from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    unresolved = _ENV_REFERENCE.findall(expanded)
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(f"unresolved environment variables: {names}")
    return expanded


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return _expand_environment(config)


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"missing config section: {name}")
    return section

