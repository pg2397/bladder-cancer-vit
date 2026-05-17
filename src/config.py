"""YAML config loader with light schema sanity checks."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config from disk and return a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Config at {p} did not parse to a mapping.")
    _require(cfg, "experiment.name")
    _require(cfg, "experiment.seed")
    return dict(cfg)


def _require(cfg: Mapping, dotted: str) -> None:
    """Validate that a dotted path exists in the config; raise otherwise."""
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"Missing required config key: {dotted}")
        node = node[part]
