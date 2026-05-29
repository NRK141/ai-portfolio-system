from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    output_dir = Path(config["report"]["output_dir"])
    for sub in ["reports", "charts", "csv"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
