import json
import os
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"

def load_config(name="config.json"):
    path = CONFIG_ROOT / name
    if not path.exists():
        return {}
    with open(path, "r") as f:
        if path.suffix == ".json":
            return json.load(f)
        else:
            return f.read()

def initialize():
    cfg = load_config()
    return cfg

__all__ = ["load_config", "initialize"]
