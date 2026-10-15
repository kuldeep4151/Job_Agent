"""Project paths, environment loading, and config/answers files. Import this before anything that reads env vars."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
LOG_DIR = DATA_DIR / "logs"
RESUME_DIR = DATA_DIR / "resumes"

CONFIG_PATH = CONFIG_DIR / "config.yaml"
ANSWERS_PATH = CONFIG_DIR / "answers.yaml"
PROFILE_PATH = CONFIG_DIR / "profile.json"
MASTER_RESUME_PDF = RESUME_DIR / "master.pdf"

# Values the user still has to fill in. Anything containing these must never reach a live application.
PLACEHOLDER = "TODO"
API_KEY_PLACEHOLDER = "sk-ant-REPLACE_ME"

load_dotenv(ROOT / ".env")

for _dir in (CONFIG_DIR, DATA_DIR, LOG_DIR, RESUME_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def stored_path(path: Path) -> str:
    """How file paths are kept in the DB: relative to the project when inside it (tests use tmp dirs)."""
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def unfilled(data: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose value is still the PLACEHOLDER (also inside lists)."""
    if isinstance(data, dict):
        return [k for key, value in data.items() for k in unfilled(value, f"{prefix}{key}.")]
    if isinstance(data, list):
        return [prefix.rstrip(".")] if any(v == PLACEHOLDER for v in data) else []
    return [prefix.rstrip(".")] if data == PLACEHOLDER else []
