"""Resolve project paths independently of the current working directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def project_path(env: str, default: str) -> Path:
    return Path(os.getenv(env, str(ROOT / default))).expanduser().resolve()

MODEL_DIR = project_path("TAXONOMYIQ_MODEL_DIR", "models")
DB_PATH = project_path("TAXONOMYIQ_DB_PATH", "runtime/taxonomyiq.sqlite3")
API_KEY = os.getenv("TAXONOMYIQ_API_KEY", "")
REVIEW_THRESHOLD = float(os.getenv("TAXONOMYIQ_REVIEW_THRESHOLD", "0.70"))
