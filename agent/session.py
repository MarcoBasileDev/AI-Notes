"""
session.py — Persistence helpers for the last active note file.
"""

import json
import os

from agent.config import SESSION_FILE


def save_session(file_path: str | None) -> None:
    """Persist the last active file path so it can be restored on next launch."""
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_file": file_path}, f)
    except OSError:
        pass  # non-critical — silently ignore write errors


def load_last_session() -> str | None:
    """Return the last active file path, or None if not found / file deleted."""
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = data.get("last_file")
        if path and os.path.isfile(path):
            return path
    except (OSError, json.JSONDecodeError):
        pass
    return None
