"""
config.py — Centralised configuration constants for Voice Notes Agent.
All tuneable values live here. Import from this module everywhere else.
"""

import os
import queue

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL = "http://localhost:12345/v1/chat/completions"
LLAMA_MODEL      = "gemma"   # llama-server ignores this field but the schema requires it

# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # 16 kHz — optimal for Whisper

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# BASE_DIR points to the project root (one level above this package).
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_TMP    = os.path.join(BASE_DIR, "rec_temp.wav")
SESSION_FILE = os.path.join(BASE_DIR, ".session.json")

# ---------------------------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------------------------
# Tracks which .md file is currently open so the agent can append in context.
agent_state: dict = {
    "current_file": None,   # absolute path of the active .md file
    "last_section": None,   # last heading/topic written (for continuity)
    "file_summary":  None,  # plain-text summary of current file (or None)
}

# In-memory queue: Whisper produces → agent worker consumes.
# Typed as Queue[str]; None is used as the shutdown sentinel.
text_queue: queue.Queue = queue.Queue()
