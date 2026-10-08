"""Test setup: make `app` importable and run without real API keys or ML models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

# Tests must never call paid APIs or depend on the developer's .env.
for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "UNSPLASH_ACCESS_KEY", "ENABLE_LIVE_MODE", "ENTITY_BACKEND"):
    os.environ[key] = ""
