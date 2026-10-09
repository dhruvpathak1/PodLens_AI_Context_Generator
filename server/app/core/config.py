"""Central configuration for the PodLens API.

Every environment variable the backend reads is declared here, in one place, with its default.
Other modules import `settings` (or call `get_settings()`) instead of reading `os.environ`
directly, so configuration is discoverable and testable.

Loading order
1. `<repo>/.env` is loaded once via python-dotenv (real environment variables win).
2. `TORCH_HOME` is defaulted to a folder inside `server/` *before* Whisper/PyTorch is imported
   anywhere, because macOS sandboxes often deny writes to `~/.cache`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: `server/` directory (this file lives at server/app/core/config.py).
SERVER_DIR: Path = Path(__file__).resolve().parents[2]
#: Repository root (parent of `server/`); holds the shared `.env` file.
PROJECT_ROOT: Path = SERVER_DIR.parent
#: Shared env file used by both the API and the Vite frontend.
ENV_FILE: Path = PROJECT_ROOT / ".env"

# `utf-8-sig` tolerates a BOM, which some editors add on Windows.
load_dotenv(ENV_FILE, encoding="utf-8-sig")

# Keep PyTorch caches inside the project. Must run before `import whisper` (which imports torch).
os.environ.setdefault("TORCH_HOME", str(SERVER_DIR / "torch_home"))


# ---------------------------------------------------------------------------
# Small env helpers
# ---------------------------------------------------------------------------


def env_str(name: str, default: str = "") -> str:
    """Return the trimmed value of an environment variable, or `default` when unset/blank."""
    value = os.environ.get(name, "").strip()
    return value or default


def env_int(name: str, default: int) -> int:
    """Return an integer environment variable, falling back to `default` when unset or invalid."""
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Return a float environment variable, falling back to `default` when unset or invalid."""
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    """Interpret `1/true/yes/on` (case-insensitive) as True; anything else as False."""
    raw = env_str(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def env_list(name: str) -> list[str]:
    """Split a comma-separated environment variable into a list of non-empty, trimmed items."""
    return [item.strip() for item in env_str(name).split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

#: Local Vite dev-server origins that are always allowed by CORS.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration (read once at startup)."""

    # --- Whisper (file transcription) ---
    whisper_model: str = field(default_factory=lambda: env_str("WHISPER_MODEL", "base"))
    whisper_download_root: Path = field(
        default_factory=lambda: Path(env_str("WHISPER_DOWNLOAD_ROOT", str(SERVER_DIR / "whisper_models")))
    )
    ffmpeg_path: str = field(default_factory=lambda: env_str("FFMPEG_PATH"))

    # --- Local storage of transcripts / entity exports ---
    transcripts_dir: Path = field(
        default_factory=lambda: Path(env_str("TRANSCRIPTS_DIR", str(SERVER_DIR / "transcripts")))
    )
    entity_json_dir: Path = field(
        default_factory=lambda: Path(env_str("ENTITY_JSON_DIR", str(SERVER_DIR / "entity_exports")))
    )

    # --- Entity extraction (NER) ---
    entity_backend: str = field(default_factory=lambda: env_str("ENTITY_BACKEND").lower())
    spacy_model: str = field(default_factory=lambda: env_str("SPACY_MODEL", "en_core_web_sm"))
    claude_model: str = field(default_factory=lambda: env_str("CLAUDE_MODEL", "claude-haiku-4-5-20251001"))
    claude_entity_batch: int = field(default_factory=lambda: env_int("CLAUDE_ENTITY_BATCH", 12))

    # --- Enrichment (Wikipedia / Nominatim / Unsplash) ---
    unsplash_access_key: str = field(default_factory=lambda: env_str("UNSPLASH_ACCESS_KEY"))
    nominatim_user_agent: str = field(
        default_factory=lambda: env_str(
            "NOMINATIM_USER_AGENT", "PodcastTranscriptEnrich/1.0 (https://github.com/local; educational use)"
        )
    )

    # --- OpenAI (timeline + ask) ---
    openai_api_key: str = field(default_factory=lambda: env_str("OPENAI_API_KEY"))
    timeline_model: str = field(default_factory=lambda: env_str("OPENAI_TIMELINE_MODEL", "gpt-5.4-mini"))
    timeline_window_chars: int = field(default_factory=lambda: env_int("TIMELINE_WINDOW_CHARS", 24000))
    timeline_max_concurrency: int = field(default_factory=lambda: env_int("TIMELINE_MAX_CONCURRENCY", 4))
    timeline_max_transcript_chars: int = field(
        default_factory=lambda: env_int("TIMELINE_MAX_TRANSCRIPT_CHARS", 600000)
    )
    ask_model: str = field(default_factory=lambda: env_str("OPENAI_ASK_MODEL", "gpt-5.4-mini"))
    embed_model: str = field(default_factory=lambda: env_str("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    ask_full_context_chars: int = field(default_factory=lambda: env_int("ASK_FULL_CONTEXT_CHARS", 60000))
    ask_chunk_sec: float = field(default_factory=lambda: env_float("ASK_CHUNK_SEC", 60.0))
    ask_chunk_overlap_sec: float = field(default_factory=lambda: env_float("ASK_CHUNK_OVERLAP_SEC", 15.0))
    ask_top_k: int = field(default_factory=lambda: env_int("ASK_TOP_K", 8))
    #: LLM review of extracted entities before/after building source cards (needs OPENAI_API_KEY).
    entity_review: bool = field(default_factory=lambda: env_bool("ENTITY_REVIEW", True))
    review_model: str = field(default_factory=lambda: env_str("OPENAI_REVIEW_MODEL", "gpt-5.4-mini"))

    # --- HTTP / deployment ---
    cors_extra_origins: tuple[str, ...] = field(default_factory=lambda: tuple(env_list("CORS_EXTRA_ORIGINS")))
    static_dist_dir: str = field(default_factory=lambda: env_str("STATIC_DIST_DIR"))

    # --- Parked features ---
    #: Live microphone mode is future scope; its routes are only mounted when this is true.
    enable_live_mode: bool = field(default_factory=lambda: env_bool("ENABLE_LIVE_MODE", False))

    @property
    def cors_allow_origins(self) -> list[str]:
        """Origins allowed by CORS: local Vite ports plus `CORS_EXTRA_ORIGINS`."""
        return [*_DEFAULT_CORS_ORIGINS, *self.cors_extra_origins]

    @property
    def openai_configured(self) -> bool:
        """True when an OpenAI key is present (required by Timeline and Ask)."""
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide `Settings` instance (created on first call)."""
    return Settings()


#: Convenience alias for modules that only need read access.
settings = get_settings()
