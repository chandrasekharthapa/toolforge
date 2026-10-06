"""Runtime settings, read from environment variables (and a .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: variables that were already set before .env was read (they take priority over .env)
PRESET_ENV = frozenset(os.environ)

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return (_env(name, str(default)) or "").lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- models -------------------------------------------------------------
    provider: str = field(default_factory=lambda: _env("TOOLFORGE_PROVIDER", "gemini"))
    model: str | None = field(default_factory=lambda: _env("TOOLFORGE_MODEL"))
    base_url: str | None = field(default_factory=lambda: _env("TOOLFORGE_BASE_URL"))
    embedder: str = field(default_factory=lambda: _env("TOOLFORGE_EMBEDDER", "hashing"))
    embed_model: str | None = field(default_factory=lambda: _env("TOOLFORGE_EMBED_MODEL"))

    # --- storage ------------------------------------------------------------
    db_path: str = field(default_factory=lambda: _env("TOOLFORGE_DB", "toolforge.db"))

    # --- retrieval (None = use the embedder's calibrated defaults) ----------
    reuse_threshold: float | None = field(
        default_factory=lambda: float(v) if (v := _env("TOOLFORGE_REUSE_THRESHOLD")) else None
    )
    consider_threshold: float | None = field(
        default_factory=lambda: float(v) if (v := _env("TOOLFORGE_CONSIDER_THRESHOLD")) else None
    )
    top_k: int = field(default_factory=lambda: _int("TOOLFORGE_TOP_K", 4))

    # --- forging ------------------------------------------------------------
    max_needs: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_NEEDS", 3))
    max_repairs: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_REPAIRS", 3))
    min_tests: int = field(default_factory=lambda: _int("TOOLFORGE_MIN_TESTS", 3))

    # differential verification: an independent second implementation is
    # fuzzed against the candidate before it may be registered
    differential: bool = field(default_factory=lambda: _bool("TOOLFORGE_DIFFERENTIAL", True))
    fuzz_cases: int = field(default_factory=lambda: _int("TOOLFORGE_FUZZ_CASES", 40))
    max_disagreement: float = field(default_factory=lambda: _float("TOOLFORGE_MAX_DISAGREEMENT", 0.0))

    # --- execution ----------------------------------------------------------
    max_exec_steps: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_EXEC_STEPS", 8))

    # --- sandbox ------------------------------------------------------------
    sandbox_timeout: float = field(default_factory=lambda: _float("TOOLFORGE_SANDBOX_TIMEOUT", 5.0))
    sandbox_memory_mb: int = field(default_factory=lambda: _int("TOOLFORGE_SANDBOX_MEMORY_MB", 256))
