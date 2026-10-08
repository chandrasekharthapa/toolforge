"""Runtime settings, read from environment variables (and a .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: variables that were already set in the process environment before .env was read
PRESET_ENV = frozenset(os.environ)
#: variables the project's .env file defines
DOTENV_KEYS: frozenset[str] = frozenset()
#: path of the .env file that was loaded ("" when none)
DOTENV_PATH = ""

try:  # optional convenience
    from dotenv import dotenv_values, find_dotenv, load_dotenv

    DOTENV_PATH = find_dotenv(usecwd=True) or find_dotenv()
    if DOTENV_PATH:
        DOTENV_KEYS = frozenset(k for k, v in dotenv_values(DOTENV_PATH).items() if v is not None)
        # The project's .env WINS over inherited variables. Editors (VS Code's Python extension)
        # copy .env into every new terminal; after the file is edited that stale copy would
        # otherwise silently override it. A variable .env does not define still comes from the shell.
        load_dotenv(DOTENV_PATH, override=True)
except ImportError:  # pragma: no cover
    pass


def env_source(name: str) -> str:
    """Where a setting's value comes from: '.env', 'shell environment' or 'default'."""
    if name in DOTENV_KEYS:
        return ".env"
    return "shell environment" if name in PRESET_ENV else "default"


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
    # Tool MAKER / tool USER split (LATM): when TOOLFORGE_USER_MODEL is set, planning and execution
    # (the per-query work) use this cheaper model, while writing, verifying and judging tools stays on
    # TOOLFORGE_MODEL. The provider defaults to TOOLFORGE_PROVIDER.
    user_provider: str | None = field(default_factory=lambda: _env("TOOLFORGE_USER_PROVIDER"))
    user_model: str | None = field(default_factory=lambda: _env("TOOLFORGE_USER_MODEL"))

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
    #: who decides reuse vs. build among retrieved candidates: "llm" (default) or "reranker"
    judge: str = field(default_factory=lambda: _env("TOOLFORGE_JUDGE", "llm"))
    reranker_path: str = field(default_factory=lambda: _env("TOOLFORGE_RERANKER_PATH", "evals/distill/reranker"))
    lexical_weight: float | None = field(
        default_factory=lambda: float(v) if (v := _env("TOOLFORGE_LEXICAL_WEIGHT")) else None
    )

    # --- forging ------------------------------------------------------------
    max_needs: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_NEEDS", 3))
    #: re-ask the planner once if it plans fewer needs (1 = "this task must produce a tool")
    min_needs: int = field(default_factory=lambda: _int("TOOLFORGE_MIN_NEEDS", 0))
    # show the planner the library's closest verified tools, so it can plan to reuse one by name
    plan_with_library: bool = field(default_factory=lambda: _bool("TOOLFORGE_PLAN_WITH_LIBRARY", False))
    max_repairs: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_REPAIRS", 3))
    min_tests: int = field(default_factory=lambda: _int("TOOLFORGE_MIN_TESTS", 3))

    # differential verification: an independent second implementation is
    # fuzzed against the candidate before it may be registered
    differential: bool = field(default_factory=lambda: _bool("TOOLFORGE_DIFFERENTIAL", True))
    fuzz_cases: int = field(default_factory=lambda: _int("TOOLFORGE_FUZZ_CASES", 40))
    max_disagreement: float = field(default_factory=lambda: _float("TOOLFORGE_MAX_DISAGREEMENT", 0.0))

    # --- execution ----------------------------------------------------------
    max_exec_steps: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_EXEC_STEPS", 8))
    #: send a tool that raised on every real call back to the maker for one repair, then retry the task
    field_repair: bool = field(default_factory=lambda: _bool("TOOLFORGE_FIELD_REPAIR", True))
    max_field_repairs: int = field(default_factory=lambda: _int("TOOLFORGE_MAX_FIELD_REPAIRS", 1))
    #: when a tool was handed the whole task ("<<TASK>>") and returned a string, that string is the
    #: answer; the model is not asked to retype it (small models garble even correct results)
    answer_from_task_tool: bool = field(default_factory=lambda: _bool("TOOLFORGE_ANSWER_FROM_TASK_TOOL", False))

    # --- sandbox ------------------------------------------------------------
    sandbox_timeout: float = field(default_factory=lambda: _float("TOOLFORGE_SANDBOX_TIMEOUT", 5.0))
    sandbox_memory_mb: int = field(default_factory=lambda: _int("TOOLFORGE_SANDBOX_MEMORY_MB", 256))
    # "process" (python -I subprocess + audit hook) or "docker" (same harness in a locked-down container)
    sandbox: str = field(default_factory=lambda: _env("TOOLFORGE_SANDBOX", "process"))
    sandbox_image: str = field(default_factory=lambda: _env("TOOLFORGE_SANDBOX_IMAGE", "python:3.12-slim"))
