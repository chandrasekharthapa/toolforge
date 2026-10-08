import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from toolforge.agent import Toolforge  # noqa: E402
from toolforge.config import Settings  # noqa: E402
from toolforge.embeddings import HashingEmbedder  # noqa: E402
from toolforge.registry import Registry  # noqa: E402

# Settings a developer's .env may set that would change behaviour under test (a real user model,
# the Docker sandbox, the reranker judge). Tests describe the defaults, so they are cleared.
_LOCAL_ONLY = ("TOOLFORGE_USER_MODEL", "TOOLFORGE_USER_PROVIDER", "TOOLFORGE_PLAN_WITH_LIBRARY", "TOOLFORGE_SANDBOX",
               "TOOLFORGE_JUDGE", "TOOLFORGE_MIN_NEEDS", "TOOLFORGE_FIELD_REPAIR")


@pytest.fixture(autouse=True)
def _defaults_only(monkeypatch):
    for name in _LOCAL_ONLY:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_forge(tmp_path):
    def factory(brain, *, db: str = "lib.db", rag: bool = True, **overrides):
        settings = Settings(provider="scripted", embedder="hashing", sandbox="process", user_model=None,
                            plan_with_library=False, db_path=str(tmp_path / db),
                            fuzz_cases=15, max_repairs=2)
        for key, value in overrides.items():
            setattr(settings, key, value)
        return Toolforge(settings, llm=brain.llm(), embedder=HashingEmbedder(),
                         registry=Registry(settings.db_path), rag=rag)

    return factory
