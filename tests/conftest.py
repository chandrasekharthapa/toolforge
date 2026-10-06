import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from toolforge.agent import Toolforge  # noqa: E402
from toolforge.config import Settings  # noqa: E402
from toolforge.embeddings import HashingEmbedder  # noqa: E402
from toolforge.registry import Registry  # noqa: E402


@pytest.fixture
def make_forge(tmp_path):
    def factory(brain, *, db: str = "lib.db", rag: bool = True, **overrides):
        settings = Settings(provider="scripted", embedder="hashing", db_path=str(tmp_path / db),
                            fuzz_cases=15, max_repairs=2)
        for key, value in overrides.items():
            setattr(settings, key, value)
        return Toolforge(settings, llm=brain.llm(), embedder=HashingEmbedder(),
                         registry=Registry(settings.db_path), rag=rag)

    return factory
