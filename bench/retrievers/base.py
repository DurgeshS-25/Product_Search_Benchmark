"""The one interface BM25, dense and hybrid retrievers all implement.

Getting this right in Phase 0 is why Phases 1-3 stay cheap: run.py never
changes, only the class you pass it does.
"""

from typing import Protocol

from bench.dataset import Dataset


class Retriever(Protocol):
    name: str

    def index(self, dataset: Dataset) -> None:
        """Build whatever this retriever needs. Called once."""
        ...

    def search(self, query: str, k: int) -> list[int]:
        """Return up to k product_ids, best first."""
        ...