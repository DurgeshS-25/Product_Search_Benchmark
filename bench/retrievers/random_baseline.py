"""Random retriever. Not a contender, a control.

Two jobs:
  1. Prove the harness runs end to end before Elasticsearch exists.
  2. Give you the floor. Any real retriever must beat this by a wide margin;
     if BM25 scores anywhere near random, your indexing is broken, not BM25.
"""

import random

from bench.dataset import Dataset


class RandomRetriever:
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.product_ids: list[int] = []

    def index(self, dataset: Dataset) -> None:
        self.product_ids = list(dataset.products.keys())

    def search(self, query: str, k: int) -> list[int]:
        if k >= len(self.product_ids):
            return list(self.product_ids)
        return self.rng.sample(self.product_ids, k)