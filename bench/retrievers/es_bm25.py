"""BM25 baseline on Elasticsearch.

Two query configs over one index:

  naive  - standard analyzer, every field weighted equally, no tuning.
           This is the strawman most "embeddings win" posts benchmark against.
  tuned  - english analyzer (stemming, stopwords), product name boosted over
           description, cross_fields so terms can match across fields.

Each field is indexed twice via multi-fields, so both configs run against a
single indexing pass. The gap between them is the measured value of lexical
tuning, and it is the number that decides whether a later "dense beats BM25"
claim is honest.
"""

import time

from elasticsearch import Elasticsearch, helpers

from bench.dataset import Dataset

INDEX_NAME = "wands_products"

TEXT_FIELDS = ("product_name", "product_class", "category_hierarchy", "product_description")

# ES's own BM25 defaults, written down instead of inherited. Phase 5 sweeps these.
BM25_K1 = 1.2
BM25_B = 0.75


def _text_field() -> dict:
    """english-analyzed primary, with a standard-analyzed .std twin and a .kw
    exact-match subfield for the Phase 4 identifier bucket."""
    return {
        "type": "text",
        "analyzer": "english",
        "similarity": "bm25_explicit",
        "fields": {
            "std": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "bm25_explicit",
            },
            "kw": {"type": "keyword", "ignore_above": 512},
        },
    }


def build_mapping() -> dict:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "similarity": {
                "bm25_explicit": {"type": "BM25", "k1": BM25_K1, "b": BM25_B}
            },
        },
        "mappings": {"properties": {f: _text_field() for f in TEXT_FIELDS}},
    }


# Boosts for the tuned config. Guesses, not gospel — Phase 5 sweeps them.
# Rationale: a product's name is the strongest relevance signal; a long
# description dilutes term frequency and is the weakest.
TUNED_BOOSTS = {
    "product_name": 3.0,
    "product_class": 2.0,
    "category_hierarchy": 1.5,
    "product_description": 1.0,
}


def build_query(query: str, variant: str) -> dict:
    """Pure function so the query shape is testable without a live cluster."""
    if variant == "naive":
        return {
            "multi_match": {
                "query": query,
                "fields": [f"{f}.std" for f in TEXT_FIELDS],
                "type": "best_fields",
            }
        }
    if variant == "tuned":
        return {
            "multi_match": {
                "query": query,
                "fields": [f"{f}^{b}" for f, b in TUNED_BOOSTS.items()],
                "type": "cross_fields",
                "operator": "or",
            }
        }
    raise ValueError(f"unknown variant: {variant!r}")


class ElasticBM25Retriever:
    def __init__(self, variant: str = "tuned", url: str = "http://localhost:9200"):
        if variant not in ("naive", "tuned"):
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant
        self.name = f"bm25-{variant}"
        self.es = Elasticsearch(url, request_timeout=60)
        self.index_stats: dict[str, float] = {}

    def index(self, dataset: Dataset) -> None:
        """Idempotent: deletes and rebuilds. Reindexing 43k docs is seconds, and
        a stale index silently invalidates every number downstream."""
        if self.es.indices.exists(index=INDEX_NAME):
            self.es.indices.delete(index=INDEX_NAME)
        self.es.indices.create(index=INDEX_NAME, **build_mapping())

        def actions():
            for pid, p in dataset.products.items():
                yield {
                    "_index": INDEX_NAME,
                    "_id": str(pid),
                    "_source": {
                        "product_name": p.name,
                        "product_class": p.product_class,
                        "category_hierarchy": p.category,
                        "product_description": p.description,
                    },
                }

        t0 = time.perf_counter()
        ok, errors = helpers.bulk(self.es, actions(), chunk_size=1000, request_timeout=120)
        self.es.indices.refresh(index=INDEX_NAME)
        # One segment, so index size is comparable across runs and against pgvector.
        self.es.indices.forcemerge(index=INDEX_NAME, max_num_segments=1)
        elapsed = time.perf_counter() - t0

        if errors:
            raise RuntimeError(f"bulk indexing errors: {errors[:3]}")

        stats = self.es.indices.stats(index=INDEX_NAME)
        size_bytes = stats["_all"]["primaries"]["store"]["size_in_bytes"]
        self.index_stats = {
            "docs_indexed": ok,
            "index_seconds": elapsed,
            "index_size_mb": size_bytes / 1024 / 1024,
        }

    def search(self, query: str, k: int) -> list[int]:
        resp = self.es.search(
            index=INDEX_NAME,
            query=build_query(query, self.variant),
            size=k,
            source=False,
            track_total_hits=False,
        )
        return [int(h["_id"]) for h in resp["hits"]["hits"]]