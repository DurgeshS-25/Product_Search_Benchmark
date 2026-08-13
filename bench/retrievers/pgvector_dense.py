"""Dense retrieval: sentence-transformer embeddings in Postgres/pgvector, HNSW.

Two models, both run on the same corpus and the same text config so the only
variable is the model:

  minilm  all-MiniLM-L6-v2         384 dims. Not a serious contender in 2026;
                                   included because it is the default in most
                                   tutorials, so "what people actually ship"
                                   deserves a measured number.
  qwen3   Qwen3-Embedding-0.6B    1024 dims. Apache 2.0, laptop-runnable,
                                   current generation.

Two text configurations, because what you embed matters as much as which model
you use:

  short  name + product class + category
  full   short + product_description

`full` is the fair comparison against tuned BM25, which indexes all four fields.
`short` tests whether the long description helps or dilutes the vector. Docs
exceeding the model's token limit are counted and reported rather than silently
truncated.

Some retrieval models expect an instruction prefix on the query but not on the
document. Getting this wrong degrades results silently, so prefixes are declared
per model below and printed on every run. Verify them against the model card
before trusting any number.
"""

import time

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from bench.dataset import Dataset

MODELS = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dims": 384,
        "query_prefix": "",
        "doc_prefix": "",
        "batch_size": 256,
    },
    "qwen3": {
        "hf_name": "Qwen/Qwen3-Embedding-0.6B",
        "dims": 1024,
        # Qwen3-Embedding expects an instruction on the query side only.
        # Official template from the model card:
        #   f"Instruct: {task_description}\nQuery:{query}"   (no space after the colon)
        # The task description is domain-adapted; the card requires a one-sentence
        # instruction on the query and none on the documents.
        "query_prefix": (
            "Instruct: Given a shopping query, retrieve relevant product listings\nQuery:"
        ),
        "doc_prefix": "",
        "batch_size": 32,
    },
}

# HNSW build parameters. Fixed across models so the comparison is fair.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


TEXT_CONFIGS = ("short", "full")


def product_text(dataset: Dataset, product_id: int, text_config: str) -> str:
    p = dataset.products[product_id]
    parts = [p.name, p.product_class, p.category]
    if text_config == "full":
        parts.append(p.description)
    return " ".join(x for x in parts if x)


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PgVectorDenseRetriever:
    def __init__(
        self,
        model_key: str = "qwen3",
        text_config: str = "full",
        dsn: str = "postgresql://bench:bench@localhost:5432/bench",
        ef_search: int = 100,
        query_prefix: str | None = None,
    ):
        if model_key not in MODELS:
            raise ValueError(f"unknown model: {model_key!r}. Options: {list(MODELS)}")
        if text_config not in TEXT_CONFIGS:
            raise ValueError(f"unknown text config: {text_config!r}")
        self.cfg = dict(MODELS[model_key])
        # Allow overriding the query instruction from the CLI so its effect can
        # be measured instead of assumed.
        if query_prefix is not None:
            self.cfg["query_prefix"] = query_prefix
        self.model_key = model_key
        self.text_config = text_config
        self.name = f"dense-{model_key}-{text_config}"
        self.table = f"emb_{model_key}_{text_config}"
        self.dsn = dsn
        self.ef_search = ef_search
        self.index_stats: dict[str, float] = {}
        self._model: SentenceTransformer | None = None
        self._conn: psycopg.Connection | None = None

    # -- lazy resources -------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            device = _pick_device()
            print(f"  loading {self.cfg['hf_name']} on {device}")
            self._model = SentenceTransformer(self.cfg["hf_name"], device=device)
            print(f"  max_seq_length={self._model.max_seq_length}")
            print(f"  query_prefix={self.cfg['query_prefix']!r}")
        return self._model

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    # -- indexing -------------------------------------------------------

    def index(self, dataset: Dataset) -> None:
        dims = self.cfg["dims"]
        ids = list(dataset.products.keys())
        texts = [
            self.cfg["doc_prefix"] + product_text(dataset, pid, self.text_config)
            for pid in ids
        ]

        # Count what the model will silently cut off.
        limit = self.model.max_seq_length
        tok = self.model.tokenizer
        lengths = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
        n_truncated = sum(1 for n in lengths if n > limit)
        print(
            f"  text_config={self.text_config} max_seq_length={limit} "
            f"tokens_p50={sorted(lengths)[len(lengths)//2]} "
            f"tokens_max={max(lengths)} docs_truncated={n_truncated}"
        )

        # Fail early on a dimension mismatch rather than after a 20-minute
        # embedding job.
        probe = self.model.encode(
            texts[:1], normalize_embeddings=True, show_progress_bar=False
        )
        if probe.shape[1] != dims:
            raise ValueError(
                f"{self.model_key}: model returned {probe.shape[1]} dims, "
                f"MODELS says {dims}. Fix the registry before indexing."
            )

        t0 = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size=self.cfg["batch_size"],
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embed_secs = time.perf_counter() - t0

        self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.conn.execute(
            f"CREATE TABLE {self.table} "
            f"(product_id integer PRIMARY KEY, embedding vector({dims}))"
        )

        t1 = time.perf_counter()
        with self.conn.cursor() as cur:
            with cur.copy(
                f"COPY {self.table} (product_id, embedding) FROM STDIN"
            ) as copy:
                for pid, vec in zip(ids, vectors):
                    copy.write_row((pid, vec))
        load_secs = time.perf_counter() - t1

        t2 = time.perf_counter()
        self.conn.execute(
            f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
        hnsw_secs = time.perf_counter() - t2

        size_bytes = self.conn.execute(
            "SELECT pg_total_relation_size(%s)", (self.table,)
        ).fetchone()[0]
        count = self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

        self.index_stats = {
            "docs_indexed": count,
            "docs_truncated": n_truncated,
            "embed_seconds": embed_secs,
            "db_load_seconds": load_secs,
            "hnsw_build_seconds": hnsw_secs,
            "index_seconds": embed_secs + load_secs + hnsw_secs,
            "index_size_mb": size_bytes / 1024 / 1024,
        }

    # -- search ---------------------------------------------------------

    def search(self, query: str, k: int) -> list[int]:
        vec = self.model.encode(
            [self.cfg["query_prefix"] + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        # ef_search trades recall against latency. Set per-session, swept later.
        self.conn.execute(f"SET hnsw.ef_search = {self.ef_search}")
        rows = self.conn.execute(
            f"SELECT product_id FROM {self.table} "
            f"ORDER BY embedding <=> %s LIMIT %s",
            (vec, k),
        ).fetchall()
        return [r[0] for r in rows]