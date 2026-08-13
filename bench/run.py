"""Benchmark runner.

    python -m bench.run --retriever random
    python -m bench.run --retriever bm25 --variant naive
    python -m bench.run --retriever bm25 --variant tuned

The measurement loop is meant to stop changing. Later phases add a retriever
class and a line in build_retriever().
"""

import argparse
import json
import time
from pathlib import Path

from bench.dataset import DEFAULT_RELEVANT_AT, load_wands
from bench.metrics import evaluate

K_VALUES = (10, 100)


def build_retriever(args):
    if args.retriever == "random":
        from bench.retrievers.random_baseline import RandomRetriever

        return RandomRetriever()
    if args.retriever == "bm25":
        from bench.retrievers.es_bm25 import ElasticBM25Retriever

        return ElasticBM25Retriever(variant=args.variant, url=args.es_url)
    # Phase 2: "dense" -> PgVectorDenseRetriever
    # Phase 3: "hybrid" -> RRFRetriever
    raise ValueError(args.retriever)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True, choices=["random", "bm25"])
    ap.add_argument("--variant", default="tuned", choices=["naive", "tuned"],
                    help="BM25 query config")
    ap.add_argument("--data-dir", default="data/WANDS/dataset")
    ap.add_argument("--es-url", default="http://localhost:9200")
    ap.add_argument("--relevant-at", type=int, default=DEFAULT_RELEVANT_AT,
                    help="1 = Partial+Exact count as relevant, 2 = Exact only")
    ap.add_argument("--skip-index", action="store_true",
                    help="Query an already-built index. Use only when the corpus is unchanged.")
    args = ap.parse_args()

    dataset = load_wands(args.data_dir)
    print(dataset.summary())
    for key, val in dataset.load_notes.items():
        print(f"  {key}={val}")

    retriever = build_retriever(args)

    if args.skip_index:
        print("skipping index build")
    else:
        retriever.index(dataset)
        for key, val in getattr(retriever, "index_stats", {}).items():
            print(f"  {key}={val:.2f}" if isinstance(val, float) else f"  {key}={val}")

    top_k = max(K_VALUES)
    results: dict[int, list[int]] = {}
    latencies: list[float] = []
    for qid, query in dataset.queries.items():
        t = time.perf_counter()
        results[qid] = retriever.search(query, top_k)
        latencies.append((time.perf_counter() - t) * 1000)

    relevant_by_query = {
        qid: dataset.relevant_ids(qid, args.relevant_at) for qid in dataset.queries
    }
    grades_by_query = {qid: dataset.grades(qid) for qid in dataset.queries}

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    print(f"\nretriever={retriever.name} relevant_at={args.relevant_at}")
    # Localhost HTTP, warm-ish cache. Relative comparison between retrievers
    # only. Not a production latency claim.
    print(f"query latency p50={p50:.1f}ms p95={p95:.1f}ms")

    scores = evaluate(results, relevant_by_query, grades_by_query)
    n_scored = int(scores.pop("n_scored"))
    n_skipped = int(scores.pop("n_skipped"))
    print("HEADLINE  (ranking quality, what the user sees)")
    for m in ("ndcg@10", "precision@10"):
        print(f"  {m}={scores[m]:.4f}")
    print("CANDIDATES (first-stage quality, what a reranker gets)")
    print(f"  recall@100={scores['recall@100']:.4f}")
    # Not a headline metric: WANDS averages ~357 relevant products per query,
    # so per-query recall@10 is bounded by 10/|rel_q|. The macro-average is not
    # bounded by 10/mean(|rel_q|) because |rel_q| is heavily skewed.
    print(f"  recall@10={scores['recall@10']:.4f}  (low ceiling, not a headline)")
    print(f"n_scored={n_scored} n_skipped={n_skipped}")

    # Persist every run. The committed results are the evidence behind any claim
    # made about this benchmark; code alone is not a finding.
    record = {
        "retriever": retriever.name,
        "relevant_at": args.relevant_at,
        "dataset": dataset.load_notes | {
            "products": len(dataset.products),
            "queries": len(dataset.queries),
        },
        "index_stats": getattr(retriever, "index_stats", {}),
        "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2)},
        "metrics": {k: round(v, 4) for k, v in scores.items()},
        "n_scored": n_scored,
        "n_skipped": n_skipped,
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{retriever.name}_rel{args.relevant_at}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()