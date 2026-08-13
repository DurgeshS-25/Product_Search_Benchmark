"""Phase 4: split queries by type and report each retriever per bucket.

Aggregate scores hide the thing that matters. Hybrid RRF beats tuned BM25 by
+3.5% nDCG@10 on average while making 111 of 379 queries worse. This module asks
whether those regressions form a recognisable type.

Buckets are derived, not hand-labelled, so the split is reproducible. For each
query we measure LEXICAL COVERAGE: the fraction of the query's content words that
literally appear in the text of its relevant products.

  coverage near 1.0  the shopper's words are present in the right products.
                     Inverted-index matching should win here.
  coverage near 0.0  the shopper's words do not appear in the correct products.
                     Only meaning-based matching can bridge that gap.

Queries are then split into terciles by coverage. Tercile boundaries are reported
so the cut is inspectable rather than asserted.

Deliberately crude tokenisation: lowercase, split on non-alphanumerics, drop a
small stopword list, strip a trailing "s". No stemmer, because a stemmer would
import the very normalisation that BM25's english analyzer performs, and the
point of this measure is to describe the query/document relationship
independently of any retriever's configuration.

    python -m bench.stratify --relevant-at 2 \\
        results/bm25-tuned_rel2.json \\
        results/dense-qwen3-full_rel2_ef500.json \\
        "results/hybrid-rrf(bm25-tuned+dense-qwen3-full)_rel2_ef500.json"
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

from bench.dataset import load_wands
from bench.significance import paired_bootstrap

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "with", "w",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Cap on relevant products inspected per query. Coverage converges quickly and
# some queries have hundreds of judged-relevant products.
MAX_RELEVANT_SAMPLED = 100


def tokenise(text: str) -> set[str]:
    out = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        # Crude plural handling; keeps "chairs"/"chair" from looking unrelated.
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.add(tok)
    return out


def lexical_coverage(query: str, product_texts: list[str]) -> float:
    """Mean fraction of query tokens present in each relevant product's text."""
    q = tokenise(query)
    if not q or not product_texts:
        return float("nan")
    per_product = []
    for text in product_texts:
        p = tokenise(text)
        per_product.append(len(q & p) / len(q))
    return mean(per_product)


def assign_terciles(coverage: dict[int, float]) -> tuple[dict[int, str], tuple[float, float]]:
    """Split query ids into three equal-sized buckets by coverage."""
    ordered = sorted(coverage.items(), key=lambda kv: kv[1])
    n = len(ordered)
    lo_cut = ordered[n // 3][1]
    hi_cut = ordered[2 * n // 3][1]

    buckets: dict[int, str] = {}
    for i, (qid, _) in enumerate(ordered):
        if i < n // 3:
            buckets[qid] = "low_overlap"
        elif i < 2 * n // 3:
            buckets[qid] = "mid_overlap"
        else:
            buckets[qid] = "high_overlap"
    return buckets, (lo_cut, hi_cut)


def load_run(path: Path, metric: str) -> tuple[str, int, dict[int, float]]:
    data = json.loads(path.read_text())
    if "per_query" not in data:
        raise SystemExit(f"{path}: no per_query scores; re-run with current bench/run.py")
    if metric not in data["per_query"]:
        raise SystemExit(f"{path}: metric {metric!r} not recorded")
    scores = {int(q): v for q, v in data["per_query"][metric].items()}
    return data["retriever"], data["relevant_at"], scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result JSON files to compare")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--relevant-at", type=int, default=2)
    ap.add_argument("--data-dir", default="data/WANDS/dataset")
    ap.add_argument("--baseline", default="bm25-tuned",
                    help="retriever name to measure regressions against")
    ap.add_argument("--challenger", default=None,
                    help="retriever name to test against the baseline "
                         "(default: the last result file given)")
    ap.add_argument("--out", default="results/stratified.json")
    args = ap.parse_args()

    runs = [load_run(Path(p), args.metric) for p in args.results]
    for name, rel_at, _ in runs:
        if rel_at != args.relevant_at:
            raise SystemExit(
                f"{name} was run at relevant_at={rel_at}, expected {args.relevant_at}"
            )

    dataset = load_wands(args.data_dir)

    # -- coverage per query -------------------------------------------------
    scored_qids = set(runs[0][2])
    for _, _, s in runs[1:]:
        scored_qids &= set(s)

    coverage: dict[int, float] = {}
    for qid in sorted(scored_qids):
        relevant = sorted(dataset.relevant_ids(qid, args.relevant_at))[:MAX_RELEVANT_SAMPLED]
        texts = [dataset.products[pid].search_text() for pid in relevant
                 if pid in dataset.products]
        cov = lexical_coverage(dataset.queries[qid], texts)
        if cov == cov:  # not nan
            coverage[qid] = cov

    buckets, (lo_cut, hi_cut) = assign_terciles(coverage)
    bucket_names = ["low_overlap", "mid_overlap", "high_overlap"]

    print(f"metric={args.metric}  relevant_at={args.relevant_at}  "
          f"queries={len(coverage)}")
    print(f"lexical coverage: min={min(coverage.values()):.3f} "
          f"median={median(coverage.values()):.3f} "
          f"max={max(coverage.values()):.3f}")
    print(f"tercile cuts at coverage {lo_cut:.3f} and {hi_cut:.3f}\n")

    # -- per-bucket scores --------------------------------------------------
    name_w = max(len(n) for n, _, _ in runs) + 2
    header = "retriever".ljust(name_w) + "".join(b.rjust(16) for b in bucket_names)
    print(header)
    print("-" * len(header))

    per_bucket: dict[str, dict[str, float]] = {}
    for name, _, scores in runs:
        row = name.ljust(name_w)
        per_bucket[name] = {}
        for b in bucket_names:
            vals = [scores[q] for q in coverage if buckets[q] == b and q in scores]
            m = mean(vals) if vals else float("nan")
            per_bucket[name][b] = m
            row += f"{m:.4f}".rjust(16)
        print(row)

    counts = {b: sum(1 for q in coverage if buckets[q] == b) for b in bucket_names}
    print("n".ljust(name_w) + "".join(str(counts[b]).rjust(16) for b in bucket_names))

    # -- per-bucket significance, baseline vs challenger -------------------
    names = [n for n, _, _ in runs]
    challenger = args.challenger or names[-1]
    if args.baseline in names and challenger in names and args.baseline != challenger:
        base_scores = dict(runs[names.index(args.baseline)][2])
        chal_scores = dict(runs[names.index(challenger)][2])

        print(f"\n{challenger}\n  vs {args.baseline}, per bucket:")
        sig: dict[str, dict] = {}
        for b in bucket_names:
            qs = [q for q in coverage if buckets[q] == b]
            a = {q: chal_scores[q] for q in qs if q in chal_scores}
            c = {q: base_scores[q] for q in qs if q in base_scores}
            r = paired_bootstrap(a, c, n_resamples=10_000)
            sig[b] = r
            verdict = "significant" if r["p_value"] < 0.05 else "not significant"
            print(
                f"  {b:<14} diff={r['observed_diff']:+.4f} "
                f"CI[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}] "
                f"p={r['p_value']:.4f} ({verdict})  "
                f"W/L/T {int(r['n_queries_a_better'])}/"
                f"{int(r['n_queries_b_better'])}/{int(r['n_queries_tied'])}"
            )

        # -- regression analysis ------------------------------------------
        regressed = [q for q in coverage
                     if q in chal_scores and q in base_scores
                     and chal_scores[q] < base_scores[q]]
        improved = [q for q in coverage
                    if q in chal_scores and q in base_scores
                    and chal_scores[q] > base_scores[q]]

        print(f"\nWhere {challenger} loses to {args.baseline}:")
        print(f"  regressed on {len(regressed)} queries, improved on {len(improved)}")
        if regressed and improved:
            print(f"  mean coverage of regressed queries: "
                  f"{mean(coverage[q] for q in regressed):.3f}")
            print(f"  mean coverage of improved queries:  "
                  f"{mean(coverage[q] for q in improved):.3f}")
            for b in bucket_names:
                nr = sum(1 for q in regressed if buckets[q] == b)
                ni = sum(1 for q in improved if buckets[q] == b)
                print(f"  {b:<14} regressed={nr:<4} improved={ni}")

        payload_sig = {
            b: {k: round(v, 6) for k, v in r.items()} for b, r in sig.items()
        }
    else:
        payload_sig = {}
        regressed = improved = []

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "metric": args.metric,
        "relevant_at": args.relevant_at,
        "tercile_cuts": {"low_mid": lo_cut, "mid_high": hi_cut},
        "bucket_counts": counts,
        "per_bucket_scores": {
            n: {b: round(v, 6) for b, v in d.items()} for n, d in per_bucket.items()
        },
        "per_bucket_significance": payload_sig,
        "coverage_by_query": {str(q): round(c, 6) for q, c in coverage.items()},
        "bucket_by_query": {str(q): b for q, b in buckets.items()},
        "regressed_queries": sorted(regressed),
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()