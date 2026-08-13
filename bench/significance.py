"""Paired bootstrap significance test for two retrievers on the same queries.

Why paired: both retrievers answer the identical query set, so the per-query
differences are paired observations. Resampling queries (not scores) preserves
that pairing and asks the only question that matters -- would this ranking of the
two systems survive a different sample of queries drawn from the same population?

Why bootstrap rather than a t-test: per-query nDCG is bounded in [0, 1], skewed,
and nowhere near normal. Resampling makes no distributional assumption.

What it cannot tell you: whether 379 WANDS queries represent your traffic. A
significant result here means the difference is unlikely to be sampling noise
*within this query set*, not that it generalises to another catalogue.
"""

import random
from statistics import mean


def paired_bootstrap(
    scores_a: dict[int, float],
    scores_b: dict[int, float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Test whether A differs from B. Returns the observed difference, a 95%
    confidence interval on it, and a two-sided p-value.

    Args:
        scores_a: query_id -> per-query metric for system A
        scores_b: query_id -> per-query metric for system B (same query ids)
    """
    qids = sorted(set(scores_a) & set(scores_b))
    if len(qids) < 2:
        raise ValueError(f"need at least 2 shared queries, got {len(qids)}")

    diffs = [scores_a[q] - scores_b[q] for q in qids]
    observed = mean(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    resampled = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled.append(sum(sample) / n)
    resampled.sort()

    lo = resampled[int(0.025 * n_resamples)]
    hi = resampled[int(0.975 * n_resamples)]

    # Two-sided p-value by shifting the resampled distribution to a zero-mean
    # null and counting how often it reaches the observed magnitude.
    centred = [r - observed for r in resampled]
    extreme = sum(1 for c in centred if abs(c) >= abs(observed))
    p = (extreme + 1) / (n_resamples + 1)

    return {
        "n_queries": float(n),
        "mean_a": mean(scores_a[q] for q in qids),
        "mean_b": mean(scores_b[q] for q in qids),
        "observed_diff": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_value": p,
        "n_queries_a_better": float(sum(1 for d in diffs if d > 0)),
        "n_queries_b_better": float(sum(1 for d in diffs if d < 0)),
        "n_queries_tied": float(sum(1 for d in diffs if d == 0)),
    }


def format_report(name_a: str, name_b: str, r: dict[str, float]) -> str:
    verdict = (
        "significant at p<0.05"
        if r["p_value"] < 0.05
        else "NOT significant at p<0.05"
    )
    sign = "wins on" if r["observed_diff"] > 0 else "loses on"
    return (
        f"{name_a} vs {name_b}  (n={int(r['n_queries'])} queries)\n"
        f"  {name_a}: {r['mean_a']:.4f}   {name_b}: {r['mean_b']:.4f}\n"
        f"  difference: {r['observed_diff']:+.4f}  "
        f"95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]\n"
        f"  p={r['p_value']:.4f}  -> {verdict}\n"
        f"  per-query: {name_a} better on {int(r['n_queries_a_better'])}, "
        f"{name_b} better on {int(r['n_queries_b_better'])}, "
        f"tied on {int(r['n_queries_tied'])}\n"
        f"  ({name_a} {sign} the mean)"
    )