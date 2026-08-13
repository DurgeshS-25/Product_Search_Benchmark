"""Retrieval metrics.

Split by what they measure, because on this dataset one number cannot do both
jobs. WANDS averages ~357 relevant products per query, so:

  precision@10, ndcg@10  -> final ranking quality. What the user sees.
  recall@100             -> candidate generation quality. What a second-stage
                            reranker has to work with.

recall@10 is reported but is capped near 0.03 by arithmetic on this dataset and
should not be used as a headline number.
"""

from math import log2
from statistics import mean

# nDCG gain function. 2^g - 1 is the TREC/BEIR convention: it rewards an Exact
# (grade 2) match disproportionately over a Partial (grade 1), which is the
# right shape for product search.
def _gain(grade: int) -> float:
    return (2.0**grade) - 1.0


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top k that is relevant. Denominator is k, not the number
    retrieved, so short result lists are penalised rather than flattered."""
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / k


def ndcg_at_k(retrieved: list[int], grades: dict[int, int], k: int) -> float:
    """Graded nDCG. `grades` maps product_id -> 0/1/2 for judged products;
    unjudged products score 0, which is the standard pooled-dataset assumption
    and understates every retriever equally."""
    if not any(g > 0 for g in grades.values()):
        return float("nan")

    dcg = sum(
        _gain(grades.get(pid, 0)) / log2(rank + 2)
        for rank, pid in enumerate(retrieved[:k])
    )
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(g_val / log2(rank + 2) for rank, g_val in enumerate(_gain(g) for g in ideal))
    return dcg / idcg if idcg > 0 else float("nan")


def evaluate(
    results: dict[int, list[int]],
    relevant_by_query: dict[int, set[int]],
    grades_by_query: dict[int, dict[int, int]],
    k_precision: int = 10,
    k_recall: int = 100,
) -> dict[str, float]:
    """Macro-average over queries that have at least one relevant product.

    Returns the metrics plus n_scored and n_skipped. An average over an unknown
    denominator is not a result.
    """
    ndcg, prec, rec_k, rec_10 = [], [], [], []
    skipped = 0

    for qid, retrieved in results.items():
        relevant = relevant_by_query.get(qid, set())
        if not relevant:
            skipped += 1
            continue
        grades = grades_by_query.get(qid, {})
        ndcg.append(ndcg_at_k(retrieved, grades, k_precision))
        prec.append(precision_at_k(retrieved, relevant, k_precision))
        rec_k.append(recall_at_k(retrieved, relevant, k_recall))
        rec_10.append(recall_at_k(retrieved, relevant, 10))

    if not ndcg:
        raise ValueError("no queries had any relevant products; check qrels loading")

    return {
        f"ndcg@{k_precision}": mean(ndcg),
        f"precision@{k_precision}": mean(prec),
        f"recall@{k_recall}": mean(rec_k),
        "recall@10": mean(rec_10),
        "n_scored": float(len(ndcg)),
        "n_skipped": float(skipped),
    }