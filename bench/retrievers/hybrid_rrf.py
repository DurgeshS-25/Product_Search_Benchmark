"""Hybrid retrieval by reciprocal rank fusion, with an overlap diagnostic.

RRF discards the scores and keeps only the positions:

    score(doc) = sum over lists of 1 / (K + rank(doc))

This is deliberate. BM25 scores are unbounded and cosine similarities live in
[-1, 1]; adding them lets the BM25 magnitude drown the other signal. Ranks are
the only commensurable thing the two retrievers produce.

K = 60 comes from the original RRF paper (Cormack et al., 2009). It flattens the
curve so rank 1 does not dominate everything beneath it. Exposed as a parameter
rather than buried, because it is a tuning knob and should be visible as one.

The diagnostic matters more than the fused score. Fusion can only help if the two
retrievers find *different* correct products. If dense surfaces the same items
BM25 already found, only ordered worse, fusion adds noise. This class records the
overlap so that question is answered with a number instead of a hope.
"""

from bench.dataset import Dataset

RRF_K = 60
# How deep each retriever's list goes before fusing. Deeper costs nothing here
# and gives fusion more to work with.
FUSION_DEPTH = 100


def rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse ranked lists of product_ids. Best first, in and out."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda pid: scores[pid], reverse=True)


class HybridRRFRetriever:
    """Wraps two already-built retrievers. Does not index anything itself."""

    def __init__(self, lexical, dense, rrf_k: int = RRF_K, depth: int = FUSION_DEPTH):
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self.depth = depth
        self.name = f"hybrid-rrf({lexical.name}+{dense.name})"
        self.index_stats: dict[str, float] = {}
        # Diagnostic accumulators, filled during search().
        self._overlap_counts: list[int] = []
        self._last_lists: dict[int, tuple[list[int], list[int]]] = {}
        self._query_seq = 0

    def index(self, dataset: Dataset) -> None:
        """Both sub-retrievers are expected to be indexed already. Reindexing
        here would re-embed 43k documents on every hybrid run."""
        self.index_stats = {}

    def search(self, query: str, k: int) -> list[int]:
        lex = self.lexical.search(query, self.depth)
        den = self.dense.search(query, self.depth)

        self._query_seq += 1
        self._last_lists[self._query_seq] = (lex, den)
        self._overlap_counts.append(len(set(lex) & set(den)))

        return rrf_fuse([lex, den], self.rrf_k)[:k]

    # -- diagnostic -----------------------------------------------------

    def overlap_report(
        self, relevant_by_query: dict[int, set[int]], query_ids: list[int]
    ) -> dict[str, float]:
        """How redundant are the two retrievers, and does either find correct
        products the other misses entirely?"""
        if not self._last_lists:
            return {}

        overlaps, lex_only_rel, den_only_rel, both_rel = [], [], [], []

        for seq, qid in enumerate(query_ids, start=1):
            pair = self._last_lists.get(seq)
            if pair is None:
                continue
            lex, den = pair
            lex_set, den_set = set(lex), set(den)
            relevant = relevant_by_query.get(qid, set())
            if not relevant:
                continue

            overlaps.append(len(lex_set & den_set) / max(len(lex_set | den_set), 1))
            lex_only_rel.append(len((lex_set - den_set) & relevant))
            den_only_rel.append(len((den_set - lex_set) & relevant))
            both_rel.append(len(lex_set & den_set & relevant))

        n = len(overlaps)
        if n == 0:
            return {}

        return {
            "jaccard_overlap_top100": sum(overlaps) / n,
            "relevant_found_by_both": sum(both_rel) / n,
            "relevant_only_lexical": sum(lex_only_rel) / n,
            "relevant_only_dense": sum(den_only_rel) / n,
        }