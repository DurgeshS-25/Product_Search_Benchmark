#!/usr/bin/env bash
# Re-run the three retrievers at the Exact-only bar, then the paired
# significance tests between them.
#
# All runs reuse the live Elasticsearch index and pgvector tables, so nothing is
# re-indexed and nothing is re-embedded. Expect roughly 1-2 minutes total.
#
# set -e so a failed run stops the script instead of feeding stale result files
# into the comparisons.
set -euo pipefail

HYBRID="results/hybrid-rrf(bm25-tuned+dense-qwen3-full)_rel2_ef500.json"
BM25="results/bm25-tuned_rel2.json"
DENSE="results/dense-qwen3-full_rel2_ef500.json"

echo "=============================================="
echo " 1/3  BM25 tuned"
echo "=============================================="
python -m bench.run --retriever bm25 --variant tuned \
  --relevant-at 2 --skip-index

echo
echo "=============================================="
echo " 2/3  Dense qwen3 full text"
echo "=============================================="
python -m bench.run --retriever dense --model qwen3 --text full \
  --ef-search 500 --relevant-at 2 --skip-index

echo
echo "=============================================="
echo " 3/3  Hybrid RRF"
echo "=============================================="
python -m bench.run --retriever hybrid --variant tuned --model qwen3 \
  --text full --ef-search 500 --relevant-at 2

echo
echo "=============================================="
echo " SIGNIFICANCE TESTS"
echo "=============================================="

echo
echo "--- Does hybrid actually beat BM25? (nDCG@10) ---"
python -m bench.compare "$HYBRID" "$BM25" --metric ndcg@10

echo
echo "--- Does hybrid actually beat BM25? (precision@10) ---"
python -m bench.compare "$HYBRID" "$BM25" --metric precision@10

echo
echo "--- Does BM25 actually beat dense? (nDCG@10) ---"
python -m bench.compare "$BM25" "$DENSE" --metric ndcg@10

echo
echo "--- Does hybrid beat dense? (nDCG@10) ---"
python -m bench.compare "$HYBRID" "$DENSE" --metric ndcg@10

echo
echo "Done. Result files are in results/"