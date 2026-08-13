# Product Search Benchmark: BM25 vs. Dense vs. Hybrid

Does semantic search replace keyword search for product search? Measure it
instead of assuming.

Retrieval architectures compared over 42,994 real product listings, scored
against human relevance judgments, reported per query type rather than as a
single average.

**Status: Phase 1 complete.** BM25 baseline established. Dense retrieval not yet
built, so no lexical-vs-semantic claim is made here yet.

---

## Result so far: lexical tuning is worth ~21%

Two BM25 configurations over the same index:

- **naive** — standard analyzer, all four fields weighted equally. This is the
  strawman that "embeddings beat keyword search" posts usually benchmark against.
- **tuned** — english analyzer (stemming, stopwords), `cross_fields`, product
  name boosted over description.

| Retriever | relevance bar | nDCG@10 | Precision@10 | Recall@100 |
|---|---|---|---|---|
| random | Partial+Exact | 0.0058 | 0.0092 | 0.0028 |
| bm25-naive | Partial+Exact | 0.5825 | 0.6795 | 0.3288 |
| **bm25-tuned** | Partial+Exact | **0.7062** | **0.8196** | **0.4022** |
| random | Exact only | 0.0049 | 0.0024 | 0.0033 |
| bm25-naive | Exact only | 0.5932 | 0.3784 | 0.5836 |
| **bm25-tuned** | Exact only | **0.7149** | **0.4631** | **0.6570** |

Tuning gain, naive → tuned:

| Metric | Partial+Exact | Exact only |
|---|---|---|
| nDCG@10 | +21.2% | +20.5% |
| Precision@10 | +20.6% | +22.4% |
| Recall@100 | +22.3% | +12.6% |

The gain holds at both relevance bars. Analyzer choice and field boosting are
often described as legacy overhead; on this dataset they are worth about a fifth
of ranking quality, and any later comparison against an untuned BM25 baseline
would overstate the alternative by roughly that much.

**Cost of the BM25 index:** 42,994 documents in 3.49s, 36.28 MB on disk, query
p50 3.1ms / p95 4.5ms. This is the figure Phase 2's embedding job is measured
against.

Raw output for every run is in [`results/`](results/).

---

## Data

[WANDS](https://github.com/wayfair/WANDS) (Wayfair Annotation Dataset, MIT
licensed): 42,994 products, 480 queries, 233,448 human judgments labeled
Exact / Partial / Irrelevant.

Loaded as: 231,873 unique query-product pairs, plus 1,559 duplicate pairs where
annotators agreed and 16 where they conflicted. Conflicts are resolved by taking
the higher grade. At 16 out of 233,448 this is immaterial, but the resolution is
explicit rather than last-write-wins.

## Metrics, and why these ones

WANDS averages **357.3 relevant products per query** at the Partial+Exact bar —
0.83% of the corpus. That single fact rules out the metric this project
originally planned to use:

- **recall@10 is not viable as a headline.** Per-query recall@10 is bounded by
  10/|relevant|, so a perfect retriever scores in the low single digits on most
  queries. It is reported but should not be quoted.
- **nDCG@10 and Precision@10** measure final ranking quality — what a shopper
  actually sees on page one.
- **Recall@100** measures candidate-generation quality — what a second-stage
  reranker would have to work with.

That split is not bookkeeping. It maps onto the two-stage architecture question
in Phase 3.5: a first stage is judged on recall@100, a reranker on nDCG@10.

nDCG uses graded gains (`2^grade - 1`, the TREC convention), so an Exact match is
worth 3 and a Partial 1.

### Two relevance bars, both reported

`--relevant-at 1` counts Partial and Exact as relevant; `--relevant-at 2` counts
Exact only. Exact-only is the primary configuration, because on a storefront a
partial match is a bad result, and because it leaves far more headroom above the
baseline: 0.54 rather than 0.18 on Precision@10. A benchmark whose baseline
already scores 0.82 cannot discriminate between what comes next.

## Known limitations

These affect how the results should be read and are not resolved.

- **Exact-only excludes 101 of 480 queries** — precisely those with no exact
  match, which skew vague and conceptual. That is the query population where
  dense retrieval is most likely to win, so the primary configuration may
  systematically disadvantage embeddings. Reporting both bars is a partial
  mitigation; Phase 4's stratification is the real one.
- **nDCG does not respond to `--relevant-at`.** It reads graded labels directly.
  The small nDCG difference between the two bars is a query-subset effect, not a
  relevance-definition effect.
- **Judgments are pooled.** Unjudged pairs score 0. This understates every
  retriever, but not necessarily equally.
- **The tuned boosts (3.0 / 2.0 / 1.5 / 1.0) were reasoned, not measured.** BM25
  `k1=1.2, b=0.75` are Elasticsearch defaults, written explicitly into the
  mapping rather than inherited. Phase 5 sweeps both with a held-out split.
- **Latency is localhost HTTP against a single-node cluster.** Valid for
  comparing retrievers to each other; not a production latency claim.
- **480 queries is small once split into buckets.** Phase 4 differences need the
  significance test in Phase 5, not eyeballing.

## Phases

| Phase | Adds | Status |
|---|---|---|
| 0 | WANDS loader, graded metrics, random control | Complete |
| 1 | Docker Compose, Elasticsearch BM25, naive vs. tuned | **Complete** |
| 2 | Bi-encoder embeddings, pgvector + HNSW | Next |
| 3 | Reciprocal rank fusion | Planned |
| 3.5 | Cross-encoder rerank over BM25 top-100 | Planned |
| 4 | Query stratification: exact-identifier / keyword / conceptual | Planned |
| 5 | Parameter sweeps, reindex cost, bootstrap significance | Planned |

Phase 3.5 exists because a bi-encoder plus HNSW is the 2020 answer. Cross-encoder
reranking over a lexical candidate set is closer to what production search
actually does, and it tests whether a vector database is needed at all.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_wands.sh
docker compose up -d          # wait for both containers to report healthy

python -m bench.run --retriever random
python -m bench.run --retriever bm25 --variant tuned
python -m bench.run --retriever bm25 --variant naive --skip-index
python -m bench.run --retriever bm25 --variant tuned --relevant-at 2 --skip-index
```

`--skip-index` reuses the live Elasticsearch index; use it only when the corpus
is unchanged. Every run writes a JSON record to `results/`.

Sanity checks worth keeping: `docs_indexed` must equal 42,994, and the random
control must land near 10/42,994 for recall@10. Random Precision@10 came out
0.0092 against a predicted 357.3/42,994 = 0.0083.

## Design note

`bench/run.py` and `bench/retrievers/base.py` are meant to stop changing. Each
later phase adds one class implementing the `Retriever` protocol and registers it
in `build_retriever()`. The measurement loop stays fixed so results stay
comparable across phases.