"""Compare two saved runs with a paired bootstrap test.

    python -m bench.compare results/A.json results/B.json --metric ndcg@10

Both files must come from runs over the same query set and the same
--relevant-at, or the comparison is meaningless. This is checked, not assumed.
"""

import argparse
import json
from pathlib import Path

from bench.significance import format_report, paired_bootstrap


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "per_query" not in data:
        raise SystemExit(
            f"{path} has no per_query scores. Re-run it with the current "
            f"bench/run.py, which records them."
        )
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--resamples", type=int, default=10_000)
    args = ap.parse_args()

    a, b = load(Path(args.file_a)), load(Path(args.file_b))

    if a["relevant_at"] != b["relevant_at"]:
        raise SystemExit(
            f"relevance bars differ ({a['relevant_at']} vs {b['relevant_at']}); "
            f"these runs are not comparable"
        )

    if args.metric not in a["per_query"]:
        raise SystemExit(
            f"metric {args.metric!r} not recorded. Available: "
            f"{list(a['per_query'])}"
        )

    scores_a = {int(q): v for q, v in a["per_query"][args.metric].items()}
    scores_b = {int(q): v for q, v in b["per_query"][args.metric].items()}

    shared = set(scores_a) & set(scores_b)
    if len(shared) != len(scores_a) or len(shared) != len(scores_b):
        print(
            f"warning: query sets differ "
            f"({len(scores_a)} vs {len(scores_b)}, {len(shared)} shared); "
            f"testing on the intersection only"
        )

    r = paired_bootstrap(scores_a, scores_b, n_resamples=args.resamples)
    print(f"metric: {args.metric}  relevant_at={a['relevant_at']}\n")
    print(format_report(a["retriever"], b["retriever"], r))


if __name__ == "__main__":
    main()