"""Load the WANDS product search dataset into corpus / queries / qrels.

WANDS ships three tab-separated CSVs from https://github.com/wayfair/WANDS.
Column headers are normalised on read (lowercased, spaces and dashes to
underscores) because the published schema and the actual file header do not
always agree, and pandas' itertuples silently renames any column that isn't a
valid Python identifier to a positional _N.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Graded relevance. Changing these changes every metric you report.
LABEL_GRADES = {"exact": 2, "partial": 1, "irrelevant": 0}

# Minimum grade that counts as "relevant" for binary metrics like recall.
# 1 = Partial and Exact both count. 2 = Exact only, a much stricter benchmark.
DEFAULT_RELEVANT_AT = 1


@dataclass
class Product:
    product_id: int
    name: str
    product_class: str
    category: str
    description: str

    def search_text(self) -> str:
        """The single text blob we index. Field weighting happens in ES."""
        parts = [self.name, self.product_class, self.category, self.description]
        return " ".join(p for p in parts if p)


@dataclass
class Dataset:
    products: dict[int, Product]
    queries: dict[int, str]
    # qrels[query_id][product_id] = grade (0, 1 or 2)
    qrels: dict[int, dict[int, int]]
    # Populated by load_wands(): data-quality counters worth printing.
    load_notes: dict[str, int] = field(default_factory=dict)

    def relevant_ids(self, query_id: int, min_grade: int = DEFAULT_RELEVANT_AT) -> set[int]:
        judged = self.qrels.get(query_id, {})
        return {pid for pid, grade in judged.items() if grade >= min_grade}

    def grades(self, query_id: int) -> dict[int, int]:
        """product_id -> grade for every judged product. Needed for nDCG."""
        return self.qrels.get(query_id, {})

    def summary(self) -> str:
        judged = sum(len(v) for v in self.qrels.values())
        rel_counts = [len(self.relevant_ids(q)) for q in self.queries]
        rel_counts = [c for c in rel_counts if c > 0]
        avg_rel = sum(rel_counts) / len(rel_counts) if rel_counts else 0
        return (
            f"products={len(self.products)} queries={len(self.queries)} "
            f"judgments={judged} queries_with_relevant={len(rel_counts)} "
            f"avg_relevant_per_query={avg_rel:.1f}"
        )


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_wands.sh first.")
    df = pd.read_csv(path, sep="\t")
    df.columns = [_normalise(c) for c in df.columns]
    return df


def _require(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name}: missing column(s) {missing}. "
            f"Columns actually present: {list(df.columns)}"
        )


def load_wands(data_dir: str | Path = "data/WANDS/dataset") -> Dataset:
    d = Path(data_dir)

    prod_path = d / "product.csv"
    prod_df = _read(prod_path).fillna("")
    _require(
        prod_df,
        ["product_id", "product_name", "product_class", "category_hierarchy",
         "product_description"],
        prod_path,
    )
    products = {
        int(pid): Product(
            product_id=int(pid),
            name=str(name),
            product_class=str(pclass),
            category=str(cat),
            description=str(desc),
        )
        for pid, name, pclass, cat, desc in zip(
            prod_df["product_id"],
            prod_df["product_name"],
            prod_df["product_class"],
            prod_df["category_hierarchy"],
            prod_df["product_description"],
        )
    }

    query_path = d / "query.csv"
    query_df = _read(query_path)
    _require(query_df, ["query_id", "query"], query_path)
    queries = {
        int(qid): str(q) for qid, q in zip(query_df["query_id"], query_df["query"])
    }

    label_path = d / "label.csv"
    label_df = _read(label_path)
    _require(label_df, ["query_id", "product_id", "label"], label_path)

    qrels: dict[int, dict[int, int]] = {}
    unknown: set[str] = set()
    dupes_agreeing = 0
    dupes_conflicting = 0

    for qid, pid, label in zip(
        label_df["query_id"], label_df["product_id"], label_df["label"]
    ):
        key = _normalise(label)
        if key not in LABEL_GRADES:
            unknown.add(str(label))
            continue
        qid, pid = int(qid), int(pid)
        grade = LABEL_GRADES[key]
        per_query = qrels.setdefault(qid, {})
        if pid in per_query:
            # The same product judged twice for the same query. Agreeing dupes
            # are harmless; conflicting ones mean ground truth is ambiguous and
            # last-write-wins would silently pick one. Keep the higher grade so
            # the choice is explicit and consistent.
            if per_query[pid] == grade:
                dupes_agreeing += 1
            else:
                dupes_conflicting += 1
                per_query[pid] = max(per_query[pid], grade)
            continue
        per_query[pid] = grade

    if unknown:
        # Fail loudly rather than scoring against a partial ground truth.
        raise ValueError(f"Unrecognised relevance labels in label.csv: {sorted(unknown)}")

    dataset = Dataset(products=products, queries=queries, qrels=qrels)
    dataset.load_notes = {
        "label_rows": int(len(label_df)),
        "unique_pairs": sum(len(v) for v in qrels.values()),
        "duplicate_pairs_agreeing": dupes_agreeing,
        "duplicate_pairs_conflicting": dupes_conflicting,
    }
    return dataset