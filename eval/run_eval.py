"""Score each retriever on the labelled questions.

Recall@5: share of questions where a correct advisory is in the top 5.
MRR@10:   mean of 1/rank of the first correct advisory (0 if not in the top 10).
          Rewards putting the right answer first, not just somewhere in the list.

    python -m eval.run_eval
"""
import json
import time
from collections import defaultdict
from pathlib import Path

from rag.db import connect
from rag.search import search

# (name, retriever, route exact CVE/GHSA ids to a lookup first)
CONFIGS = [
    ("vector", "vector", False),
    ("keyword", "keyword", False),
    ("hybrid", "hybrid", False),
    ("hybrid+ids", "hybrid", True),
]

ROOT = Path(__file__).resolve().parent.parent


def first_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    for i, aid in enumerate(retrieved, start=1):
        if aid in relevant:
            return i
    return None


def main() -> None:
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    conn = connect()
    results = {}
    for name, mode, use_ids in CONFIGS:
        per_kind = defaultdict(lambda: {"n": 0, "recall5": 0, "rr": 0.0})
        t0 = time.time()
        for q in questions:
            ids = [h.advisory_id for h in search(conn, q["question"], mode, n=10, use_ids=use_ids)]
            r = first_rank(ids, set(q["relevant"]))
            for kind in (q["kind"], "all"):
                s = per_kind[kind]
                s["n"] += 1
                s["recall5"] += int(r is not None and r <= 5)
                s["rr"] += 1 / r if r else 0.0
        ms = (time.time() - t0) / len(questions) * 1000
        results[name] = {
            k: {"recall@5": round(v["recall5"] / v["n"], 3), "mrr@10": round(v["rr"] / v["n"], 3), "n": v["n"]}
            for k, v in per_kind.items()
        }
        results[name]["ms_per_query"] = round(ms, 1)

    kinds = ["id", "weakness", "symptom", "all"]
    print(f'{"retriever":<11} ' + " ".join(f"{k:>18}" for k in kinds) + "   ms/q")
    for mode, r in results.items():
        cells = " ".join(f'{r[k]["recall@5"]:>8.2f} / {r[k]["mrr@10"]:<7.2f}' for k in kinds)
        print(f"{mode:<11} {cells}   {r['ms_per_query']}")
    print("(each cell: recall@5 / mrr@10)")
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
