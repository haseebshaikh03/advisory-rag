"""Build a labelled question set from the advisories themselves, with no LLM.

Three kinds of question, each testing something different:

- id:       "What is CVE-2025-1234 about?"           exact identifier lookup
- weakness: "Is there a path traversal issue in X?"  package + weakness type
- symptom:  the advisory summary with the package name removed, so the retriever
            has to match on what the bug does rather than on a name

Each question lists every advisory that counts as a correct answer.

    python -m eval.make_questions
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
random.seed(7)


def main() -> None:
    advs = [json.loads(l) for l in (ROOT / "data" / "advisories.jsonl").open(encoding="utf-8") if l.strip()]
    by_pkg_cwe = defaultdict(set)
    for a in advs:
        for p in a["packages"]:
            for c in a["cwes"]:
                by_pkg_cwe[(p, c["id"])].add(a["id"])

    qs = []
    for a in random.sample(advs, 60):
        ident = a["cve"] or a["id"]
        qs.append({"kind": "id", "question": f"What is {ident} about?", "relevant": [a["id"]]})

    with_cwe = [a for a in advs if a["cwes"] and a["packages"]]
    for a in random.sample(with_cwe, 60):
        pkg, cwe = a["packages"][0], a["cwes"][0]
        qs.append({
            "kind": "weakness",
            "question": f"Is there a known {cwe['name'].lower()} vulnerability in {pkg}?",
            "relevant": sorted(by_pkg_cwe[(pkg, cwe["id"])]),
        })

    for a in random.sample([a for a in advs if a["packages"]], 60):
        text = a["summary"]
        for p in a["packages"]:
            text = re.sub(re.escape(p), "a package", text, flags=re.I)
        text = re.sub(r"^[^:]{1,40}:\s*", "", text)  # drop "Pkgname: " prefixes
        qs.append({"kind": "symptom", "question": f"Which advisory describes this: {text}", "relevant": [a["id"]]})

    out = ROOT / "eval" / "questions.json"
    out.write_text(json.dumps(qs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(qs)} questions to {out}")


if __name__ == "__main__":
    main()
