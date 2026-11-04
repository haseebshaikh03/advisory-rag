"""Download reviewed PyPI security advisories from the GitHub Advisory Database.

Uses the public REST endpoint GET /advisories through the GitHub CLI, so it needs
`gh auth login` once. Writes a normalized data/advisories.jsonl that the rest of
the pipeline reads.

    python -m rag.fetch --pages 3
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def normalize(raw: dict) -> dict:
    packages = sorted({v["package"]["name"] for v in raw.get("vulnerabilities") or [] if v.get("package")})
    ranges = [
        f'{v["package"]["name"]} {v.get("vulnerable_version_range") or "?"} (fixed in {v.get("first_patched_version") or "no fix yet"})'
        for v in raw.get("vulnerabilities") or []
        if v.get("package")
    ]
    return {
        "id": raw["ghsa_id"],
        "cve": raw.get("cve_id"),
        "summary": raw.get("summary") or "",
        "description": raw.get("description") or "",
        "severity": raw.get("severity"),
        "packages": packages,
        "affected": ranges,
        "cwes": [{"id": c["cwe_id"], "name": c["name"]} for c in raw.get("cwes") or []],
        "published": raw.get("published_at"),
        "url": raw.get("html_url"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="100 advisories per page")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    out = []
    url = "/advisories?ecosystem=pip&type=reviewed&per_page=100"
    for _ in range(args.pages):
        # This endpoint pages with a cursor in the Link header, not with ?page=N,
        # so ask gh for the headers (-i) and follow rel="next".
        res = subprocess.run(["gh", "api", "-i", url], capture_output=True, text=True, encoding="utf-8", check=True)
        head, _, body = res.stdout.partition("\r\n\r\n") if "\r\n\r\n" in res.stdout else res.stdout.partition("\n\n")
        out.extend(normalize(a) for a in json.loads(body))
        nxt = re.search(r'<https://api\.github\.com(/advisories[^>]+)>;\s*rel="next"', head)
        if not nxt:
            break
        url = nxt.group(1)
    # New advisories shift the pages while we read them, so the same one can show up twice.
    out = list({a["id"]: a for a in out}.values())
    path = DATA / "advisories.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for a in out:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} advisories to {path}")


if __name__ == "__main__":
    main()
