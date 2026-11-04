"""Turn one advisory into retrieval chunks.

Every chunk starts with a short header (id, package, severity, weakness) so a chunk
still makes sense when it is retrieved on its own, without its neighbours. The body
is split into overlapping word windows so a sentence cut at a boundary still appears
whole in one of the two chunks.
"""
import re


def header(adv: dict) -> str:
    cwes = ", ".join(c["name"] for c in adv["cwes"]) or "unspecified weakness"
    ids = adv["id"] + (f' / {adv["cve"]}' if adv.get("cve") else "")
    return (
        f'{adv["summary"]}\n'
        f'Advisory: {ids}. Packages: {", ".join(adv["packages"]) or "unknown"}. '
        f'Severity: {adv.get("severity") or "unknown"}. Weakness: {cwes}.\n'
        f'Affected: {"; ".join(adv["affected"]) or "unknown"}.'
    )


def clean(text: str) -> str:
    text = re.sub(r"```.*?```", " [code omitted] ", text, flags=re.S)  # long PoC code adds noise
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images
    return re.sub(r"\s+", " ", text).strip()


def windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    if len(words) <= size:
        return [words]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_advisory(adv: dict, size: int = 180, overlap: int = 40) -> list[dict]:
    head = header(adv)
    body = clean(adv["description"]).split()
    parts = windows(body, size, overlap) if body else [[]]
    return [
        {"advisory_id": adv["id"], "ord": i, "text": head + "\n\n" + " ".join(p)}
        for i, p in enumerate(parts)
    ]
