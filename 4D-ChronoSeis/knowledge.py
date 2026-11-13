"""Small offline search over the user's reference papers (no internet, no extra packages).

Passages live in knowledge/*.jsonl, one JSON object per line: {"source", "file", "page", "text"}.
Build or extend them from PDFs with:  python knowledge/build_index.py <folder with PDFs>
Gemma gets the best passages as background and cites them as [source, p. N]."""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

DIR = Path(os.environ.get("APP4D_KNOWLEDGE", Path(__file__).parent / "knowledge"))
_STOP = set("""a an the of and or to in on for with by from as at is are was were be been this that these those it its into
than then there their which who what when where how why can could would should may might will we our you your they them not no
also using used use based such more most other some any each between both over under about via per all""".split())
_IDX = None


def _tok(s: str):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", s.lower()) if w not in _STOP and len(w) > 1]


def _load():
    global _IDX
    if _IDX is not None:
        return _IDX
    docs = []
    for f in sorted(DIR.glob("*.jsonl")) if DIR.is_dir() else []:
        for line in f.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("text", "")
            if t and t.count(". .") < 15 and t.count("...") < 15:   # skip tables of contents
                docs.append(d)
    toks = [Counter(_tok(d["text"] + " " + d.get("source", ""))) for d in docs]
    df = Counter(w for t in toks for w in t)
    n = len(docs) or 1
    avg = sum(sum(t.values()) for t in toks) / n
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    _IDX = dict(docs=docs, toks=toks, idf=idf, avg=avg or 1.0)
    return _IDX


def search(query: str, k: int = 4, min_score: float = 0.0):
    """BM25 ranking; at most two passages per paper so answers draw on several sources."""
    ix = _load()
    q = _tok(query)
    if not q or not ix["docs"]:
        return []
    scores = []
    for i, t in enumerate(ix["toks"]):
        L = sum(t.values())
        s = 0.0
        for w in q:
            f = t.get(w)
            if f:
                s += ix["idf"][w] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * L / ix["avg"]))
        if s > min_score:
            scores.append((s, i))
    scores.sort(reverse=True)
    out, per = [], Counter()
    for s, i in scores:
        d = ix["docs"][i]
        if per[d["source"]] >= 2:
            continue
        per[d["source"]] += 1
        out.append(dict(source=d["source"], page=d.get("page"), score=round(s, 1), text=d["text"][:1400]))
        if len(out) >= k:
            break
    return out


def sources():
    ix = _load()
    return sorted(Counter(d["source"] for d in ix["docs"]).items())
