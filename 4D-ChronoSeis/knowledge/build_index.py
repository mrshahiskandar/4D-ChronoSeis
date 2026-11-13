"""Add PDFs to Gemma's reference library:  python knowledge/build_index.py /path/to/pdf_folder
Writes knowledge/<folder name>.jsonl (page-numbered passages). Needs pdftotext or the pypdf package.
Restart the app afterwards."""
import json, re, shutil, subprocess, sys
from pathlib import Path

src = Path(sys.argv[1]); out = Path(__file__).parent / f"{src.name}.jsonl"
rows = []
for pdf in sorted(src.glob("*.pdf")):
    if shutil.which("pdftotext"):
        pages = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout.split("\f")
    else:
        from pypdf import PdfReader
        pages = [p.extract_text() or "" for p in PdfReader(str(pdf)).pages]
    for pi, pg in enumerate(pages, 1):
        w = re.sub(r"\s+", " ", pg).split()
        for i in range(0, len(w), 170):
            seg = w[i:i + 220]
            if len(seg) >= 40:
                rows.append(dict(source=pdf.stem, file=pdf.name, page=pi, text=" ".join(seg)))
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"{len(rows)} passages from {src} -> {out}")
