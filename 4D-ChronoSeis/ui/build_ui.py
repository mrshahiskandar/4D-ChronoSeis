"""Build the single self-contained web page.

Reads ui/index.src.html, embeds the fonts in ui/fonts/ as base64 @font-face rules and three.js
(ui/vendor/three.bundle.js, MIT, see ui/vendor/THREE_LICENSE.txt),
and writes static/index.html (the file the server serves). Re-run after editing the source:

    python ui/build_ui.py
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "index.src.html"
OUT = HERE.parent / "static" / "index.html"
FONTS = [
    ("Inter", "inter-latin-wght-normal.woff2"),
    ("Inter Tight", "inter-tight-latin-wght-normal.woff2"),
]


def face(family: str, file: str) -> str:
    b64 = base64.b64encode((HERE / "fonts" / file).read_bytes()).decode()
    return (f'@font-face{{font-family:"{family}";font-style:normal;font-weight:100 900;font-display:block;'
            f'src:url(data:font/woff2;base64,{b64}) format("woff2")}}')


def main():
    html = SRC.read_text(encoding="utf-8")
    assert "/*@FONTS@*/" in html, "font placeholder missing"
    html = html.replace("/*@FONTS@*/", "\n".join(face(*x) for x in FONTS))
    # three.js (MIT) for the 3D horizon viewer, embedded so the page works without internet
    assert "<!--@THREE@-->" in html, "three.js placeholder missing"
    three = (HERE / "vendor" / "three.bundle.js").read_text(encoding="utf-8")
    assert "</script" not in three.lower()
    html = html.replace("<!--@THREE@-->", "<script>" + three + "</script>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
