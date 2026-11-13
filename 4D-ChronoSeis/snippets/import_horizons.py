# 4D ChronoSeis app: use EXISTING horizon files instead of generating them (seeds, picking, refinement).
# Builds `processed_horizons` (marker -> inline x crossline grid of TWT, ms) and `horizon_order` in the kernel,
# exactly the variables the notebook's later stages (QC, export, RMS/NdRMS, sweetness, classes) expect.
import re, numpy as np, pandas as pd
from pathlib import Path

_HZ = Path(r"__HZ_DIR__")
for _n in ("ilines", "xlines", "trace_grid", "times"):
    if _n not in globals():
        raise NameError(f"'{_n}' is missing: run 'Load & match Baseline / Monitor' first (in this same session).")
if not _HZ.is_dir():
    raise FileNotFoundError(f"Horizon folder not found: {_HZ}")

_il = np.asarray(ilines); _xl = np.asarray(xlines)
_files = []
_summ = _HZ / "Autopick_Export_Summary.csv"
if _summ.is_file():
    _s = pd.read_csv(_summ)
    if {"Horizon", "File"} <= set(_s.columns):
        _files = [(str(h), _HZ / str(f)) for h, f in zip(_s["Horizon"], _s["File"]) if (_HZ / str(f)).is_file()]
if not _files:
    _files = [(p.stem, p) for p in sorted(_HZ.glob("*.txt"))]
if not _files:
    raise FileNotFoundError(f"No horizon .txt files (tab-separated X, Y, Inline, Crossline, TWT) in {_HZ}")

_PH, _med = {}, {}
for _name, _p in _files:
    _d = pd.read_csv(_p, sep="\t")
    _d.columns = [str(c).strip() for c in _d.columns]
    _miss = [c for c in ("Inline", "Crossline", "TWT") if c not in _d.columns]
    if _miss:
        print(f"[app] skipped {_p.name}: missing column(s) {', '.join(_miss)}"); continue
    _i = np.searchsorted(_il, _d["Inline"].to_numpy()); _j = np.searchsorted(_xl, _d["Crossline"].to_numpy())
    _ok = (_i < len(_il)) & (_j < len(_xl))
    _ok[_ok] &= (_il[_i[_ok]] == _d["Inline"].to_numpy()[_ok]) & (_xl[_j[_ok]] == _d["Crossline"].to_numpy()[_ok])
    _g = np.full(trace_grid.shape, np.nan, dtype=np.float64)
    _g[_i[_ok], _j[_ok]] = pd.to_numeric(_d["TWT"], errors="coerce").to_numpy()[_ok]
    _g[trace_grid < 0] = np.nan                      # keep only bins that exist in the matched survey
    _n = int(np.isfinite(_g).sum())
    if not _n:
        print(f"[app] skipped {_name}: no pick falls inside the matched survey grid (check inline/crossline)"); continue
    _PH[_name] = _g; _med[_name] = float(np.nanmedian(_g))
    print(f"[app] {_name}: {_n:,} picks on the survey grid ({100*_n/np.count_nonzero(trace_grid >= 0):.1f}% coverage), median TWT {_med[_name]:.0f} ms"
          + ("  [outside the seismic time range]" if not (times[0] <= _med[_name] <= times[-1]) else ""))
if not _PH:
    raise ValueError("None of the horizon files could be used.")
processed_horizons = _PH
horizon_order = sorted(_PH, key=lambda h: (_med[h], h))          # shallow to deep, as the notebook does
print(f"[app] Using {len(_PH)} existing horizon(s) from {_HZ}. Horizon generation was skipped.")
