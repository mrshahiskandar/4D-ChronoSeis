"""Arbitrary vertical section A-A' through the Baseline / Monitor SEG-Y, with horizons, well paths, markers and logs.

Traces are taken at a regular step along the line (the nearest Baseline trace to each point, by X/Y); Monitor
traces are paired with Baseline traces exactly as for the time-shift map (inline/crossline, else nearest X/Y).
Only the traces on the line are read. Logs come from each nearby well's LAS file, converted MD -> TWT with the
same path the app draws on the maps (deviation + checkshot).
"""
from __future__ import annotations

import base64
import re

import numpy as np

import slices
import wellio

_TREE: dict = {}
_PAIR: dict = {}

LOG_GROUPS = [
    ("GR", "Gamma ray", ("GR", "GRC", "SGR", "CGR", "GR_EDTC", "HGR")),
    ("SW", "Water saturation", ("SW", "SWE", "SWT", "SWA", "SUWI", "SWT_1", "SW_AR")),
    ("RT", "Resistivity", ("RT", "ILD", "LLD", "RDEP", "RD", "RES", "AT90", "HDRS", "LLD_1")),
    ("PHI", "Porosity", ("PHIE", "PHIT", "NPHI", "PHI", "TNPH")),
    ("AI", "Acoustic impedance", ("AI", "PIMP", "ZP")),
]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def _tree(v):
    k = str(v.path)
    if k not in _TREE:
        from scipy.spatial import cKDTree
        xy = np.column_stack([v.x, v.y])
        t = cKDTree(xy)
        s = xy[:: max(1, len(xy) // 4000)]
        sp = float(np.median(t.query(s, k=2)[0][:, 1])) if len(s) > 2 else 25.0
        _TREE[k] = (t, max(sp, 1.0))
    return _TREE[k]


def _pairmap(vb, vm):
    k = (str(vb.path), str(vm.path))
    if k not in _PAIR:
        ib, im, how = slices._pair(vb, vm)
        m = np.full(vb.ntr, -1, np.int64)
        m[ib] = im
        _PAIR[k] = (m, how)
    return _PAIR[k]


def line_points(a, b, spacing, n_max):
    L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if L < spacing:
        raise ValueError("A and A' are closer than one trace spacing; pick points further apart.")
    n = int(min(n_max, max(2, L / spacing + 1)))
    s = np.linspace(0.0, L, n)
    u = np.array([(b[0] - a[0]) / L, (b[1] - a[1]) / L])
    return s, a[0] + s * u[0], a[1] + s * u[1], u, L


def section(base, mon, vol, a, b, cache_dir=None, t0=None, t1=None, n_max=500, wells=None, horizons=None, corridor=None):
    """a, b = (x, y). vol: 'diff' (Monitor - Baseline), 'baseline', 'monitor'. wells: well_paths() records. horizons:
    dict(x0, y0, cell, nx, ny, grids=[(name, 2D TWT array)])."""
    if vol not in ("diff", "baseline", "monitor"):
        raise ValueError("vol must be diff, baseline or monitor")
    if not base or (vol != "baseline" and not mon):
        raise FileNotFoundError("Choose the Baseline and Monitor SEG-Y first.")
    slices.note("Reading trace headers…")
    vb = slices.volume(base, cache_dir)
    vm = slices.volume(mon, cache_dir) if mon and vol != "baseline" else None
    if vm is not None:
        slices._align_coords([vm], vb)
    tree, sp = _tree(vb)
    s, xs, ys, u, L = line_points(a, b, sp, n_max)
    d, idx = tree.query(np.column_stack([xs, ys]))
    ok = d <= 0.75 * sp * 1.42
    i0 = vb.index_of(t0 if t0 is not None else vb.t0)
    i1 = vb.index_of(t1 if t1 is not None else vb.t0 + (vb.ns - 1) * vb.dt)
    nw = i1 - i0 + 1
    if nw < 4:
        raise ValueError("Time window too short.")
    out = np.full((len(s), nw), np.nan, np.float32)
    slices.note(f"Reading {int(ok.sum())} traces along the line…")
    rows_b = idx[ok]
    ub, inv = np.unique(rows_b, return_inverse=True)
    B = vb.block(ub, i0, nw)[inv] if ub.size else np.zeros((0, nw), np.float32)
    if vol == "baseline":
        V = B
    else:
        pm, how = _pairmap(vb, vm)
        rm = pm[rows_b]
        good = rm >= 0
        M = np.full_like(B, np.nan)
        if good.any():
            um, inv2 = np.unique(rm[good], return_inverse=True)
            M[good] = vm.block(um, i0, nw)[inv2]
        V = M if vol == "monitor" else M - B
    out[ok] = V
    fin = out[np.isfinite(out)]
    clip = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0
    t_start = vb.t0 + i0 * vb.dt
    res = dict(vol=vol, a=list(map(float, a)), b=list(map(float, b)), length=L, spacing=sp, ntr=len(s), ns=nw, t0=t_start, dt=vb.dt,
               dist=np.round(s, 1).tolist(), x=np.round(xs, 1).tolist(), y=np.round(ys, 1).tolist(),
               values=_b64(out), clip=clip or 1.0, wells=[], horizons=[],
               title={"diff": "Monitor − Baseline", "baseline": "Baseline", "monitor": "Monitor"}[vol])
    # horizons along the line
    if horizons:
        g = horizons
        ix = np.clip(((xs - g["x0"]) / g["cell"]).astype(int), 0, g["nx"] - 1)
        iy = np.clip(((ys - g["y0"]) / g["cell"]).astype(int), 0, g["ny"] - 1)
        for name, Z in g["grids"]:
            z = Z[iy, ix]
            if np.isfinite(z).sum() >= 2:
                res["horizons"].append(dict(name=name, twt=[None if not np.isfinite(q) else round(float(q), 1) for q in z]))
    # wells near the line, projected; logs in TWT
    cor = corridor or max(250.0, 3 * sp)
    for w in wells or []:
        p = w.get("path")
        if not p:
            continue
        px, py, pt = (np.asarray(p[k], float) for k in ("x", "y", "twt"))
        rel = np.column_stack([px - a[0], py - a[1]])
        along, off = rel @ u, rel @ np.array([-u[1], u[0]])
        near = (np.abs(off) <= cor) & (along >= -cor) & (along <= L + cor)
        if not near.any():
            continue
        rec = dict(name=w["name"], offset=round(float(np.min(np.abs(off[near]))), 1), path=dict(s=np.round(along[near], 1).tolist(), twt=np.round(pt[near], 1).tolist()),
                   markers=[], logs=[])
        for m in w.get("markers", []):
            if m.get("twt") is None or m.get("x") is None:
                continue
            r = np.array([m["x"] - a[0], m["y"] - a[1]])
            if abs(float(r @ np.array([-u[1], u[0]]))) <= cor:
                rec["markers"].append(dict(name=m["name"], s=round(float(r @ u), 1), twt=m["twt"]))
        rec["logs"] = well_logs(w, t_start, t_start + (nw - 1) * vb.dt)
        rec["fluid"] = w.get("fluid", [])
        rec["s_mid"] = float(np.median(along[near]))
        res["wells"].append(rec)
    res["wells"].sort(key=lambda r: r["s_mid"])
    return res


def well_logs(w, t0, t1, max_pts=600):
    """Up to three standard curves of a well, resampled onto TWT (ms) using its MD->TWT path."""
    lp, p = w.get("las_path"), w.get("path")
    if not lp or not p or not p.get("md"):
        return []
    try:
        las = wellio.read_las(lp)
    except Exception:
        return []
    cur = las["curves"]
    if not cur:
        return []
    idx_name = next(iter(cur))
    md = np.asarray(cur[idx_name][1], float)
    if str(las.get("index_unit", "")).lower().startswith(("f", "ft")):
        md = md * 0.3048
    pmd, ptwt = np.asarray(p["md"], float), np.asarray(p["twt"], float)
    okp = np.isfinite(pmd) & np.isfinite(ptwt)
    if okp.sum() < 2:
        return []
    twt = np.interp(md, pmd[okp], ptwt[okp], left=np.nan, right=np.nan)
    out = []
    names = {k.upper(): k for k in cur}
    for k in list(cur):                      # GR_P, RT_P, SWE_1 ... also answer to their base name
        names.setdefault(re.sub(r"(_[A-Z0-9]{1,6})+$", "", k.upper().split(".")[0]), k)
    for gid, label, cands in LOG_GROUPS:
        key = next((names[c] for c in cands if c in names), None)
        if key is None:
            continue
        unit, v = cur[key]
        v = np.asarray(v, float)
        m = np.isfinite(v) & np.isfinite(twt) & (twt >= t0) & (twt <= t1)
        if m.sum() < 5:
            continue
        sel = np.flatnonzero(m)
        sel = sel[:: max(1, len(sel) // max_pts)]
        vv = v[sel]
        lo, hi = np.percentile(vv, [1, 99])
        if gid == "SW" and hi > 1.5:
            vv, lo, hi = vv / 100.0, lo / 100.0, hi / 100.0
        out.append(dict(id=gid, label=label, mnem=key, unit=unit, twt=np.round(twt[sel], 1).tolist(), v=np.round(vv, 4).tolist(),
                        lo=float(lo), hi=float(hi), log=gid == "RT"))
        if len(out) >= 3:
            break
    return out
