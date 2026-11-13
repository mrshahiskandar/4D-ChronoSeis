"""
Read the files written by the team's 4D notebook and turn them into the app's Summary and
Detail Analysis views. Nothing here recomputes the seismic analysis: it only reads, aggregates
and maps what the notebook produced.

Workspace layout (as written by the notebook, with /home/apgce_team_08 redirected to the workspace):
  Autopick_Horizon/<horizon>.txt                                   X Y Inline Crossline TWT
  Autopick_Horizon/4D_RMS_NRMS_3/<horizon>_4D_RMS_NRMS.txt          + RMS_Baseline RMS_Monitor dRMS NdRMS_pct NRMS
  Autopick_Horizon/4D_RMS_NRMS_3/Maps/<horizon>/*.png
  Autopick_Horizon/4D_RMS_NRMS_3/4D_Platform_Constrained_Maps/Constrained/<horizon>_4D_Classes.csv
  Autopick_Horizon/Sweetness/<horizon>_Sweetness.txt
  4D_difference.sgy, 4D_Matched_Inline_Crossline.csv

Assumption (prototype): seismic X/Y, well-head X/Y and platform X/Y are in the same projected
coordinate system and units (metres). This is not checked.
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

CLASS_LABELS = {-1: "No data", 0: "Unswept", 1: "Hardening", 2: "Softening"}
# Thresholds used by the notebook's class cell (for "significant change" statistics on raw files)
NDRMS_T, NRMS_T = 15.0, 15.0


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.hz = self.root / "Autopick_Horizon"
        self.rms = self.hz / "4D_RMS_NRMS_3"
        self.maps = self.rms / "Maps"
        self.cls = self.rms / "4D_Platform_Constrained_Maps" / "Constrained"
        self.cls_un = self.rms / "4D_Platform_Constrained_Maps" / "Unconstrained"
        self.sweet = self.hz / "Sweetness"

    # ------------------------------------------------------------------ discovery
    def horizons(self) -> list[str]:
        names = {p.name[: -len("_4D_RMS_NRMS.txt")] for p in self.rms.glob("*_4D_RMS_NRMS.txt")}
        return sorted(names, key=_natural)

    def files(self, name: str) -> dict:
        f = dict(rms=self.rms / f"{name}_4D_RMS_NRMS.txt", classes=self.cls / f"{name}_4D_Classes.csv",
                 sweet=self.sweet / f"{name}_Sweetness.txt", horizon=self.hz / f"{name}.txt")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)      # the notebook's file-name rule for class exports
        if not f["classes"].is_file():
            f["classes"] = self.cls / f"{safe}_4D_Classes.csv"
        if not f["sweet"].is_file():
            f["sweet"] = self.sweet / f"{safe}_Sweetness.txt"
        out = {k: v for k, v in f.items() if v.is_file()}
        r = _rfi_match(self, name)
        if r is not None:
            out["rfi"] = r
        return out

    def status(self) -> dict:
        return dict(root=str(self.root), exists=self.root.is_dir(), horizons=len(self.horizons()),
                    classes=len(list(self.cls.glob("*_4D_Classes.csv"))) if self.cls.is_dir() else 0,
                    sweetness=len(list(self.sweet.glob("*_Sweetness.txt"))) if self.sweet.is_dir() else 0,
                    difference=(self.root / "4D_difference.sgy").is_file())

    def figures(self) -> list[dict]:
        out = []
        for base, label in [(self.maps, "Attribute map"), (self.cls, "4D classes"), (self.cls_un, "4D classes (unconstrained)")]:
            if base.is_dir():
                for p in sorted(base.rglob("*.png"), key=lambda p: _natural(str(p))):
                    rel = p.relative_to(self.root)
                    out.append(dict(path=str(rel), title=p.stem.replace("_", " "), group=label,
                                    horizon=p.parent.name if p.parent != base else _hz_from_name(p.stem)))
        return out


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _hz_from_name(stem):
    return re.sub(r"_4D_.*$", "", stem)


@lru_cache(maxsize=24)
def _read(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, sep="\t" if p.suffix == ".txt" else ",", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read(path: Path) -> pd.DataFrame:
    return _read(str(path), path.stat().st_mtime)


# ====================================================================== wells & platforms
def read_platforms(path) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    xk = next((c for c in df.columns if c.lower().replace(" ", "") in ("x(m)", "x", "easting")), None)
    yk = next((c for c in df.columns if c.lower().replace(" ", "") in ("y(m)", "y", "northing")), None)
    nk = next((c for c in df.columns if c not in (xk, yk)), None)
    if xk is None or yk is None:
        return []
    return [dict(name=str(r[nk]) if nk else f"Platform {i + 1}", x=float(r[xk]), y=float(r[yk]))
            for i, r in df.iterrows() if pd.notna(r[xk]) and pd.notna(r[yk])]


def read_wells(folder, well_list=None, overrides=None) -> list[dict]:
    """Well heads: X/Y from the LAS headers (as the notebook's overlay helper does), else from the well list.
    Searches sub-folders (las/, checkshots/ ...) too."""
    import wellpath
    return wellpath.heads(folder, well_list, overrides)


# ====================================================================== per-horizon statistics
def horizon_stats(ws: Workspace) -> list[dict]:
    rows = []
    for h in ws.horizons():
        f = ws.files(h)
        d = read(f["rms"])
        nd, nr = d["NdRMS_pct"].to_numpy(float), d["NRMS"].to_numpy(float)
        ok = np.isfinite(nd) & np.isfinite(nr)
        sig = ok & (np.abs(nd) >= NDRMS_T) & (nr >= NRMS_T)
        r = dict(horizon=h, points=int(ok.sum()),
                 twt_median=float(np.nanmedian(d["TWT"])) if "TWT" in d else None,
                 nrms_median=float(np.nanmedian(nr[ok])) if ok.any() else None,
                 ndrms_mean=float(np.nanmean(nd[ok])) if ok.any() else None,
                 sig_pct=float(100 * sig.sum() / max(ok.sum(), 1)),
                 dimming_pct=float(100 * (sig & (nd > 0)).sum() / max(ok.sum(), 1)),
                 brightening_pct=float(100 * (sig & (nd < 0)).sum() / max(ok.sum(), 1)))
        if "classes" in f:
            c = read(f["classes"])
            code = c["4D_Class_Code"].to_numpy(int)
            near = c["Within_Platform_Radius"].astype(str).str.lower().isin(["true", "1"]).to_numpy() \
                if "Within_Platform_Radius" in c else np.ones(len(c), bool)
            valid = code >= 0
            denom = max(int((valid & near).sum()), 1)
            for k, lab in ((0, "unswept"), (1, "hardening"), (2, "softening")):
                r[f"{lab}_pct"] = float(100 * ((code == k) & near).sum() / denom)
            r["classes"] = True
        else:
            r["classes"] = False
        if "sweet" in f:
            sw = read(f["sweet"])
            if "Sweetness_Difference" in sw:
                r["sweet_change_mean"] = float(np.nanmean(sw["Sweetness_Difference"]))
        rows.append(r)
    return rows


# ====================================================================== possible untapped areas
def _grid(df):
    il, xl = df["Inline"].to_numpy(int), df["Crossline"].to_numpy(int)
    uil, uxl = np.unique(il), np.unique(xl)
    return uil, uxl, np.searchsorted(uil, il), np.searchsorted(uxl, xl)


def untapped(ws: Workspace, wells: list[dict], platforms: list[dict], max_n=8, min_bins=40, min_brightness=1.3) -> list[dict]:
    """Unswept (no significant change) + bright on Baseline (reservoir proxy) + next to changed areas,
    within the platform radius. Uses the notebook's own 4D classes; a screening aid, not reserves."""
    cands = []
    for h in ws.horizons():
        f = ws.files(h)
        if "classes" not in f:
            continue
        d = read(f["classes"])
        if not {"Inline", "Crossline", "X", "Y", "RMS_Baseline", "4D_Class_Code"} <= set(d.columns):
            continue
        uil, uxl, ri, ci = _grid(d)
        shape = (len(uil), len(uxl))
        code = np.full(shape, -1, np.int8); code[ri, ci] = d["4D_Class_Code"].to_numpy(int)
        rb = np.full(shape, np.nan); rb[ri, ci] = d["RMS_Baseline"].to_numpy(float)
        X = np.full(shape, np.nan); X[ri, ci] = d["X"].to_numpy(float)
        Y = np.full(shape, np.nan); Y[ri, ci] = d["Y"].to_numpy(float)
        near = np.ones(shape, bool)
        if "Within_Platform_Radius" in d:
            near = np.zeros(shape, bool)
            near[ri, ci] = d["Within_Platform_Radius"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        changed = (code == 1) | (code == 2)
        valid = np.isfinite(rb) & (code >= 0)
        if valid.sum() < 100 or not changed.any():
            continue
        med = float(np.nanmedian(rb[valid]))
        bright = valid & (rb >= np.nanpercentile(rb[valid], 70))
        cand = (code == 0) & bright & near
        cand = ndi.binary_opening(cand, structure=np.ones((3, 3)))
        lab, n = ndi.label(cand, structure=np.ones((3, 3)))
        if n == 0:
            continue
        dist = ndi.distance_transform_edt(~changed)
        R = max(5.0, 0.05 * min(shape))
        sizes = ndi.sum(np.ones(shape), lab, index=np.arange(1, n + 1))
        bin_area = _bin_area(X, Y)
        for k in np.flatnonzero(sizes >= min_bins) + 1:
            m = lab == k
            dmin = float(dist[m].min())
            if dmin > R:
                continue
            br = float(np.nanmean(rb[m]) / med) if med else 0.0
            if br < min_brightness:
                continue
            cx, cy = float(np.nanmean(X[m])), float(np.nanmean(Y[m]))
            u = dict(horizon=h, x=cx, y=cy, inline=float(np.mean(uil[np.where(m)[0]])), xline=float(np.mean(uxl[np.where(m)[1]])),
                     bins=int(m.sum()), area_km2=float(m.sum() * bin_area / 1e6) if bin_area else None,
                     brightness=br, bins_from_change=dmin)
            ev = ["no significant 4D change (notebook class: unswept)", f"Baseline RMS {br:.1f}× the horizon median",
                  f"{dmin:.0f} bins from a hardening/softening area"]
            if platforms:
                dp = min(((math.hypot(cx - p["x"], cy - p["y"]), p["name"]) for p in platforms))
                u["nearest_platform"], u["platform_km"] = dp[1], dp[0] / 1000
                ev.append(f"{dp[0] / 1000:.1f} km from {dp[1]}")
            if wells:
                dw = min(((math.hypot(cx - w["x"], cy - w["y"]), w["name"]) for w in wells))
                u["nearest_well"], u["well_km"] = dw[1], dw[0] / 1000
                ev.append(f"{dw[0] / 1000:.1f} km from {dw[1]}")
            u["evidence"] = ev
            u["score"] = u["bins"] * br / (1 + dmin / R)
            cands.append(u)
    cands.sort(key=lambda u: -u["score"])
    out = cands[:max_n]
    for i, u in enumerate(out, 1):
        u["rank"] = i
        u["confidence"] = "High" if (u["brightness"] >= 1.6 and u["bins_from_change"] <= 2) else "Medium" if u["brightness"] >= 1.4 else "Low"
    return out


def _bin_area(X, Y):
    """Approximate area of one bin (m²) from neighbouring X/Y spacing."""
    try:
        dxi = np.nanmedian(np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)))
        dxj = np.nanmedian(np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)))
        a = float(dxi * dxj)
        return a if np.isfinite(a) and a > 0 else None
    except Exception:
        return None


# ====================================================================== summary
# ====================================================================== Phase 4 / Phase 5 well results
TEAM_ROOT = Path(os.environ.get("APP4D_TEAM_ROOT", "/home/apgce_team_08/link-to-apgce_team_08"))
P5 = "Phase5_Well_Correlation_Output"
FLUID_NAME = {1: "Water", 2: "Oil", 3: "Gas"}


def _wnum(name) -> str:
    m = re.search(r"(\d+)", str(name))
    return m.group(1) if m else str(name)


def _p45_file(ws: Workspace, rel: str):
    """This project's own Phase 4/5 run first (<workspace>/team_root), then the team's existing results (read only)."""
    for label, base in (("this project", ws.root / "team_root"), ("team folder", TEAM_ROOT)):
        p = base / rel
        if p.is_file():
            return p, label
    return None, None


def phase45(ws: Workspace) -> dict:
    """Fluid intervals (Phase 4 classes -> Phase 5 intervals paired with the nearest marker) and proposed markers, in TWT."""
    out = dict(found=False)
    p, src = _p45_file(ws, f"{P5}/Phase5_Fluid_Intervals_Nearest_Markers_All.csv")
    if p is not None:
        d = pd.read_csv(p)
        d["Num"] = d["Well"].map(_wnum)
        wells = []
        for num, g in d.groupby("Num", sort=True):
            rec = dict(name=f"Well-{num}", n_intervals=int(len(g)))
            for k, nm in FLUID_NAME.items():
                gg = g[g["Fluid_Class"] == k]
                rec[nm.lower()] = dict(n=int(len(gg)), twt_ms=round(float(gg["Interval_TWT_ms"].sum()), 1))
            wells.append(rec)
        hc = d[d["Fluid_Class"].isin([2, 3])].sort_values("Interval_TWT_ms", ascending=False).head(12)
        keep = ["Well", "Fluid_Type", "Top_TWT_ms", "Base_TWT_ms", "Interval_TWT_ms", "Nearest_Marker", "Delta_TWT_ms"]
        out.update(found=True, source=src, file=str(p), wells=wells,
                   intervals=[{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k in keep}
                              for r in hc.fillna("").to_dict("records")])
    p, src = _p45_file(ws, f"{P5}/Phase5_Proposed_Markers_X_Y_TWT_All_7_Wells.csv")
    if p is not None:
        m = pd.read_csv(p)
        tw = [c for c in m.columns if c.endswith(" TWT")]
        out["markers"] = dict(source=src, n_markers=int(len(m)), per_well={c[:-4]: int(m[c].notna().sum()) for c in tw})
        out["found"] = True
    return out


def fluid_for(ws: Workspace, name) -> list[dict]:
    """Top/base TWT of each water/oil/gas interval of one well (for the section's fluid track)."""
    p, _ = _p45_file(ws, f"{P5}/Phase5_Fluid_Intervals_Nearest_Markers_All.csv")
    if p is None:
        return []
    d = _read(str(p), p.stat().st_mtime)
    g = d[d["Well"].map(_wnum) == _wnum(name)]
    return [dict(top=float(r.Top_TWT_ms), base=float(r.Base_TWT_ms), fluid=int(r.Fluid_Class)) for r in g.itertuples()]


def _fluid_text(rec) -> str:
    bits = [f"{nm.lower()} {rec[nm.lower()]['n']} interval{'s' if rec[nm.lower()]['n'] != 1 else ''} ({rec[nm.lower()]['twt_ms']:.0f} ms)"
            for nm in ("Oil", "Gas", "Water") if rec[nm.lower()]["n"]]
    return f"{rec['name']}: " + (", ".join(bits) if bits else "no classified intervals")



# ====================================================================== Phase 6: Remaining Fluid Index (RFI)
RFI_DIR_DEFAULT = Path(os.environ.get("APP4D_RFI_DIR", "/data/epic20/APGCE/phase6_3D/RFI"))


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def rfi_dir(ws: Workspace):
    """Own run first (<workspace>/phase6_3D/RFI), then the team's Phase 6 Part 3D results (read only)."""
    for label, p in (("this project", ws.root / "phase6_3D" / "RFI"), ("team folder", RFI_DIR_DEFAULT)):
        try:
            if p.is_dir() and any(p.glob("*_RFI.csv")):
                return p, label
        except OSError:
            pass
    return None, None


def _rfi_files(ws: Workspace) -> dict:
    p, _ = rfi_dir(ws)
    return {} if p is None else {_norm(f.name[: -len("_RFI.csv")]): f for f in p.glob("*_RFI.csv")}


def _rfi_match(ws: Workspace, name: str):
    files = _rfi_files(ws)
    k = _norm(name)
    if k in files:
        return files[k]
    def pre(a, b):  # a starts with b, and the next character is not a digit (H1 must not match H10)
        return a.startswith(b) and (len(a) == len(b) or not a[len(b)].isdigit())
    hit = [f for kk, f in files.items() if kk and (pre(kk, k) or pre(k, kk))]
    return hit[0] if len(hit) == 1 else None


def rfi(ws: Workspace, wells=None, platforms=None, cell_m=200.0, thr=40.0, max_n=8) -> dict:
    """Per-horizon RFI statistics and ranked hotspots (connected cells with mean RFI >= thr). The RFI is the team's
    relative screening score (Phase 6 Part 3D): not a saturation or reserve estimate."""
    p, src = rfi_dir(ws)
    if p is None:
        return dict(found=False)
    from scipy import ndimage as ndi
    hz, spots = [], []
    for f in sorted(p.glob("*_RFI.csv"), key=lambda q: _natural(q.name)):
        name = f.name[: -len("_RFI.csv")]
        cols = ["X", "Y", "TWT", "RFI_Final", "Verified_Structural_Closure"]
        try:
            d = pd.read_csv(f, usecols=lambda c: c in cols)
        except Exception:
            continue
        if not {"X", "Y", "RFI_Final"} <= set(d.columns):
            continue
        v = d["RFI_Final"].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(d["X"]) & np.isfinite(d["Y"])
        if ok.sum() < 10:
            continue
        x, y, v = d["X"].to_numpy(float)[ok], d["Y"].to_numpy(float)[ok], v[ok]
        t = d["TWT"].to_numpy(float)[ok] if "TWT" in d else np.full(len(v), np.nan)
        clo = d["Verified_Structural_Closure"].to_numpy(bool)[ok] if "Verified_Structural_Closure" in d else np.zeros(len(v), bool)
        hz.append(dict(horizon=name, n=int(ok.sum()), median=round(float(np.median(v)), 1), p90=round(float(np.percentile(v, 90)), 1),
                       max=round(float(v.max()), 1), closure_pct=round(100 * float(clo.mean()), 1),
                       high_pct=round(100 * float((v >= thr).mean()), 2)))
        x0, y0 = x.min(), y.min()
        ix, iy = ((x - x0) / cell_m).astype(int), ((y - y0) / cell_m).astype(int)
        nx, ny = ix.max() + 1, iy.max() + 1
        flat = iy * nx + ix
        c = np.bincount(flat, minlength=nx * ny)
        mean = np.where(c > 0, np.bincount(flat, weights=v, minlength=nx * ny) / np.maximum(c, 1), np.nan).reshape(ny, nx)
        lab, n = ndi.label(np.nan_to_num(mean, nan=0) >= thr)
        for k in range(1, n + 1):
            m = lab == k
            sel = m.ravel()[flat]
            if sel.sum() < 20:
                continue
            area = float(m.sum() * cell_m * cell_m / 1e6)
            cx, cy = float(x[sel].mean()), float(y[sel].mean())
            u = dict(horizon=name, x=cx, y=cy, area_km2=round(area, 2), mean_rfi=round(float(v[sel].mean()), 1),
                     max_rfi=round(float(v[sel].max()), 1), twt_ms=None if not np.isfinite(t[sel]).any() else round(float(np.nanmedian(t[sel])), 0),
                     in_closure_pct=round(100 * float(clo[sel].mean()), 0))
            u["score"] = area * u["mean_rfi"]
            ev = [f"RFI mean {u['mean_rfi']} (max {u['max_rfi']}) over {u['area_km2']} km²"]
            if wells:
                dw = min(((math.hypot(cx - w["x"], cy - w["y"]), w["name"]) for w in wells))
                u["nearest_well"], u["well_km"] = dw[1], round(dw[0] / 1000, 2)
                ev.append(f"{u['well_km']} km from {dw[1]}")
            if platforms:
                dp = min(((math.hypot(cx - q["x"], cy - q["y"]), q["name"]) for q in platforms))
                u["nearest_platform"], u["platform_km"] = dp[1], round(dp[0] / 1000, 2)
                ev.append(f"{u['platform_km']} km from {dp[1]}")
            u["evidence"] = ev
            spots.append(u)
    if not hz:
        return dict(found=False)
    spots.sort(key=lambda u: -u["score"])
    spots = spots[:max_n]
    for i, u in enumerate(spots, 1):
        u["rank"] = i
    return dict(found=True, source=src, folder=str(p), threshold=thr, horizons=hz, hotspots=spots,
                note="RFI is a relative screening score (0-100, scored per horizon), not a saturation or reserve estimate; "
                     "outside identified structural highs it is capped low, and a strong 4D response near oil wells lowers it.")


def _summary(ws: Workspace, wells, platforms) -> dict:
    stats = horizon_stats(ws)
    if not stats:
        return dict(ready=False, status=ws.status())
    has_cls = any(s["classes"] for s in stats)
    key = (lambda s: s.get("hardening_pct", 0) + s.get("softening_pct", 0)) if has_cls else (lambda s: s["sig_pct"])
    ranked = sorted(stats, key=key, reverse=True)
    changed = [s for s in stats if key(s) >= 1.0]
    bg = [s["nrms_median"] for s in stats if s["nrms_median"] is not None]
    bg_nrms = float(np.median(bg)) if bg else None
    verdict = None if bg_nrms is None else "good" if bg_nrms < 20 else "moderate" if bg_nrms < 40 else "poor"
    un = untapped(ws, wells, platforms)
    p45 = phase45(ws)
    if p45.get("wells"):
        byw = {_wnum(w["name"]): w for w in p45["wells"]}
        for u in un:
            w = byw.get(_wnum(u.get("nearest_well", "")))
            if w:
                u["well_fluid"] = _fluid_text(w)
                u["evidence"].append(f"Nearest well, classified in TWT (Phase 4/5): {u['well_fluid']}")
    try:
        rf = rfi(ws, wells, platforms)
    except Exception:
        rf = dict(found=False)
    return dict(ready=True, wells_fluid=p45, rfi=rf if rf.get('found') else None, status=ws.status(), horizons=stats, ranked=[s["horizon"] for s in ranked],
                changed_horizons=[s["horizon"] for s in changed], classes=has_cls,
                background_nrms=bg_nrms, quality=verdict, untapped=un, wells=wells, platforms=platforms,
                thresholds=dict(ndrms_pct=NDRMS_T, nrms_pct=NRMS_T, platform_radius_m=3000.0))  # 4D ChronoSeis default PLATFORM_RADIUS_M


# ====================================================================== XY raster for the map viewer
ATTRS = {
    "class": ("4D class", "", "classes", "4D_Class_Code"),
    "NdRMS_pct": ("NdRMS (Baseline − Monitor, normalised)", "%", "rms", "NdRMS_pct"),
    "NRMS": ("NRMS", "%", "rms", "NRMS"),
    "dRMS": ("dRMS (Baseline − Monitor)", "", "rms", "dRMS"),
    "RMS_Baseline": ("RMS Baseline", "", "rms", "RMS_Baseline"),
    "RMS_Monitor": ("RMS Monitor", "", "rms", "RMS_Monitor"),
    "TWT": ("Horizon TWT", "ms", "rms", "TWT"),
    "Sweetness_Difference": ("Sweetness change (Monitor − Baseline)", "", "sweet", "Sweetness_Difference"),
    "Sweetness_Baseline": ("Sweetness Baseline", "", "sweet", "Sweetness_Baseline"),
    "RFI_Final": ("Remaining Fluid Index (Phase 6, 0-100 screening score)", "", "rfi", "RFI_Final"),
    "Oil_4D_Penalty": ("Oil-proximity 4D penalty (Phase 6)", "", "rfi", "Oil_4D_Penalty"),
}


def _rfi_cols(f) -> set:
    try:
        return set(pd.read_csv(f, nrows=0).columns)
    except Exception:
        return set()


def available_attrs(ws, h):
    """Attributes for a horizon. Phase 6 horizons (RFI only) also offer the RMS / NRMS / dRMS / sweetness columns
    that the RFI file carries."""
    f = ws.files(h)
    out = [k for k, (_, _, src, _) in ATTRS.items() if src in f]
    if "rfi" in f and "rms" not in f:
        cols = _rfi_cols(f["rfi"])
        out += [k for k, (_, _, src, col) in ATTRS.items() if k not in out and src != "classes" and col in cols]
    return out


def raster(ws: Workspace, h: str, attr: str, n=320) -> dict:
    if attr not in ATTRS:
        raise KeyError(attr)
    title, unit, src, col = ATTRS[attr]
    f = ws.files(h)
    if src not in f:
        if "rfi" in f and col in _rfi_cols(f["rfi"]):   # Phase 6 horizon: the RFI file carries this attribute too
            src = "rfi"
        else:
            raise FileNotFoundError(f"No {src} file for horizon {h}")
    d = read(f[src])
    x, y, v = d["X"].to_numpy(float), d["Y"].to_numpy(float), d[col].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if attr == "class":
        ok &= v >= 0
    x, y, v = x[ok], y[ok], v[ok]
    x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    span = max(x1 - x0, y1 - y0) or 1.0
    # never use cells smaller than the trace spacing (otherwise the map is full of holes)
    from scipy.spatial import cKDTree
    smp = np.c_[x, y][:: max(1, len(x) // 3000)]
    spacing = float(np.median(cKDTree(smp).query(smp, k=2)[0][:, 1])) if len(smp) > 2 else 0.0
    if len(smp) < len(x):  # decimated sample: estimate trace spacing from the bounding area instead
        spacing = math.sqrt(max((x1 - x0) * (y1 - y0), 1.0) / len(x))
    cell = max(span / n, 1.05 * spacing)
    nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
    ix = np.minimum(((x - x0) / cell).astype(int), nx - 1)
    iy = np.minimum(((y - y0) / cell).astype(int), ny - 1)
    flat = iy * nx + ix
    if attr == "class":  # majority class per cell
        counts = np.zeros((3, ny * nx))
        for k in range(3):
            counts[k] = np.bincount(flat[v == k], minlength=ny * nx)
        g = np.where(counts.sum(0) > 0, counts.argmax(0), -1).astype(float)
        g[g < 0] = np.nan
    else:
        s = np.bincount(flat, weights=v, minlength=ny * nx)
        c = np.bincount(flat, minlength=ny * nx)
        g = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    g = g.reshape(ny, nx)
    fin = g[np.isfinite(g)]
    stats = dict(min=float(fin.min()), max=float(fin.max()), mean=float(fin.mean()), p01=float(np.percentile(fin, 1)),
                 p99=float(np.percentile(fin, 99))) if fin.size else {}
    return dict(horizon=h, attr=attr, title=title, unit=unit, cat=attr == "class", x0=x0, y0=y0, cell=cell, nx=nx, ny=ny,
                values=[[None if not np.isfinite(t) else round(float(t), 3) for t in row] for row in g],
                stats=stats, points=int(ok.sum()))


# ====================================================================== 3D scene
def _bin(x, y, v, x0, y0, cell, nx, ny, cat=False):
    ix = np.clip(((x - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    if cat:
        counts = np.stack([np.bincount(flat[v == k], minlength=ny * nx) for k in range(3)])
        g = np.where(counts.sum(0) > 0, counts.argmax(0), -1).astype(np.float32)
        g[g < 0] = np.nan
    else:
        s = np.bincount(flat, weights=v, minlength=ny * nx)
        c = np.bincount(flat, minlength=ny * nx)
        g = np.where(c > 0, s / np.maximum(c, 1), np.nan).astype(np.float32)
    return g


def _b64(a: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def scene3d(ws: Workspace, attr: str = "class", n: int | None = None) -> dict:
    """All horizons on one shared X/Y grid: TWT surface + one attribute per cell (float32, base64, NaN = no data)."""
    if attr not in ATTRS:
        raise KeyError(attr)
    title, unit, src, col = ATTRS[attr]
    hs = ws.horizons()
    if not hs:
        raise FileNotFoundError("No horizons in the output working directory.")
    n = n or (200 if len(hs) <= 8 else 150)
    pts = {}
    for h in hs:
        f = ws.files(h)
        zsrc = "rms" if "rms" in f else "horizon" if "horizon" in f else None
        if not zsrc:
            continue
        d = read(f[zsrc])
        x, y, z = (d[k].to_numpy(float) for k in ("X", "Y", "TWT"))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        pts[h] = (x[ok], y[ok], z[ok])
    if not pts:
        raise FileNotFoundError("No horizon TWT found.")
    allx = np.concatenate([p[0] for p in pts.values()]); ally = np.concatenate([p[1] for p in pts.values()])
    x0, x1, y0, y1 = float(allx.min()), float(allx.max()), float(ally.min()), float(ally.max())
    from scipy.spatial import cKDTree
    big = max(pts.values(), key=lambda p: len(p[0]))
    smp = np.c_[big[0], big[1]][:: max(1, len(big[0]) // 3000)]
    spacing = float(np.median(cKDTree(smp).query(smp, k=2)[0][:, 1])) if len(smp) > 2 else 0.0
    if len(smp) < len(big[0]):
        spacing = math.sqrt(max((big[0].max() - big[0].min()) * (big[1].max() - big[1].min()), 1.0) / len(big[0]))
    cell = max(max(x1 - x0, y1 - y0, 1.0) / n, 1.05 * spacing)
    nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
    out = []
    for h, (x, y, z) in pts.items():
        zg = _bin(x, y, z, x0, y0, cell, nx, ny)
        rec = dict(name=h, z=_b64(zg), zmin=float(np.nanmin(zg)), zmax=float(np.nanmax(zg)), has_v=False)
        f = ws.files(h)
        if attr != "TWT" and src in f:
            d = read(f[src])
            vx, vy, vv = d["X"].to_numpy(float), d["Y"].to_numpy(float), d[col].to_numpy(float)
            ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vv)
            if attr == "class":
                ok &= vv >= 0
            vg = _bin(vx[ok], vy[ok], vv[ok], x0, y0, cell, nx, ny, cat=attr == "class")
            fin = vg[np.isfinite(vg)]
            if fin.size:
                rec.update(v=_b64(vg), has_v=True, p01=float(np.percentile(fin, 1)), p99=float(np.percentile(fin, 99)))
        out.append(rec)
    return dict(attr=attr, title=title, unit=unit, cat=attr == "class", x0=x0, y0=y0, cell=cell, nx=nx, ny=ny, horizons=out)


# ====================================================================== summary cache
import hashlib as _hashlib
import json as _json
import threading as _threading

_SUM_MEM: dict = {}
_SUM_LOCK = _threading.Lock()


def _summary_sig(ws: Workspace, wells, platforms) -> str:
    """Changes whenever a result file, the wells or the platforms change."""
    h = _hashlib.sha1()
    for d in (ws.rms, ws.cls, ws.sweet):
        if d.is_dir():
            for p in sorted(d.glob("*")):
                if p.is_file():
                    st = p.stat(); h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    try:
        rd, _ = rfi_dir(ws)
        if rd is not None:
            for p in sorted(Path(rd).glob("*_RFI.csv")):
                st = p.stat(); h.update(f"rfi|{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    except Exception:
        pass
    for d in (Path(ws.root) / "team_root" / P5, Path(TEAM_ROOT) / P5):
        if d.is_dir():
            for p in sorted(d.glob("*.csv")):
                st = p.stat(); h.update(f"p5|{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(_json.dumps([(w.get("name"), w.get("x"), w.get("y")) for w in wells or []], default=str).encode())
    h.update(_json.dumps([(p.get("name"), p.get("x"), p.get("y")) for p in platforms or []], default=str).encode())
    h.update(str(ws.root).encode())
    return h.hexdigest()[:16]


def summary(ws: Workspace, wells, platforms) -> dict:
    """Cached: the first call reads every horizon file (minutes on large surveys); later calls are instant.
    Kept in memory and in <results>/.chronoseis_cache so it survives an app restart."""
    sig = _summary_sig(ws, wells, platforms)
    if sig in _SUM_MEM:
        return _SUM_MEM[sig]
    with _SUM_LOCK:                                   # one computation at a time (page + Advisor asking together)
        if sig in _SUM_MEM:
            return _SUM_MEM[sig]
        cdir = Path(ws.root) / ".chronoseis_cache"
        cf = cdir / f"summary_{sig}.json"
        if cf.is_file():
            try:
                out = _json.loads(cf.read_text())
                _SUM_MEM[sig] = out
                return out
            except ValueError:
                pass
        out = _json.loads(_json.dumps(_summary(ws, wells, platforms), default=lambda v: v.item() if hasattr(v, "item") else str(v)))
        if out.get("ready"):
            _SUM_MEM.clear(); _SUM_MEM[sig] = out
            try:
                cdir.mkdir(exist_ok=True)
                for old in cdir.glob("summary_*.json"):
                    old.unlink()
                cf.write_text(_json.dumps(out))
            except OSError:
                pass
        return out
