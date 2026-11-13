"""Off-horizon 4D scan: the Baseline/Monitor difference sampled every `step` ms up to +-`max` ms above and below a picked
horizon. The engine's horizon attributes only see +-10 ms around the pick; this looks for 4D response that sits away from
the horizon (shallower or deeper) and may have been missed. Screening only: an off-horizon response can be real, but it can
also be noise, multiples or processing differences, so NRMS (repeatability) is reported with it."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import results
import slices


def _keys(il, xl):
    return np.asarray(il, np.int64) * 1_000_000 + np.asarray(xl, np.int64)


def _trace_index(vol, il, xl):
    k = _keys(vol.il, vol.xl)
    order = np.argsort(k)
    ks = k[order]
    q = _keys(il, xl)
    pos = np.searchsorted(ks, q)
    pos = np.clip(pos, 0, len(ks) - 1)
    hit = ks[pos] == q
    return np.where(hit, order[pos], -1)


def scan(B, M, ws, horizon, wells=None, platforms=None, step_ms=4.0, max_ms=150.0, max_traces=6000, guard_ms=10.0):
    f = ws.files(horizon)
    src = f.get("rms") or f.get("horizon")
    if src is None:
        raise FileNotFoundError(f"No horizon file for {horizon}.")
    h = results.read(src)
    need = {"Inline", "Crossline", "X", "Y", "TWT"}
    if not need <= set(h.columns):
        raise ValueError(f"{horizon}: the horizon file needs {sorted(need)}.")
    h = h.dropna(subset=list(need))
    if len(h) > max_traces:
        h = h.sample(max_traces, random_state=7)
    il, xl = h["Inline"].to_numpy(int), h["Crossline"].to_numpy(int)
    tb, tm = _trace_index(B, il, xl), _trace_index(M, il, xl)
    ok = (tb >= 0) & (tm >= 0)
    h, tb, tm = h[ok].reset_index(drop=True), tb[ok], tm[ok]
    if len(h) < 50:
        raise ValueError("Too few horizon traces match the volumes' inline/crossline numbers.")
    if abs(B.dt - M.dt) > 1e-6 or B.ns != M.ns:
        raise ValueError("Baseline and Monitor must share sample interval and length (use the matched volumes).")
    dt, t0, ns = B.dt, B.t0, B.ns
    hw = max(0, int(round(step_ms / 2.0 / dt)))
    offs = np.arange(-max_ms, max_ms + step_ms / 2, step_ms)
    osamp = np.round(offs / dt).astype(int)
    tt = h["TWT"].to_numpy(float)
    ic = np.round((tt - t0) / dt).astype(int)
    order = np.argsort(tb)
    nrms = np.full((len(h), len(offs)), np.nan, np.float32)
    ndrms = np.full_like(nrms, np.nan)
    CH = 256
    for a in range(0, len(order), CH):
        slices.note(f"Scanning {horizon} off-horizon: {100 * a // len(order)}%…")
        sel = order[a:a + CH]
        lo = int(max(0, ic[sel].min() + osamp.min() - hw)); hi = int(min(ns - 1, ic[sel].max() + osamp.max() + hw))
        if hi <= lo:
            continue
        nw = hi - lo + 1
        rb, rm = tb[sel], tm[sel]
        ob = np.argsort(rb); om = np.argsort(rm)
        bb = np.empty((len(sel), nw), np.float32); mm = np.empty_like(bb)
        bb[ob] = B.block(rb[ob], lo, nw); mm[om] = M.block(rm[om], lo, nw)
        dd = mm - bb
        def wsum(x):
            c = np.concatenate([np.zeros((len(sel), 1)), np.cumsum(x.astype(np.float64) ** 2, axis=1)], axis=1)
            P = ic[sel][:, None] - lo + osamp[None, :]
            lo_i, hi_i = P - hw, P + hw + 1
            good = (lo_i >= 0) & (hi_i <= nw)
            li, hi2 = np.clip(lo_i, 0, nw), np.clip(hi_i, 0, nw)
            s = np.take_along_axis(c, hi2, 1) - np.take_along_axis(c, li, 1)
            return np.sqrt(np.where(good, s / (2 * hw + 1), np.nan))
        Rb, Rm, Rd = wsum(bb), wsum(mm), wsum(dd)
        den = Rb + Rm
        with np.errstate(invalid="ignore", divide="ignore"):
            nrms[sel] = np.where(den > 1e-12, 200 * Rd / den, np.nan)
            ndrms[sel] = np.where(den > 1e-12, 200 * (Rb - Rm) / den, np.nan)
    prof = np.nanmedian(nrms, axis=0)
    p90 = np.nanpercentile(nrms, 90, axis=0)
    ndp = np.nanmedian(np.abs(ndrms), axis=0)
    far = np.abs(offs) >= guard_ms
    bg = float(np.nanmedian(prof[far])); sg = float(1.4826 * np.nanmedian(np.abs(prof[far] - bg))) or 1e-6
    peaks = []
    for k in np.where(far & (prof > bg + 3 * sg))[0]:
        if not peaks or offs[k] - peaks[-1]["to_ms"] > step_ms * 1.5:
            peaks.append(dict(from_ms=float(offs[k]), to_ms=float(offs[k]), max_nrms=float(prof[k]), at_ms=float(offs[k])))
        else:
            p = peaks[-1]; p["to_ms"] = float(offs[k])
            if prof[k] > p["max_nrms"]:
                p["max_nrms"], p["at_ms"] = float(prof[k]), float(offs[k])
    for p in peaks:
        p["z"] = round((p["max_nrms"] - bg) / sg, 1)
        p["max_nrms"] = round(p["max_nrms"], 1)
    # voxel clusters away from the horizon
    v = nrms[:, far]
    vv = v[np.isfinite(v)]
    thr = float(np.median(vv) + 3 * 1.4826 * np.median(np.abs(vv - np.median(vv)))) if vv.size else float("nan")
    cl = []
    if vv.size:
        from scipy import ndimage as ndi
        x, y = h["X"].to_numpy(float), h["Y"].to_numpy(float)
        cell = max(150.0, 3 * math.sqrt(max(np.ptp(x) * np.ptp(y), 1.0) / len(h)))
        ix, iy = ((x - x.min()) / cell).astype(int), ((y - y.min()) / cell).astype(int)
        nx, ny = ix.max() + 1, iy.max() + 1
        fi = np.where(far)[0]
        G = np.zeros((ny, nx, len(fi)), np.float32); C = np.zeros_like(G)
        for j, k in enumerate(fi):
            col = nrms[:, k]; ok2 = np.isfinite(col)
            np.add.at(G[:, :, j], (iy[ok2], ix[ok2]), col[ok2]); np.add.at(C[:, :, j], (iy[ok2], ix[ok2]), 1)
        mean = np.where(C > 0, G / np.maximum(C, 1), np.nan)
        lab, n = ndi.label(np.nan_to_num(mean, nan=0) > thr)
        known = None
        if "classes" in f:
            c = results.read(f["classes"])
            if {"Inline", "Crossline", "4D_Class_Code"} <= set(c.columns):
                kk = dict(zip(_keys(c["Inline"], c["Crossline"]), c["4D_Class_Code"]))
                known = np.array([kk.get(q, -1) in (1, 2) for q in _keys(il[ok], xl[ok])])  # hardening/softening at the horizon
        for q in range(1, n + 1):
            m = lab == q
            if m.sum() < 3:
                continue
            ys, xs, ks = np.where(m)
            pts = np.zeros(len(h), bool)
            for yy, xx in set(zip(ys, xs)):
                pts |= (iy == yy) & (ix == xx)
            offc = offs[fi][ks]
            u = dict(x=float(x[pts].mean()), y=float(y[pts].mean()), area_km2=round(len(set(zip(ys, xs))) * cell * cell / 1e6, 2),
                     from_ms=float(offc.min()), to_ms=float(offc.max()), mean_nrms=round(float(mean[m].mean()), 1),
                     twt_ms=round(float(tt[pts].mean() + offc.mean()), 0), voxels=int(m.sum()))
            u["score"] = u["voxels"] * u["mean_nrms"]
            if known is not None:
                u["over_known_change_pct"] = round(100 * float(known[pts].mean()), 0)
            if wells:
                dw = min(((math.hypot(u["x"] - w["x"], u["y"] - w["y"]), w["name"]) for w in wells)); u["nearest_well"], u["well_km"] = dw[1], round(dw[0] / 1000, 2)
            if platforms:
                dp = min(((math.hypot(u["x"] - p["x"], u["y"] - p["y"]), p["name"]) for p in platforms)); u["nearest_platform"], u["platform_km"] = dp[1], round(dp[0] / 1000, 2)
            cl.append(u)
        cl.sort(key=lambda u: -u["score"]); cl = cl[:6]
        for i, u in enumerate(cl, 1):
            u["rank"] = i; u.pop("score", None)
    return dict(horizon=horizon, step_ms=step_ms, max_ms=max_ms, window_ms=round((2 * hw + 1) * dt, 1), traces=int(len(h)),
                offsets=[float(o) for o in offs], median_nrms=[None if not np.isfinite(p) else round(float(p), 1) for p in prof],
                p90_nrms=[None if not np.isfinite(p) else round(float(p), 1) for p in p90],
                median_abs_ndrms=[None if not np.isfinite(p) else round(float(p), 1) for p in ndp],
                at_horizon_nrms=round(float(prof[int(np.argmin(np.abs(offs)))]), 1), background_nrms=round(bg, 1), sigma=round(sg, 2),
                peaks=peaks, clusters=cl, cluster_threshold=None if not np.isfinite(thr) else round(thr, 1), guard_ms=guard_ms)
