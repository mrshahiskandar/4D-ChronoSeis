"""
TEST FIXTURE ONLY: writes files in the same formats as the team 4D notebook's outputs, computed
from the synthetic demo SEG-Y, so the Summary / Detail Analysis views can be tested without a GPU.
These are NOT outputs of the team notebook and must never be presented as results.

    python tests/make_fixture_outputs.py demo_data workspace_fixture
Also writes demo inputs: Platform.csv, a wide well-marker CSV and a flat well folder with X/Y in LAS headers.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import segyio
from scipy import ndimage as ndi

demo = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
ws = Path(sys.argv[2] if len(sys.argv) > 2 else "workspace_fixture")
rms_dir = ws / "Autopick_Horizon" / "4D_RMS_NRMS_3"
cls_dir = rms_dir / "4D_Platform_Constrained_Maps" / "Constrained"
sw_dir = ws / "Autopick_Horizon" / "Sweetness"
for d in (rms_dir / "Maps", cls_dir, sw_dir):
    d.mkdir(parents=True, exist_ok=True)


def load(p):
    with segyio.open(p, ignore_geometry=True) as f:
        return (np.stack([f.trace[i] for i in range(f.tracecount)]), np.asarray(f.attributes(189)[:]),
                np.asarray(f.attributes(193)[:]), np.asarray(f.attributes(181)[:], float), np.asarray(f.attributes(185)[:], float),
                np.asarray(f.samples))


B, il, xl, X, Y, t = load(str(demo / "Baseline.sgy"))
M = load(str(demo / "Monitor.sgy"))[0]
dt = t[1] - t[0]
hz = pd.read_csv(demo / "horizon.txt", sep=r"\s+")
twt_res = hz.set_index(["inline", "xline"]).loc[list(zip(il, xl)), "twt_ms"].to_numpy()
platforms = pd.DataFrame({"Platform": ["PLT-A", "PLT-B"], "X(m)": [451250.0, 452800.0], "Y(m)": [6200625.0, 6203500.0]})
platforms.to_csv(demo / "Platform.csv", index=False)
RADIUS = 3000.0


def along(twt, name):
    k = np.rint((twt - t[0]) / dt).astype(int)
    hw = int(round(10 / dt))
    idx = np.clip(k[:, None] + np.arange(-hw, hw + 1)[None, :], 0, len(t) - 1)
    b = np.take_along_axis(B, idx, 1).astype(float)
    m = np.take_along_axis(M, idx, 1).astype(float)
    rb, rm = np.sqrt((b ** 2).mean(1)), np.sqrt((m ** 2).mean(1))
    den = rb + rm
    df = pd.DataFrame(dict(X=X, Y=Y, Inline=il, Crossline=xl, TWT=twt, RMS_Baseline=rb, RMS_Monitor=rm, dRMS=rb - rm,
                           NdRMS_pct=200 * (rb - rm) / den, NRMS=200 * np.sqrt(((m - b) ** 2).mean(1)) / den))
    df.to_csv(rms_dir / f"{name}_4D_RMS_NRMS.txt", sep="\t", index=False)
    pd.DataFrame(dict(X=X, Y=Y, Inline=il, Crossline=xl, TWT=twt)).to_csv(ws / "Autopick_Horizon" / f"{name}.txt", sep="\t", index=False)
    # classes (simplified version of the notebook's rule, for fixture purposes)
    uil, uxl = np.unique(il), np.unique(xl)
    ri, ci = np.searchsorted(uil, il), np.searchsorted(uxl, xl)
    g = np.full((len(uil), len(uxl)), np.nan); g[ri, ci] = df.NdRMS_pct
    gn = np.full_like(g, np.nan); gn[ri, ci] = df.NRMS
    nd = ndi.gaussian_filter(np.nan_to_num(g), 1.5)[ri, ci]
    nr = ndi.gaussian_filter(np.nan_to_num(gn), 1.5)[ri, ci]
    d2 = np.min([(X - px) ** 2 + (Y - py) ** 2 for px, py in zip(platforms["X(m)"], platforms["Y(m)"])], axis=0)
    near = d2 <= RADIUS ** 2
    code = np.zeros(len(df), int)
    ch = near & (np.abs(nd) >= 15) & (nr >= 15)
    code[ch & (nd > 0)] = 1
    code[ch & (nd < 0)] = 2
    c = df.copy()
    c["Nearest_Platform_Distance_m"] = np.sqrt(d2); c["Within_Platform_Radius"] = near
    c["NdRMS_Smoothed_pct"] = nd; c["NRMS_Smoothed"] = nr; c["4D_Class_Code"] = code
    c["4D_Class"] = [["Unswept", "Hardening", "Softening"][k] for k in code]
    c.to_csv(cls_dir / f"{name}_4D_Classes.csv", index=False)
    # sweetness (envelope / sqrt(frequency) proxy, fixture only)
    from scipy.signal import hilbert
    eb, em = np.abs(hilbert(b, axis=1)).mean(1), np.abs(hilbert(m, axis=1)).mean(1)
    pd.DataFrame(dict(X=X, Y=Y, Inline=il, Crossline=xl, TWT=twt, Sweetness_Baseline=eb, Sweetness_Monitor=em,
                      Sweetness_Difference=em - eb, Valid_Samples_Baseline=b.shape[1], Valid_Samples_Monitor=m.shape[1])
                 ).to_csv(sw_dir / f"{name}_Sweetness.txt", sep="\t", index=False)


along(np.full(len(il), 500.0), "Shallow Marker")
along(twt_res, "Top Reservoir")
along(twt_res + 120, "Mid Reservoir")
along(np.full(len(il), 1800.0), "Deep Marker")

# flat well folder (LAS with X/Y headers) + wide marker CSV
wl = pd.read_csv(demo / "well_data" / "WellList_Information.csv")
wf = demo / "wells_flat"
wf.mkdir(exist_ok=True)
for _, r in wl.iterrows():
    src = demo / "well_data" / "las" / f"{r['Well name']}.las"
    body = src.read_text() if src.exists() else "~Version\nVERS. 2.0 :\n~Well\nNULL. -999.25 :\n~Curve\nDEPT.M :\n~A\n"
    body = body.replace("~Well\n", f"~Well\nWELL. {r['Well name']} :\nXWELL.M {r['X']} :\nYWELL.M {r['Y']} :\n", 1)
    (wf / f"{r['Well name']}.las").write_text(body)
# TEST ONLY: checkshots (TVDSS m, TWT ms at ~2000 m/s) and deviation surveys (MDKB, TVD, INCL, AZIM)
def _dev(kop, build_per_30m, max_inc, azi, td=2600.0, step=30.0, kb=25.0):
    md, inc, rows_ = 0.0, 0.0, []
    tvd = 0.0
    while md <= td:
        rows_.append((md, tvd, inc, azi))
        nxt = inc + (build_per_30m if md >= kop else 0.0)
        nxt = min(nxt, max_inc)
        tvd += step * np.cos(np.deg2rad((inc + nxt) / 2)); md += step; inc = nxt
    return rows_
plans = {0: None, 1: (500, 3.0, 40.0, 125.0), 2: (700, 2.0, 25.0, 250.0)}
for k, (_, r) in enumerate(wl.iterrows()):
    nm = r["Well name"]
    if k == 3:
        continue                                   # one well without checkshot (wellhead only)
    cs = "# TVDSS(m) TWT(ms)\n" + "\n".join(f"{z:.1f} {z * 1.0:.2f}" for z in np.arange(-25.0, 2500.1, 50.0))
    (wf / f"{nm}_Checkshot.txt").write_text(cs + "\n")
    if plans.get(k):
        (wf / f"{nm}_Deviation.txt").write_text("# MDKB TVD INCL AZIM\n" + "\n".join(" ".join(f"{v:.2f}" for v in row) for row in _dev(*plans[k])) + "\n")
rows = []
for mk, off in (("Top Reservoir", 0), ("Mid Reservoir", 120)):
    row = {"Marker": mk}
    for _, r in wl.iterrows():
        i = int(round((r["Y"] - 6200000) / 25)) + 100; j = int(round((r["X"] - 450000) / 25)) + 300
        if 100 <= i < 160 and 300 <= j < 380:
            row[f"{r['Well name']} X (m)"] = r["X"]; row[f"{r['Well name']} Y (m)"] = r["Y"]
            row[f"{r['Well name']} TWT (ms)"] = float(hz.set_index(["inline", "xline"]).loc[(i, j), "twt_ms"]) + off
    rows.append(row)
pd.DataFrame(rows).to_csv(demo / "Combined_Well_Markers.csv", index=False)
print("fixture workspace:", ws)
