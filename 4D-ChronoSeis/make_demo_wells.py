"""Synthetic well folder matching make_demo_data.py (same coordinates, same reservoir times).

    python make_demo_wells.py demo_data/well_data

Velocity model: 2500 m/s average from MSL (so TWT ms = 2000 * TVDSS / 2500 = 0.8 * TVDSS).
Reservoir: top at 1250 ms (TVDSS 1562.5 m), base at 1550 ms (TVDSS 1937.5 m).
  W-A : vertical, checkshot (TVDSS vs TWT) + LAS with sonic, markers in MD.   -> converts with checkshot
  W-B : deviated (MD/INC/AZI survey), LAS with sonic (µs/ft), no checkshot.  -> converts with sonic
  W-C : markers only (no checkshot, no sonic).                               -> needs seismic velocity
  W-D : outside the survey.
Also writes a depth-domain horizon (TVDSS m) one level up: horizon_depth.txt
"""
import math
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data/well_data")
for d in ("checkshots", "deviations", "las", "markers"):
    (out / d).mkdir(parents=True, exist_ok=True)
V, KB = 2500.0, 25.0
TOP, BASE = 1562.5, 1937.5          # TVDSS
twt = lambda z: 2000.0 * z / V

(out / "WellList_Information.csv").write_text(
    "Well name,X,Y,KB elevation (m)\n"
    f"W-A,{450000 + 50 * 25},{6200000 + 25 * 25},{KB}\n"
    f"W-B,{450000 + 20 * 25},{6200000 + 45 * 25},{KB}\n"
    f"W-C,{450000 + 70 * 25},{6200000 + 10 * 25},{KB}\n"
    "W-D,470000,6300000,25\n")

# checkshot W-A: TVDSS (m) vs TWT (ms), with slight velocity gradient noise but exact at reservoir
z = np.arange(0, 2600, 100.0)
(out / "checkshots" / "W-A_Checkshot.txt").write_text(
    "TVDSS TWT\n" + "\n".join(f"{-zz:.1f} {twt(zz):.2f}" for zz in z))

# LAS helper
def las(name, md, curves, kb):
    hdr = ("~Version\nVERS. 2.0 :\nWRAP. NO :\n~Well\n"
           f"STRT.M {md[0]:.2f} :\nSTOP.M {md[-1]:.2f} :\nSTEP.M {md[1] - md[0]:.2f} :\nNULL. -999.25 :\nWELL. {name} :\nEKB.M {kb} :\n~Curve\nDEPT.M : measured depth\n")
    hdr += "".join(f"{m}.{u} : \n" for m, u, _ in curves) + "~A\n"
    rows = "\n".join(" ".join([f"{md[i]:.2f}"] + [f"{c[i]:.3f}" for _, _, c in curves]) for i in range(len(md)))
    (out / "las" / f"{name}.las").write_text(hdr + rows + "\n")

rng = np.random.default_rng(3)
# W-A vertical: MD = TVDSS + KB
md = np.arange(1000 + KB, 2300 + KB, 0.5)
las("W-A", md, [("GR", "API", 60 + 20 * rng.standard_normal(len(md))),
                ("DT", "US/M", 1e6 / V + 4 * rng.standard_normal(len(md)))], KB)

# W-B deviated: 30 deg inclination below 800 m MD
mds = np.arange(0, 3201, 50.0)
inc = np.where(mds < 800, 0.0, np.minimum(30.0, (mds - 800) / 20.0))
(out / "deviations" / "W-B_Deviation.txt").write_text(
    "MD INCL AZIM\n" + "\n".join(f"{a:.1f} {b:.2f} 45.0" for a, b in zip(mds, inc)))
# tvd by minimum curvature (replicates reader) for marker placement
sys.path.insert(0, str(Path(__file__).parent))
import wellio  # noqa: E402
dev = wellio.read_deviation(out / "deviations" / "W-B_Deviation.txt")
md_top_b = float(np.interp(TOP + KB, dev["tvd"], dev["md"]))
md_base_b = float(np.interp(BASE + KB, dev["tvd"], dev["md"]))
mdl = np.arange(50 + 0.0, 2700, 0.5)  # sonic from 50 m MD
las("W-B", mdl, [("GR", "API", 70 + 15 * rng.standard_normal(len(mdl))),
                 ("DTC", "US/F", (1e6 / V) * 0.3048 + 1.5 * rng.standard_normal(len(mdl)))], KB)

# markers (MD from KB)
def markers(name, top_md, base_md):
    (out / "markers" / f"{name}_Markers.txt").write_text(
        f"Marker MD\nTop Reservoir {top_md:.2f}\nBase Reservoir {base_md:.2f}\n")

markers("W-A", TOP + KB, BASE + KB)
markers("W-B", md_top_b, md_base_b)
markers("W-C", TOP + KB, BASE + KB)
markers("W-D", TOP + KB, BASE + KB)

# depth-domain horizon matching demo_data/horizon.txt (time) via 2500 m/s
with open(out.parent / "horizon_depth.txt", "w") as h:
    h.write("inline xline depth_tvdss\n")
    for il in range(100, 160):
        for xl in range(300, 380):
            t = 1400 + 40 * math.sin((il - 100) / 12.0)
            h.write(f"{il} {xl} {t * V / 2000.0:.2f}\n")
print("wrote", out)
