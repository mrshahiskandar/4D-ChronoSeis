"""Create a small synthetic base/monitor SEG-Y pair (+ horizon, wells) with a known 4D anomaly.

    python make_demo_data.py demo_data
"""
import sys
from pathlib import Path

import numpy as np
import segyio

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

ILS, XLS, NS, DT = np.arange(100, 160), np.arange(300, 380), 500, 4.0
t = np.arange(NS) * DT


def ricker(f=25.0):
    tt = np.arange(-60, 61) * DT / 1000.0
    return (1 - 2 * (np.pi * f * tt) ** 2) * np.exp(-((np.pi * f * tt) ** 2))


w = ricker()
# smooth structural trend so neighbouring traces look alike
refl = rng.standard_normal((len(ILS), len(XLS), NS)).astype(np.float32)
refl = np.apply_along_axis(lambda r: r, 2, refl)
from scipy.ndimage import gaussian_filter  # noqa: E402

refl = gaussian_filter(refl, sigma=(2.5, 2.5, 0.0))
base = np.apply_along_axis(lambda r: np.convolve(r, w, mode="same"), 2, refl).astype(np.float32)
base /= np.abs(base).max()

# 4D change: amplitude up + 2-sample slow-down inside an elliptical patch, window 1200-1600 ms
mon = base.copy()
ii, xx = np.meshgrid(np.arange(len(ILS)), np.arange(len(XLS)), indexing="ij")
patch = ((ii - 25) / 9.0) ** 2 + ((xx - 50) / 14.0) ** 2 < 1
s0, s1 = int(1200 / DT), int(1600 / DT)
for a, b in zip(*np.where(patch)):
    tr = base[a, b]
    shifted = np.interp(t, t + 6.0, tr)  # events arrive 6 ms LATER in the monitor (slow-down)
    mon[a, b, s0:s1] = 1.35 * shifted[s0:s1]
# a bright, UNCHANGED lens next to the swept area: a planted "bypassed" target (inline ~140, crossline ~352)
byp = ((ii - 40) / 5.0) ** 2 + ((xx - 52) / 9.0) ** 2 < 1
for a, b in zip(*np.where(byp)):
    base[a, b, s0:s1] *= 2.2
    mon[a, b, s0:s1] = base[a, b, s0:s1]
mon += rng.standard_normal(mon.shape).astype(np.float32) * 0.03
base += rng.standard_normal(base.shape).astype(np.float32) * 0.03


def write(name, vol):
    spec = segyio.spec()
    spec.ilines, spec.xlines, spec.samples = ILS, XLS, t
    spec.format, spec.sorting = 5, 2
    with segyio.create(out / name, spec) as f:
        k = 0
        for a, il in enumerate(ILS):
            for b, xl in enumerate(XLS):
                f.header[k] = {segyio.su.iline: int(il), segyio.su.xline: int(xl),
                               segyio.su.cdpx: int(450000 + (xl - 300) * 25), segyio.su.cdpy: int(6200000 + (il - 100) * 25),
                               segyio.su.scalco: 1}
                f.trace[k] = vol[a, b]
                k += 1
        f.bin.update(hns=NS, hdt=int(DT * 1000))


write("Baseline.sgy", base)
write("Monitor.sgy", mon)

with open(out / "horizon.txt", "w") as h:
    h.write("inline xline twt_ms\n")
    for il in ILS:
        for xl in XLS:
            h.write(f"{il} {xl} {1400 + 40 * np.sin((il - 100) / 12.0):.1f}\n")

# well folder (well list, checkshots, sonic logs, deviations, markers) + a depth-domain horizon
import subprocess  # noqa: E402
subprocess.run([sys.executable, str(Path(__file__).parent / "make_demo_wells.py"), str(out / "well_data")], check=True)
print("wrote", out)
