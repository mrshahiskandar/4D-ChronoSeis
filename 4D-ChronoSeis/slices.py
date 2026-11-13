"""
Time-slice QC straight from the SEG-Y volumes (Baseline, Monitor, Monitor − Baseline).

Reads ONE time sample from every trace (strided read of the file, no 6.7 GB array in memory),
bins it onto a real X/Y grid and returns it in the same shape the horizon maps use, so the 2D
map and the 3D viewer can draw it like a horizon.  Nothing is written unless "Save" is pressed,
and then only into the output working directory.

Trace headers (X/Y/inline/crossline) are cached per file in the app's own state folder.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

import results

IL_BYTE, XL_BYTE = 189, 193
_INFO: dict = {}
_COL: dict = {}          # (path, mtime, sample index) -> float32 column (LRU, max 4)
_COL_ORDER: list = []
_PROG: dict = {}         # request id -> {stage, t0, t1}
_TL = threading.local()


def begin(pid: str | None):
    """Start tracking progress for the current request (pid comes from the browser)."""
    _TL.pid = pid or None
    if pid:
        if len(_PROG) > 50:
            _PROG.pop(next(iter(_PROG)), None)
        _PROG[pid] = dict(stage="Starting…", t0=time.time(), t1=None)


def note(stage: str):
    p = _PROG.get(getattr(_TL, "pid", None))
    if p:
        p["stage"] = stage


def end():
    p = _PROG.get(getattr(_TL, "pid", None))
    if p:
        p["t1"] = time.time()
    _TL.pid = None


def progress(pid: str) -> dict:
    p = _PROG.get(pid)
    if not p:
        return dict(stage="", elapsed=0.0, done=False)
    return dict(stage=p["stage"], elapsed=round((p["t1"] or time.time()) - p["t0"], 1), done=p["t1"] is not None)



def _ibm2ieee(u: np.ndarray) -> np.ndarray:
    u = u.astype(np.uint32)
    sign = np.where((u >> 31) & 1, -1.0, 1.0)
    exp = ((u >> 24) & 0x7F).astype(np.int32)
    frac = (u & 0x00FFFFFF).astype(np.float64) / 16777216.0
    return (sign * frac * np.power(16.0, exp - 64)).astype(np.float32)


class Volume:
    def __init__(self, path: Path, cache_dir: Path | None = None):
        import segyio
        self.path = Path(path)
        st = self.path.stat()
        self.key = (str(self.path), st.st_mtime_ns, st.st_size)
        note(f"Opening {self.path.name} and reading its trace headers (first time only; cached afterwards)…")
        with segyio.open(str(self.path), "r", ignore_geometry=True) as f:
            self.ntr = int(f.tracecount)
            self.ns = int(len(f.samples))
            self.dt = float(segyio.tools.dt(f)) / 1000.0
            self.t0 = float(f.samples[0])
            self.fmt = int(f.format)
            self.endian = "<" if str(getattr(f, "endian", "big")).startswith("l") else ">"
            ext = int(f.bin[segyio.BinField.ExtendedHeaders] or 0)
            self.tr0 = 3600 + 3200 * max(ext, 0)
            bps = {1: 4, 2: 4, 5: 4, 3: 2, 8: 1}.get(self.fmt)
            if bps is None:
                raise ValueError(f"SEG-Y sample format {self.fmt} is not supported for slice QC (supported: IBM float, IEEE float, int32, int16, int8).")
            self.bps = bps
            self.trsize = 240 + self.ns * bps
            if self.tr0 + self.ntr * self.trsize != st.st_size:
                raise ValueError("This SEG-Y does not have fixed-length traces; slice QC can't read it directly.")
            self._load_headers(f, segyio, cache_dir)

    def _load_headers(self, f, segyio, cache_dir):
        cf = None
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cf = Path(cache_dir) / ("hdr_" + hashlib.sha1(repr(self.key).encode()).hexdigest()[:16] + ".npz")
            if cf.exists():
                try:
                    z = np.load(cf)
                    self.x, self.y, self.il, self.xl = z["x"], z["y"], z["il"], z["xl"]
                    return
                except Exception:
                    pass
        sc = np.asarray(f.attributes(segyio.TraceField.SourceGroupScalar)[:], float)
        sc = np.where(sc == 0, 1.0, np.where(sc < 0, -1.0 / np.where(sc == 0, 1, sc), sc))
        x = np.asarray(f.attributes(segyio.TraceField.CDP_X)[:], float) * sc
        y = np.asarray(f.attributes(segyio.TraceField.CDP_Y)[:], float) * sc
        if not (np.ptp(x) > 0 or np.ptp(y) > 0):           # CDP X/Y empty: use the source coordinates
            x = np.asarray(f.attributes(segyio.TraceField.SourceX)[:], float) * sc
            y = np.asarray(f.attributes(segyio.TraceField.SourceY)[:], float) * sc
        self.x, self.y = x.astype(np.float64), y.astype(np.float64)
        self.il = np.asarray(f.attributes(IL_BYTE)[:], np.int32)
        self.xl = np.asarray(f.attributes(XL_BYTE)[:], np.int32)
        if cf:
            try:
                np.savez(cf, x=self.x, y=self.y, il=self.il, xl=self.xl)
            except Exception:
                pass

    def meta(self) -> dict:
        return dict(name=self.path.name, traces=self.ntr, samples=self.ns, dt_ms=self.dt, t0_ms=self.t0,
                    t1_ms=self.t0 + (self.ns - 1) * self.dt,
                    x_range=[float(self.x.min()), float(self.x.max())], y_range=[float(self.y.min()), float(self.y.max())])

    def index_of(self, t_ms: float) -> int:
        i = int(round((t_ms - self.t0) / self.dt))
        if i < 0 or i >= self.ns:
            raise ValueError(f"{t_ms:g} ms is outside this volume ({self.t0:g}–{self.t0 + (self.ns - 1) * self.dt:g} ms).")
        return i

    def column(self, i: int) -> np.ndarray:
        k = (self.key, i)
        if k in _COL:
            return _COL[k]
        mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.tr0, shape=(self.ntr, self.trsize))
        out = np.empty(self.ntr, np.float32)
        a0 = 240 + i * self.bps
        step = 100_000
        for a in range(0, self.ntr, step):
            note(f"Reading {self.path.name} at sample {i}: {100 * a // max(self.ntr, 1)}% of traces…")
            raw = np.ascontiguousarray(mm[a:a + step, a0:a0 + self.bps])
            if self.fmt == 1:
                out[a:a + step] = _ibm2ieee(raw.view(self.endian + "u4")[:, 0])
            elif self.fmt == 5:
                out[a:a + step] = raw.view(self.endian + "f4")[:, 0]
            elif self.fmt == 2:
                out[a:a + step] = raw.view(self.endian + "i4")[:, 0]
            elif self.fmt == 3:
                out[a:a + step] = raw.view(self.endian + "i2")[:, 0]
            else:
                out[a:a + step] = raw.view(np.int8)[:, 0]
        del mm
        _COL[k] = out
        _COL_ORDER.append(k)
        while len(_COL_ORDER) > 4:
            _COL.pop(_COL_ORDER.pop(0), None)
        return out


def _decode_block(self, raw: np.ndarray) -> np.ndarray:
    """raw: (rows, nsamp*bps) uint8 -> float32 (rows, nsamp)."""
    n = raw.shape[0]
    raw = np.ascontiguousarray(raw).reshape(-1)
    if self.fmt == 1:
        out = _ibm2ieee(raw.view(self.endian + "u4"))
    elif self.fmt == 5:
        out = raw.view(self.endian + "f4").astype(np.float32)
    elif self.fmt == 2:
        out = raw.view(self.endian + "i4").astype(np.float32)
    elif self.fmt == 3:
        out = raw.view(self.endian + "i2").astype(np.float32)
    else:
        out = raw.view(np.int8).astype(np.float32)
    return out.reshape(n, -1)


def _block(self, rows: np.ndarray, i0: int, nw: int) -> np.ndarray:
    """Samples i0 .. i0+nw-1 of the given trace indices (ascending) -> float32 (len(rows), nw)."""
    mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.tr0, shape=(self.ntr, self.trsize))
    a0 = 240 + i0 * self.bps
    raw = np.asarray(mm[rows, a0:a0 + nw * self.bps])
    del mm
    return _decode_block(self, raw)


Volume.block = _block


def cached_volume(path, cache_dir):
    """The Volume only if its header cache already exists (cheap); otherwise None."""
    p = Path(path)
    if not p.is_file() or not cache_dir:
        return None
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key in _INFO:
        return _INFO[key]
    cf = Path(cache_dir) / ("hdr_" + hashlib.sha1(repr(key).encode()).hexdigest()[:16] + ".npz")
    return volume(p, cache_dir) if cf.exists() else None


def volume(path, cache_dir=None) -> Volume:
    p = Path(path)
    st = p.stat()
    k = (str(p), st.st_mtime_ns, st.st_size)
    if k not in _INFO:
        _INFO.clear()
        _INFO[k] = Volume(p, cache_dir)
    return _INFO[k]


# ------------------------------------------------------------------ gridding
def _grid_def(vols: list[Volume], n: int):
    x0 = min(float(v.x.min()) for v in vols); x1 = max(float(v.x.max()) for v in vols)
    y0 = min(float(v.y.min()) for v in vols); y1 = max(float(v.y.max()) for v in vols)
    v0 = vols[0]
    step = np.hypot(np.diff(v0.x[:200_000]), np.diff(v0.y[:200_000]))
    step = step[step > 0]
    spacing = float(np.median(step)) if step.size else 0.0
    cell = max(max(x1 - x0, y1 - y0, 1.0) / n, 1.05 * spacing)
    return x0, y0, cell, int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1


def _bin(vol: Volume, v: np.ndarray, g):
    x0, y0, cell, nx, ny = g
    ok = np.isfinite(v)
    return results._bin(vol.x[ok], vol.y[ok], v[ok].astype(np.float64), x0, y0, cell, nx, ny).reshape(ny, nx)



def _affine(ref: Volume):
    """Least-squares map (inline, crossline) -> (X, Y) from a volume whose header coordinates are sane."""
    idx = np.linspace(0, ref.ntr - 1, min(ref.ntr, 50_000)).astype(np.int64)
    A = np.column_stack([np.ones(idx.size), ref.il[idx].astype(float), ref.xl[idx].astype(float)])
    cx, *_ = np.linalg.lstsq(A, ref.x[idx], rcond=None)
    cy, *_ = np.linalg.lstsq(A, ref.y[idx], rcond=None)
    err = max(float(np.sqrt(np.mean((A @ cx - ref.x[idx]) ** 2))), float(np.sqrt(np.mean((A @ cy - ref.y[idx]) ** 2))))
    return cx, cy, err


def _align_coords(vols: list[Volume], ref: Volume):
    """If a volume's header X/Y are not in the Baseline's coordinate frame (wrong byte/scalar), rebuild them from
    inline/crossline using the Baseline's geometry. Only in memory; the SEG-Y files are never changed."""
    xl0, xl1, yl0, yl1 = float(ref.x.min()), float(ref.x.max()), float(ref.y.min()), float(ref.y.max())
    px, py = 0.1 * (xl1 - xl0) + 1.0, 0.1 * (yl1 - yl0) + 1.0
    for v in vols:
        if v is ref or getattr(v, "_aligned_to", None) == ref.key:
            continue
        inside = np.mean((v.x >= xl0 - px) & (v.x <= xl1 + px) & (v.y >= yl0 - py) & (v.y <= yl1 + py))
        if inside >= 0.9:
            continue
        if not (np.ptp(v.il) > 0 and np.ptp(v.xl) > 0):
            raise ValueError(f"{v.path.name}: header X/Y are outside the Baseline's area and inline/crossline numbers are not usable to repair them.")
        cx, cy, err = _affine(ref)
        spacing = max(float(np.median(np.abs(np.diff(ref.x[:50_000])) + np.abs(np.diff(ref.y[:50_000])))), 1.0)
        if err > 2 * spacing:
            raise ValueError(f"{v.path.name}: header X/Y differ from the Baseline and the Baseline's inline/crossline do not map cleanly to X/Y (rms {err:.1f} m).")
        B = np.column_stack([np.ones(v.ntr), v.il.astype(float), v.xl.astype(float)])
        v.x, v.y = B @ cx, B @ cy
        v._aligned_to = ref.key
        v.note = "Monitor header X/Y were outside the Baseline frame; X/Y rebuilt from inline/crossline using the Baseline geometry."

LABEL = {"baseline": "Baseline", "monitor": "Monitor", "diff": "Monitor − Baseline"}


def slice_grid(base: Path | None, mon: Path | None, which: str, t_ms: float, n: int = 320, cache_dir=None):
    """Return (grid[ny,nx] float32 with NaN, grid definition, title, unit, true sample time)."""
    if which not in LABEL:
        raise KeyError(which)
    need = {"baseline": [base], "monitor": [mon], "diff": [base, mon]}[which]
    if not all(need):
        raise FileNotFoundError("Choose the " + ("Baseline and Monitor SEG-Y" if which == "diff" else LABEL[which] + " SEG-Y") + " first.")
    vols = [volume(p, cache_dir) for p in need]
    if base and which != "baseline":
        _align_coords(vols, volume(base, cache_dir))
    note("Matching coordinates and building the map grid…")
    g = _grid_def(vols, n)
    grids, ts = [], []
    for v in vols:
        i = v.index_of(t_ms)
        ts.append(v.t0 + i * v.dt)
        col = v.column(i)
        note(f"Gridding {v.path.name} onto X/Y…")
        grids.append(_bin(v, col, g))
    grid = grids[0] if which != "diff" else (grids[1] - grids[0]).astype(np.float32)
    if which == "diff":
        grid[~(np.isfinite(grids[0]) & np.isfinite(grids[1]))] = np.nan
    unit = "Δ amplitude" if which == "diff" else "amplitude"
    return grid, g, f"{LABEL[which]} · time slice", unit, float(np.mean(ts))


def _stats(grid):
    fin = grid[np.isfinite(grid)]
    if not fin.size:
        raise ValueError("No data at this time (all traces are empty or zero here).")
    return fin, dict(min=float(fin.min()), max=float(fin.max()), mean=float(fin.mean()),
                     p01=float(np.percentile(fin, 1)), p99=float(np.percentile(fin, 99)))


def raster(base, mon, which, t_ms, n=320, cache_dir=None, **kw) -> dict:
    if which in ("tshift", "tcc"):
        r, grid, attr, unit, title = _shift_pack(base, mon, which, n, cache_dir, dict(kw, t=t_ms))
        x0, y0, cell, nx, ny = r["g"]
        fin, st = _stats(grid)
        t = 0.5 * (r["window"][0] + r["window"][1])
        return dict(horizon=title, attr=attr, title=title, unit=unit, cat=False, x0=x0, y0=y0, cell=cell, nx=nx, ny=ny, t_ms=t,
                    stats=st, points=int(fin.size), note=_shift_note(r, grid), noise=r["noise"],
                    pct_above_noise=(float(100 * np.mean(np.abs(fin) > 2 * r["noise"]["sigma"])) if r["noise"] and which == "tshift" else None),
                    values=[[None if not np.isfinite(q) else round(float(q), 3) for q in row] for row in grid])
    grid, (x0, y0, cell, nx, ny), title, unit, t = slice_grid(base, mon, which, t_ms, n, cache_dir)
    fin, st = _stats(grid)
    return dict(horizon=f"{LABEL[which]} @ {t:g} ms", attr=f"slice_{which}", title=f"{title} @ {t:g} ms", unit=unit, cat=False,
                x0=x0, y0=y0, cell=cell, nx=nx, ny=ny, t_ms=t, stats=st, points=int(fin.size),
                values=[[None if not np.isfinite(q) else round(float(q), 3) for q in row] for row in grid])


def scene(base, mon, which, t_ms, n=200, cache_dir=None, **kw) -> dict:
    """Same structure as results.scene3d, with one flat 'horizon' at the slice time."""
    if which in ("tshift", "tcc"):
        r, grid, attr, unit, title = _shift_pack(base, mon, which, n, cache_dir, dict(kw, t=t_ms))
        x0, y0, cell, nx, ny = r["g"]
        fin, st = _stats(grid)
        t = 0.5 * (r["window"][0] + r["window"][1])
        m = max(abs(st["p01"]), abs(st["p99"])) or 1.0
        lo, hi = (-m, m) if which == "tshift" else (st["p01"], st["p99"])
        z = np.where(np.isfinite(grid), np.float32(t), np.float32(np.nan)).astype(np.float32)
        rec = dict(name=title, z=results._b64(z), zmin=float(t), zmax=float(t), has_v=True, v=results._b64(grid.astype(np.float32)), p01=lo, p99=hi)
        v0 = volume(base, cache_dir)
        return dict(attr=attr, title=title, unit=unit, cat=False, x0=x0, y0=y0, cell=cell, nx=nx, ny=ny, horizons=[rec],
                    tspan=[v0.t0, v0.t0 + (v0.ns - 1) * v0.dt])
    grid, (x0, y0, cell, nx, ny), title, unit, t = slice_grid(base, mon, which, t_ms, n, cache_dir)
    fin, st = _stats(grid)
    m = max(abs(st["p01"]), abs(st["p99"])) or 1.0           # amplitudes are centred on zero
    z = np.where(np.isfinite(grid), np.float32(t), np.float32(np.nan)).astype(np.float32)
    rec = dict(name=f"{LABEL[which]} @ {t:g} ms", z=results._b64(z), zmin=float(t), zmax=float(t), has_v=True,
               v=results._b64(grid), p01=-m, p99=m)
    v0 = volume(base if which != "monitor" else mon, cache_dir)
    return dict(attr=f"slice_{which}", title=f"{title} @ {t:g} ms", unit=unit, cat=False, x0=x0, y0=y0, cell=cell, nx=nx, ny=ny,
                horizons=[rec], tspan=[v0.t0, v0.t0 + (v0.ns - 1) * v0.dt])


# ------------------------------------------------------------------ 4D time shift (windowed cross-correlation)
_SHIFT: dict = {}
_SHIFT_ORDER: list = []


def _pair(vb: Volume, vm: Volume):
    """Trace indices (ib, im) of Baseline and Monitor traces at the same position: by inline/crossline when they
    match, else by nearest X/Y (after the coordinate alignment)."""
    if np.ptp(vb.il) > 0 and np.ptp(vb.xl) > 0 and np.ptp(vm.il) > 0:
        kb = vb.il.astype(np.int64) * 100_000 + vb.xl.astype(np.int64)
        km = vm.il.astype(np.int64) * 100_000 + vm.xl.astype(np.int64)
        _, ib, im = np.intersect1d(kb, km, return_indices=True)
        if ib.size >= 0.5 * min(vb.ntr, vm.ntr):
            return ib, im, "inline/crossline"
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ValueError("Baseline and Monitor traces cannot be matched by inline/crossline here, and scipy is missing for the X/Y fallback.")
    tr = cKDTree(np.column_stack([vm.x, vm.y]))
    d, im = tr.query(np.column_stack([vb.x, vb.y]))
    sp = max(float(np.median(np.hypot(np.diff(vb.x[:100_000]), np.diff(vb.y[:100_000])))), 1.0)
    ok = d <= 0.6 * sp
    if ok.sum() < 0.3 * vb.ntr:
        raise ValueError("Baseline and Monitor traces do not coincide in X/Y; check that both use the same coordinate system.")
    return np.flatnonzero(ok), im[ok], "nearest X/Y"


def _xcorr_shift(a: np.ndarray, m: np.ndarray, L: int, w: np.ndarray):
    """a: (rows, nw) Baseline window; m: (rows, nw + 2L) Monitor window with L samples of margin each side.
    Returns (shift in samples, correlation coefficient); positive shift = the Monitor event arrives later."""
    rows, nw = a.shape
    a = a - (a * w).sum(1, keepdims=True) / w.sum()
    na = np.sqrt((w * a * a).sum(1)) + 1e-12
    cc = np.empty((rows, 2 * L + 1), np.float32)
    for k, lag in enumerate(range(-L, L + 1)):
        seg = m[:, L + lag:L + lag + nw]
        seg = seg - (seg * w).sum(1, keepdims=True) / w.sum()
        cc[:, k] = (w * a * seg).sum(1) / (na * (np.sqrt((w * seg * seg).sum(1)) + 1e-12))
    k0 = np.argmax(cc, 1)
    r = np.arange(rows)
    c0 = cc[r, k0]
    km, kp = np.clip(k0 - 1, 0, 2 * L), np.clip(k0 + 1, 0, 2 * L)
    ym, yp = cc[r, km], cc[r, kp]
    den = ym - 2 * c0 + yp
    frac = np.where((k0 > 0) & (k0 < 2 * L) & (den < -1e-9), 0.5 * (ym - yp) / den, 0.0)
    return (k0 - L) + frac, c0


def _shift_core(vb: Volume, vm: Volume, ib, im, t0: float, t1: float, maxlag_ms: float):
    dt = vb.dt
    L = max(1, int(round(maxlag_ms / dt)))
    i0, i1 = vb.index_of(t0), vb.index_of(t1)
    if i1 - i0 < 8:
        raise ValueError(f"The window {t0:g}–{t1:g} ms is too short (needs at least {9 * dt:g} ms).")
    nw = i1 - i0 + 1
    if i0 - L < 0 or i1 + L >= min(vb.ns, vm.ns):
        raise ValueError(f"The window plus the ±{maxlag_ms:g} ms lag margin leaves the volume's time range ({vb.t0:g}–{vb.t0 + (vb.ns - 1) * dt:g} ms).")
    w = np.hanning(nw + 2)[1:-1].astype(np.float32)[None, :]
    shifts, ccs = np.empty(ib.size, np.float32), np.empty(ib.size, np.float32)
    order = np.argsort(ib, kind="stable")
    step = 20_000
    for a in range(0, ib.size, step):
        note(f"Time shift {t0:g}–{t1:g} ms: {100 * a // max(ib.size, 1)}% of traces…")
        sel = order[a:a + step]
        rb, rm = ib[sel], im[sel]
        o2 = np.argsort(rm, kind="stable")
        A = vb.block(rb, i0, nw)
        M = np.empty((sel.size, nw + 2 * L), np.float32)
        M[o2] = vm.block(rm[o2], i0 - L, nw + 2 * L)
        s_, c_ = _xcorr_shift(A, M, L, w)
        shifts[sel], ccs[sel] = s_ * dt, c_
    return shifts, ccs


def shift_grid(base, mon, t0: float, t1: float, n: int = 320, cache_dir=None, lag: float = 16.0, cc: float = 0.5,
               r0: float | None = None, r1: float | None = None):
    """Time-shift map (ms; Monitor later = +) and correlation map between Baseline and Monitor in a time window."""
    if not (base and mon):
        raise FileNotFoundError("Choose the Baseline and Monitor SEG-Y first.")
    if t1 <= t0:
        raise ValueError("The window end must be later than its start.")
    key = (str(base), str(mon), round(t0, 3), round(t1, 3), round(lag, 3), round(cc, 3), r0, r1, n)
    if key in _SHIFT:
        return _SHIFT[key]
    vb, vm = volume(base, cache_dir), volume(mon, cache_dir)
    if abs(vb.dt - vm.dt) > 1e-6 or abs(vb.t0 - vm.t0) > 1e-6:
        raise ValueError("Baseline and Monitor have different sample intervals or start times; resample them first.")
    _align_coords([vm], vb)
    note("Pairing Baseline and Monitor traces…")
    ib, im, how = _pair(vb, vm)
    shifts, ccs = _shift_core(vb, vm, ib, im, t0, t1, lag)
    noise = None
    if r0 is not None and r1 is not None and r1 > r0:
        rs, rc = _shift_core(vb, vm, ib, im, r0, r1, lag)
        good = rc >= cc
        if good.sum() > 100:
            med = float(np.median(rs[good]))
            noise = dict(sigma=float(1.4826 * np.median(np.abs(rs[good] - med))), median=med, window=[r0, r1], used=int(good.sum()))
    sh = np.full(vb.ntr, np.nan, np.float32); co = np.full(vb.ntr, np.nan, np.float32)
    sh[ib], co[ib] = shifts, ccs
    sh_m = np.where(co >= cc, sh, np.nan)
    g = _grid_def([vb], n)
    note("Gridding the time-shift map…")
    gs, gc = _bin(vb, sh_m, g), _bin(vb, co, g)
    out = dict(shift=gs, cc=gc, g=g, how=how, noise=noise, window=(t0, t1), paired=int(ib.size), ntr=vb.ntr, cc_min=cc, lag=lag, dt=vb.dt)
    _SHIFT[key] = out
    _SHIFT_ORDER.append(key)
    while len(_SHIFT_ORDER) > 3:
        _SHIFT.pop(_SHIFT_ORDER.pop(0), None)
    return out


def _shift_pack(base, mon, which, n, cache_dir, kw):
    t0, t1 = float(kw["t"]), float(kw.get("t1") or kw["t"] + 100.0)
    r = shift_grid(base, mon, t0, t1, n, cache_dir, float(kw.get("lag") or 16.0), float(kw.get("cc") or 0.5), kw.get("r0"), kw.get("r1"))
    if which == "tshift":
        grid, attr, unit, title = r["shift"], "tshift", "ms (Monitor later = +)", "4D time shift"
    else:
        grid, attr, unit, title = r["cc"], "tshift_cc", "correlation", "Time-shift confidence (correlation)"
    return r, grid, attr, unit, f"{title} · {t0:g}–{t1:g} ms"


def _shift_note(r, grid):
    fin = grid[np.isfinite(grid)]
    msg = f"{r['paired']:,} of {r['ntr']:,} traces paired by {r['how']}; correlation ≥ {r['cc_min']:g} kept; max lag ±{r['lag']:g} ms."
    if r["noise"]:
        msg += f" Noise floor (reference {r['noise']['window'][0]:g}–{r['noise']['window'][1]:g} ms): σ = {r['noise']['sigma']:.2f} ms."
    return msg


def save_csv(grid_dict: dict, path: Path):
    r = grid_dict
    v = np.array([[np.nan if q is None else q for q in row] for row in r["values"]], float)
    iy, ix = np.where(np.isfinite(v))
    import pandas as pd
    pd.DataFrame(dict(X=r["x0"] + (ix + 0.5) * r["cell"], Y=r["y0"] + (iy + 0.5) * r["cell"], value=v[iy, ix])).to_csv(path, index=False)
