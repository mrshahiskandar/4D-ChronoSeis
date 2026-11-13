"""
Well, horizon and folder handling for 4D Seismic Insight.

- scan_folder(): classify what a folder holds (seismic, LAS, checkshots, markers, deviations,
  well list, horizons) with a domain hint for each and per-well availability.
- prepare_wells(): read markers per well and bring them into the seismic domain
  (time or depth), using per well: checkshot  ->  sonic log  ->  seismic velocity.
- prepare_horizon(): bring a horizon into the seismic domain.

Formats are read heuristically (column names, units, value ranges). Every assumption made
is reported back so a user can check it. Depths are handled as TVDSS (metres, positive down)
and seismic datum is assumed at mean sea level unless stated.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

SEGY_EXT = {".sgy", ".segy", ".seg"}
TABLE_EXT = {".txt", ".csv", ".dat", ".asc", ".prn", ".tsv", ".xyz", ".xlsx", ".xls"}
SONIC_MNEM = ("DT", "DTC", "DTCO", "DT4P", "DTP", "AC", "SONIC", "DT24", "DTCOMP", "DTLN", "DTL")
FT = 0.3048


# ============================================================================ small helpers
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


KIND_PAT = re.compile(
    r"[_\-\s.]*(composite[_\-\s]*interp|checkshots?|chk|tdr|time[_\-\s]*depth|deviations?|dev|survey|"
    r"markers?|tops|picks|interp|composite|las|log)s?$", re.I)


def well_key(name: str) -> str:
    """'Well-4_Composite_Interp' -> 'well4';  'W-A_Checkshot' -> 'wa'."""
    s = Path(name).stem
    for _ in range(3):
        s2 = KIND_PAT.sub("", s)
        if s2 == s:
            break
        s = s2
    return norm(s)


def well_label(name: str) -> str:
    s = Path(name).stem
    for _ in range(3):
        s2 = KIND_PAT.sub("", s)
        if s2 == s:
            break
        s = s2
    return s.strip("_- ") or Path(name).stem


def _num(tok: str):
    try:
        v = float(tok.replace("D", "E").replace("d", "e"))
        return v if math.isfinite(v) else None
    except (ValueError, AttributeError):
        return None


def read_table(path) -> tuple[list[str], list[list]]:
    """Generic text/CSV/Excel table reader. Returns (header, rows); numeric cells as float."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(p)
        return [str(c) for c in df.columns], [[(None if (isinstance(v, float) and math.isnan(v)) else v) for v in r]
                                              for r in df.itertuples(index=False)]
    lines = [l.rstrip("\n\r") for l in p.read_text(errors="ignore").splitlines()]
    lines = [l for l in lines if l.strip() and not l.lstrip().startswith(("#", "//", "!", "~", ";"))]
    if not lines:
        return [], []
    sample = "\n".join(lines[:20])
    delim = "," if sample.count(",") >= len(lines[:20]) else "\t" if "\t" in sample else ";" if sample.count(";") >= len(lines[:20]) else None
    split = (lambda l: [t.strip().strip('"') for t in l.split(delim)]) if delim else (lambda l: l.split())
    toks = [split(l) for l in lines]
    # header = last line before the first mostly-numeric line, if it is mostly non-numeric
    first_num = next((i for i, t in enumerate(toks) if sum(_num(x) is not None for x in t) >= max(1, len(t) // 2)), None)
    header = []
    if first_num is not None and first_num > 0:
        cand = toks[first_num - 1]
        if sum(_num(x) is None for x in cand) >= len(cand) / 2:
            header = cand
    data = toks[first_num:] if first_num is not None else []
    ncol = len(header) or max((len(t) for t in data), default=0)
    rows = []
    for t in data:
        if len(t) > ncol and not delim:  # whitespace-split name with spaces: merge leading text tokens
            k = len(t) - ncol + 1
            t = [" ".join(t[:k])] + t[k:]
        row = [(_num(x) if _num(x) is not None else x) for x in t[:ncol]]
        rows.append(row + [None] * (ncol - len(row)))
    if not header:
        header = [f"col{i + 1}" for i in range(ncol)]
    return header, rows


def find_col(header, *pats, exclude=()):
    """Index of the first column whose normalised name matches a pattern (exact first, then contains)."""
    H = [norm(h) for h in header]
    for p in pats:
        for i, h in enumerate(H):
            if h == p and not any(e in h for e in exclude):
                return i
    for p in pats:
        for i, h in enumerate(H):
            if p in h and not any(e in h for e in exclude):
                return i
    return None


def col(rows, i):
    return np.array([r[i] if (i is not None and i < len(r) and isinstance(r[i], (int, float))) else np.nan for r in rows], float)


# ============================================================================ LAS
def read_las(path) -> dict:
    """Minimal LAS 2.0 reader (uses lasio when installed). Returns dict(well, header, curves{mnem:(unit, array)}, index_unit)."""
    try:
        import lasio  # type: ignore
        L = lasio.read(str(path), ignore_header_errors=True)
        hdr = {it.mnemonic.upper(): it.value for it in L.well}
        hdr.update({it.mnemonic.upper(): it.value for it in L.params})
        curves = {c.mnemonic.upper(): (str(c.unit or ""), np.asarray(c.data, float)) for c in L.curves}
        return dict(well=str(hdr.get("WELL") or Path(path).stem), header=hdr, curves=curves,
                    index_unit=str(L.curves[0].unit or ""))
    except ImportError:
        pass
    sec, hdr, cur, data = None, {}, [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            sec = s[1].upper()
            continue
        if sec in ("W", "P") and "." in s and ":" in s:
            m = re.match(r"([^.\s]+)\s*\.(\S*)\s+(.*?):", s + " ")
            if m:
                hdr[m.group(1).upper()] = m.group(3).strip()
        elif sec == "C" and "." in s:
            m = re.match(r"([^.\s]+)\s*\.(\S*)", s)
            if m:
                cur.append((m.group(1).upper(), m.group(2)))
        elif sec == "A":
            data.extend(_num(t) for t in s.split())
    null = _num(str(hdr.get("NULL", "-999.25")).split()[0]) or -999.25
    n = len(cur)
    arr = np.array([np.nan if v is None else v for v in data[: (len(data) // n) * n]], float).reshape(-1, n) if n else np.zeros((0, 0))
    arr[np.isclose(arr, null)] = np.nan
    curves = {m: (u, arr[:, i]) for i, (m, u) in enumerate(cur)}
    return dict(well=str(hdr.get("WELL") or Path(path).stem), header=hdr, curves=curves, index_unit=cur[0][1] if cur else "")


def las_sonic(las: dict):
    """Return (depth_array, slowness_s_per_m, mnemonic, unit_note) or None."""
    curves = las["curves"]
    if not curves:
        return None
    idx_m, (idx_u, depth) = next(iter(curves.items()))
    base = {}
    for k in curves:                         # DTC_P, DTCO_1, DT.US/F ... -> base mnemonic
        b = re.sub(r"(_[A-Z0-9]{1,6})+$", "", k.upper().split(".")[0])
        base.setdefault(b, k)
    for m in SONIC_MNEM + ("DTCOM",):
        key = m if m in curves else base.get(m)
        if key:
            m = key
            unit, v = curves[m]
            v = np.where((v > 0) & np.isfinite(v), v, np.nan)
            if np.isfinite(v).sum() < 10:
                continue
            u = unit.lower().replace(" ", "")
            if "ft" in u or u in ("us/f", "usec/f", "us/ft"):
                per_m, note = v / FT, f"{m} [{unit}] (µs/ft)"
            elif "/m" in u:
                per_m, note = v, f"{m} [{unit}] (µs/m)"
            else:
                med = np.nanmedian(v)
                per_m, note = (v / FT, f"{m} [{unit or '?'}] assumed µs/ft") if med < 200 else (v, f"{m} [{unit or '?'}] assumed µs/m")
            d = depth * (FT if idx_u.lower().startswith("f") else 1.0)
            per_m = np.clip(per_m, 40 / FT, 240 / FT)  # plausible rock range 40-240 µs/ft
            return d, per_m * 1e-6, m, note
    return None


def _hdr_num(las: dict, *keys):
    for k in keys:
        v = las["header"].get(k)
        if v not in (None, ""):
            x = _num(str(v).split()[0])
            if x is not None:
                return x
    return None


def las_kb(las: dict):
    for k in ("EKB", "KB", "KBE", "EREF", "ELEV", "EDF", "RKB", "APD", "KBELEV"):
        v = las["header"].get(k)
        if v not in (None, ""):
            x = _num(str(v).split()[0])
            if x is not None:
                return x
    return None


# ============================================================================ deviation / datum
def tvd_from_dev(dev, md):
    """Interpolate TVD at MD from a deviation table (uses TVD column or minimum curvature)."""
    if dev is None:
        return np.asarray(md, float), "no deviation survey: assumed vertical (TVD = MD)"
    return np.interp(md, dev["md"], dev["tvd"]), dev["note"]


def read_deviation(path):
    h, rows = read_table(path)
    i_md = find_col(h, "md", "measureddepth", "depth")
    i_tvd = find_col(h, "tvd", "truevertical", exclude=("ss", "subsea"))
    i_tvdss = find_col(h, "tvdss", "subsea")
    i_inc = find_col(h, "inc", "incl", "inclination", "dev", "angle")
    i_az = find_col(h, "azi", "azim", "azimuth")
    if i_md is None:
        return None
    md = col(rows, i_md)
    ok = np.isfinite(md)
    if i_tvd is not None:
        tvd = col(rows, i_tvd)
        ok &= np.isfinite(tvd)
        o = np.argsort(md[ok])
        return dict(md=md[ok][o], tvd=tvd[ok][o], note="deviation survey (TVD column)")
    if i_inc is not None:
        inc = np.radians(col(rows, i_inc))
        az = np.radians(col(rows, i_az)) if i_az is not None else np.zeros_like(inc)
        ok &= np.isfinite(inc) & np.isfinite(az)
        md, inc, az = md[ok], inc[ok], az[ok]
        o = np.argsort(md)
        md, inc, az = md[o], inc[o], az[o]
        tvd = np.zeros_like(md)
        for k in range(1, len(md)):
            dmd = md[k] - md[k - 1]
            cosdl = np.cos(inc[k] - inc[k - 1]) - np.sin(inc[k - 1]) * np.sin(inc[k]) * (1 - np.cos(az[k] - az[k - 1]))
            dl = np.arccos(np.clip(cosdl, -1, 1))
            rf = 1.0 if dl < 1e-9 else 2 / dl * np.tan(dl / 2)
            tvd[k] = tvd[k - 1] + dmd / 2 * (np.cos(inc[k - 1]) + np.cos(inc[k])) * rf
        if md[0] > 0:
            tvd = tvd + md[0] * np.cos(inc[0])
        return dict(md=md, tvd=tvd, note="deviation survey (minimum curvature)")
    if i_tvdss is not None:
        return None
    return None


# ============================================================================ checkshot / markers
def read_checkshot(path):
    """-> dict(z_kind in {'tvdss','tvd','md'}, z, twt_ms, note) or None."""
    h, rows = read_table(path)
    i_t = find_col(h, "twt", "twotime", "twowaytime", "twtms", "time", "owt", "onewaytime", "t")
    i_ss = find_col(h, "tvdss", "subsea", "zss", "ss")
    i_tvd = find_col(h, "tvd", "truevertical", exclude=("ss",))
    i_md = find_col(h, "md", "measured")
    i_z = find_col(h, "depth", "z")
    if i_t is None:
        return None
    t = col(rows, i_t)
    name_t = norm(h[i_t])
    one_way = "owt" in name_t or "oneway" in name_t
    notes = []
    if i_ss is not None:
        z, kind = np.abs(col(rows, i_ss)), "tvdss"
    elif i_tvd is not None:
        z, kind = col(rows, i_tvd), "tvd"
    elif i_md is not None:
        z, kind = col(rows, i_md), "md"
    elif i_z is not None:
        z = col(rows, i_z)
        kind = "tvdss" if np.nanmedian(z) < 0 else "tvd"
        z = np.abs(z)
        notes.append(f"depth column '{h[i_z]}' read as {kind.upper()}")
    else:
        num = [i for i in range(len(h)) if i != i_t and np.isfinite(col(rows, i)).sum() > 2]
        if not num:
            return None
        z, kind = np.abs(col(rows, num[0])), "tvd"
        notes.append(f"column '{h[num[0]]}' assumed to be depth (TVD)")
    if np.nanmax(t) < 20:  # seconds
        t = t * 1000.0
        notes.append("time in seconds converted to ms")
    if one_way:
        t = t * 2.0
        notes.append("one-way time doubled to two-way time")
    if "ft" in " ".join(h).lower():
        z = z * FT
        notes.append("depth in feet converted to metres")
    ok = np.isfinite(t) & np.isfinite(z)
    o = np.argsort(z[ok])
    return dict(z_kind=kind, z=z[ok][o], twt=t[ok][o], note="; ".join(notes))


def read_markers(path):
    """-> list of dict(name, md, tvd, tvdss, twt) with None where unknown, plus column report."""
    h, rows = read_table(path)
    i_name = find_col(h, "marker", "surface", "name", "horizon", "top", "pick", "formation", "zone")
    i_md = find_col(h, "md", "measured")
    i_tvd = find_col(h, "tvd", "truevertical", exclude=("ss",))
    i_ss = find_col(h, "tvdss", "subsea", "zss")
    i_twt = find_col(h, "twt", "time", "owt")
    i_z = find_col(h, "depth", "z")
    if i_name is None:
        i_name = next((i for i in range(len(h)) if any(isinstance(r[i], str) for r in rows if i < len(r))), None)
    out = []
    for r in rows:
        g = lambda i: (float(r[i]) if i is not None and i < len(r) and isinstance(r[i], (int, float)) else None)
        name = str(r[i_name]).strip() if i_name is not None and r[i_name] is not None else "marker"
        m = dict(name=name, md=g(i_md), tvd=g(i_tvd), tvdss=(abs(g(i_ss)) if g(i_ss) is not None else None), twt=g(i_twt))
        if m["md"] is None and m["tvd"] is None and m["tvdss"] is None and g(i_z) is not None:
            m["md"] = g(i_z)
        if m["twt"] is not None:
            if i_twt is not None and "owt" in norm(h[i_twt]):
                m["twt"] *= 2
            if m["twt"] < 20:
                m["twt"] *= 1000
        out.append(m)
    cols = {k: (h[i] if i is not None else None) for k, i in
            dict(name=i_name, md=i_md, tvd=i_tvd, tvdss=i_ss, twt=i_twt).items()}
    return out, cols


def domain_of_markers(markers) -> str:
    has_t = any(m["twt"] is not None for m in markers)
    has_z = any(any(m[k] is not None for k in ("md", "tvd", "tvdss")) for m in markers)
    return "both" if has_t and has_z else "time" if has_t else "depth" if has_z else "unknown"


# ============================================================================ well list
def read_well_list(path):
    h, rows = read_table(path)
    i_n = find_col(h, "well", "wellname", "name", "uwi", "borehole")
    i_x = find_col(h, "x", "easting", "surfacex", "xcoord", "utmx", "east")
    i_y = find_col(h, "y", "northing", "surfacey", "ycoord", "utmy", "north")
    i_kb = find_col(h, "kb", "kbelev", "ekb", "rkb", "elevation", "datum", "kbm")
    if i_x is None or i_y is None:
        return None
    out = {}
    for r in rows:
        try:
            name = str(r[i_n if i_n is not None else 0]).strip()
            x, y = float(r[i_x]), float(r[i_y])
        except (TypeError, ValueError, IndexError):
            continue
        kb = None
        if i_kb is not None and isinstance(r[i_kb], (int, float)):
            kb = float(r[i_kb])
        out[norm(name)] = dict(name=name, x=x, y=y, kb=kb)
    return dict(wells=out, cols=dict(name=h[i_n] if i_n is not None else h[0], x=h[i_x], y=h[i_y],
                                     kb=h[i_kb] if i_kb is not None else None))


# ============================================================================ horizons
def read_horizon(path):
    """-> dict(il, xl, v (value), value_col, domain_hint, unit_note) for 'inline xline value' files."""
    h, rows = read_table(path)
    i_il = find_col(h, "inline", "il", "iline", "row")
    i_xl = find_col(h, "crossline", "xline", "xl", "cdp", "trace", "col")
    i_v = find_col(h, "twt", "time", "ms", "depth", "tvdss", "z", "value")
    num = [i for i in range(len(h)) if np.isfinite(col(rows[:200], i)).sum() > 0]
    if i_il is None or i_xl is None:
        if len(num) < 3:
            return None
        i_il, i_xl = num[0], num[1]
    if i_v is None or i_v in (i_il, i_xl):
        rest = [i for i in num if i not in (i_il, i_xl)]
        if not rest:
            return None
        i_v = rest[-1]
    il, xl, v = col(rows, i_il), col(rows, i_xl), col(rows, i_v)
    ok = np.isfinite(il) & np.isfinite(xl) & np.isfinite(v)
    il, xl, v = il[ok], xl[ok], v[ok]
    name = norm(h[i_v]) + " " + norm(Path(path).stem)
    if any(k in name for k in ("twt", "time", "ms", "owt")):
        hint = "time"
    elif any(k in name for k in ("depth", "tvdss", "tvd", "zss", "meter", "metre")):
        hint = "depth"
    else:
        hint = "unknown"
    return dict(il=il, xl=xl, v=v, value_col=h[i_v], domain_hint=hint, header=h)


# ============================================================================ time <-> depth functions
class TD:
    """Monotonic TVDSS (m) <-> TWT (ms) relation with linear extrapolation using end velocities."""

    def __init__(self, z, t, label):
        z, t = np.asarray(z, float), np.asarray(t, float)
        ok = np.isfinite(z) & np.isfinite(t)
        z, t = z[ok], t[ok]
        o = np.argsort(z)
        z, t = z[o], np.maximum.accumulate(t[o])
        if z[0] > 0:  # anchor at datum
            z, t = np.r_[0.0, z], np.r_[0.0, t]
        keep = np.r_[True, (np.diff(z) > 1e-6) & (np.diff(t) > 1e-9)]
        self.z, self.t, self.label = z[keep], t[keep], label

    def _v(self, a, b):  # interval velocity m/s between indices
        dz, dt = self.z[b] - self.z[a], (self.t[b] - self.t[a]) / 2000.0
        return dz / dt if dt > 0 else 2000.0

    def twt(self, z):
        z = np.asarray(z, float)
        out = np.interp(z, self.z, self.t)
        hi = z > self.z[-1]
        if hi.any():
            out[hi] = self.t[-1] + 2000.0 * (z[hi] - self.z[-1]) / self._v(-2, -1)
        return out

    def depth(self, t):
        t = np.asarray(t, float)
        out = np.interp(t, self.t, self.z)
        hi = t > self.t[-1]
        if hi.any():
            out[hi] = self.z[-1] + (t[hi] - self.t[-1]) / 2000.0 * self._v(-2, -1)
        return out


def td_constant(v):
    return TD([0, 20000], [0, 2000.0 * 20000 / v], f"constant average velocity {v:,.0f} m/s")


def td_from_sonic(z_tvdss, slow, v_rep):
    ok = np.isfinite(z_tvdss) & np.isfinite(slow)
    z, s = z_tvdss[ok], slow[ok]
    o = np.argsort(z)
    z, s = z[o], s[o]
    t0 = 2000.0 * z[0] / v_rep
    t = t0 + 2000.0 * np.r_[0.0, np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(z))]
    return TD(z, t, f"integrated sonic, replacement velocity {v_rep:,.0f} m/s above {z[0]:,.0f} m")


class VelocityVolume:
    """Interval velocity SEG-Y (m/s) in the TIME domain; depth(t) by integration at the nearest trace."""

    def __init__(self, path, il_byte=189, xl_byte=193):
        import segyio
        f = segyio.open(str(path), "r", ignore_geometry=True)
        f.mmap()
        self.f = f
        self.dt = segyio.tools.dt(f) / 1000.0
        self.t0 = float(f.samples[0])
        self.ns = len(f.samples)
        self.il = np.asarray(f.attributes(il_byte)[:], np.int64)
        self.xl = np.asarray(f.attributes(xl_byte)[:], np.int64)
        sc = np.asarray(f.attributes(segyio.TraceField.SourceGroupScalar)[:], float)
        sc = np.where(sc == 0, 1.0, np.where(sc < 0, -1.0 / np.where(sc == 0, 1, sc), sc))
        self.x = np.asarray(f.attributes(segyio.TraceField.CDP_X)[:], float) * sc
        self.y = np.asarray(f.attributes(segyio.TraceField.CDP_Y)[:], float) * sc
        self.key = {int(a) * 10_000_000 + int(b): i for i, (a, b) in enumerate(zip(self.il, self.xl))}
        v = np.asarray(f.trace.raw[0], float)
        if np.nanmedian(v) < 300 or np.nanmedian(v) > 10000:
            raise ValueError("velocity volume values don't look like m/s (median %.0f)" % np.nanmedian(v))

    def td_at_trace(self, i, label):
        v = np.asarray(self.f.trace.raw[int(i)], float)
        t = self.t0 + np.arange(self.ns) * self.dt
        z = np.cumsum(v * self.dt / 2000.0) + v[0] * self.t0 / 2000.0
        return TD(z, t, label)

    def td_at_xy(self, x, y):
        i = int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))
        d = math.hypot(self.x[i] - x, self.y[i] - y)
        return self.td_at_trace(i, f"seismic velocity volume (trace {d:,.0f} m from well)")


# ============================================================================ folder scan
def classify(p: Path) -> str:
    n = p.name.lower()
    e = p.suffix.lower()
    if e in SEGY_EXT:
        return "seismic"
    if e == ".las":
        return "las"
    if re.search(r"checkshot|chk|\btdr\b|time[_\- ]?depth|_cs\b", n):
        return "checkshot"
    if re.search(r"deviation|\bdev\b|_dev|survey|trajectory", n):
        return "deviation"
    if re.search(r"marker|tops|picks", n):
        return "markers"
    if re.search(r"well[_\- ]?list|well[_\- ]?info|wellhead|well[_\- ]?header|location|coordinates", n):
        return "welllist"
    if re.search(r"horizon|hrz|surface|grid|_hor\b|map", n):
        return "horizon"
    if e in (".xlsx", ".xls", ".csv"):
        return "table"
    if e in TABLE_EXT:
        return "table"
    return "other"


def seismic_domain_hint(path: Path, text_header: str) -> tuple[str, str]:
    s = (path.name + " " + text_header).lower()
    if re.search(r"\bdepth\b|psdm.*depth|_dz\b|_z\b|tvdss|in depth|depth domain|metres|meters", s):
        return "depth", "file name or text header mentions depth"
    if re.search(r"\btime\b|twt|\bms\b|pstm|msec|time domain|milliseconds", s):
        return "time", "file name or text header mentions time"
    return "unknown", "no domain information found in name or header"


def inspect_segy(path: Path) -> dict:
    import segyio
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        try:
            txt = segyio.tools.wrap(f.text[0])
        except Exception:
            txt = ""
        dt = segyio.tools.dt(f) / 1000.0
        d, why = seismic_domain_hint(path, txt)
        return dict(file=path.name, path=str(path), size_mb=round(path.stat().st_size / 1e6, 1), traces=int(f.tracecount),
                    samples=len(f.samples), sample_interval=dt, start=float(f.samples[0]),
                    end=float(f.samples[0] + (len(f.samples) - 1) * dt), domain_hint=d, domain_reason=why)


def scan_folder(root: str | Path, max_files=400) -> dict:
    root = Path(root)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and len(p.relative_to(root).parts) <= 4:
            files.append(p)
        if len(files) >= max_files:
            break
    out = dict(folder=str(root), seismic=[], horizons=[], wells={}, well_list=None, other=[], notes=[])
    W = out["wells"]

    def w(key, label):
        return W.setdefault(key, dict(name=label, key=key, las=[], checkshot=None, deviation=None, markers=None,
                                      sonic=None, location=None, kb=None))

    tables = []
    for p in files:
        k = classify(p)
        try:
            if k == "seismic":
                out["seismic"].append(inspect_segy(p))
            elif k == "las":
                las = read_las(p)
                key = well_key(las["well"]) or well_key(p.name)
                if key not in W and well_key(p.name) in W:
                    key = well_key(p.name)
                rec = w(key, well_label(las["well"] if las["well"] else p.name))
                son = las_sonic(las)
                idx = next(iter(las["curves"].values()))[1] if las["curves"] else np.array([])
                rec["las"].append(dict(file=p.name, path=str(p), curves=list(las["curves"].keys())[:40],
                                       sonic=son[3] if son else None,
                                       depth_range=[float(np.nanmin(idx)), float(np.nanmax(idx))] if idx.size else None,
                                       index_unit=las["index_unit"]))
                if son and not rec["sonic"]:
                    rec["sonic"] = dict(file=p.name, path=str(p), curve=son[3])
                kb = las_kb(las)
                if kb is not None and rec["kb"] is None:
                    rec["kb"] = kb
                hx, hy = _hdr_num(las, "XWELL", "X", "XCOORD", "EAST"), _hdr_num(las, "YWELL", "Y", "YCOORD", "NORTH")
                if hx is not None and hy is not None and not rec["location"]:
                    rec["location"] = dict(x=hx, y=hy, source="LAS header")
            elif k == "checkshot":
                cs = read_checkshot(p)
                rec = w(well_key(p.name), well_label(p.name))
                rec["checkshot"] = dict(file=p.name, path=str(p), ok=cs is not None, points=len(cs["z"]) if cs else 0,
                                        depth_kind=cs["z_kind"] if cs else None, note=cs["note"] if cs else "could not read time/depth columns")
            elif k == "deviation":
                dv = read_deviation(p)
                rec = w(well_key(p.name), well_label(p.name))
                rec["deviation"] = dict(file=p.name, path=str(p), ok=dv is not None, note=dv["note"] if dv else "could not read MD/TVD or MD/INC/AZI")
            elif k == "markers":
                mk, cols = read_markers(p)
                rec = w(well_key(p.name), well_label(p.name))
                rec["markers"] = dict(file=p.name, path=str(p), count=len(mk), names=[m["name"] for m in mk][:30],
                                      domain=domain_of_markers(mk), columns=cols)
            elif k in ("welllist", "table", "horizon"):
                tables.append((k, p))
            else:
                out["other"].append(p.name)
        except Exception as e:  # never fail the whole scan on one file
            out["notes"].append(f"{p.name}: could not read ({type(e).__name__}: {e})")

    for k, p in tables:
        try:
            wl = read_well_list(p) if k in ("welllist", "table") else None
            if wl and wl["wells"]:
                out["well_list"] = dict(file=p.name, path=str(p), count=len(wl["wells"]), columns=wl["cols"])
                for key, v in wl["wells"].items():
                    rec = w(key, v["name"])
                    rec["location"] = dict(x=v["x"], y=v["y"], source=p.name)
                    if v["kb"] is not None:
                        rec["kb"] = v["kb"]
                continue
            hz = read_horizon(p) if k in ("horizon", "table") else None
            if hz is not None and len(hz["v"]) > 3:
                out["horizons"].append(dict(file=p.name, path=str(p), points=int(len(hz["v"])), value_col=hz["value_col"],
                                            domain_hint=hz["domain_hint"], vmin=float(hz["v"].min()), vmax=float(hz["v"].max()),
                                            il=[float(hz["il"].min()), float(hz["il"].max())],
                                            xl=[float(hz["xl"].min()), float(hz["xl"].max())]))
            else:
                out["other"].append(p.name)
        except Exception as e:
            out["notes"].append(f"{p.name}: could not read ({type(e).__name__}: {e})")

    for rec in W.values():
        rec["conversion"] = ("checkshot" if rec["checkshot"] and rec["checkshot"]["ok"] else
                             "sonic" if rec["sonic"] else "seismic velocity")
    out["wells"] = sorted(W.values(), key=lambda r: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", r["name"])])
    m_dom = {r["markers"]["domain"] for r in out["wells"] if r["markers"]}
    out["markers_domain_hint"] = (m_dom.pop() if len(m_dom) == 1 else "mixed" if m_dom else None)
    return out


# ============================================================================ wells -> seismic domain
def _depth_to_tvdss(z, kind, dev, kb, notes, what):
    """Bring depths of a given kind (md / tvd / tvdss) to TVDSS (m)."""
    z = np.asarray(z, float)
    if kind == "tvdss":
        return z
    if kind == "md":
        z, n = tvd_from_dev(dev, z)
        if n.startswith("no deviation"):
            notes.add(f"{what}: {n}")
    if kb is None:
        notes.add(f"{what}: no KB elevation found, so TVD is treated as TVDSS (datum at KB)")
        return z
    return z - kb


def prepare_wells(scan: dict, seismic_domain: str, markers_domain: str, v_replacement=2000.0,
                  v_average: Optional[float] = None, velocity_path: Optional[str] = None,
                  prefer: str = "auto") -> dict:
    """Markers of every well brought into the seismic domain, with the method and assumptions used.

    prefer: 'auto' (checkshot -> sonic -> seismic velocity), or force 'checkshot' / 'sonic' / 'velocity'."""
    vel = None
    warnings = []
    if velocity_path:
        try:
            vel = VelocityVolume(velocity_path)
        except Exception as e:
            warnings.append(f"Velocity volume not used: {e}")
    out = []
    for rec in scan["wells"]:
        notes: set[str] = set()
        res = dict(name=rec["name"], x=None, y=None, kb=rec.get("kb"), method=None, method_note=None,
                   markers=[], notes=[], status="ok")
        if rec.get("location"):
            res["x"], res["y"] = rec["location"]["x"], rec["location"]["y"]
        dev = read_deviation(rec["deviation"]["path"]) if rec.get("deviation") and rec["deviation"]["ok"] else None
        kb = rec.get("kb")
        mk = read_markers(rec["markers"]["path"])[0] if rec.get("markers") else []

        # ---- time-depth relation for this well (only needed if domains differ)
        need_td = bool(mk) and markers_domain != seismic_domain
        td = None
        if need_td:
            order = {"auto": ["checkshot", "sonic", "velocity"], "checkshot": ["checkshot"], "sonic": ["sonic"],
                     "velocity": ["velocity"]}[prefer]
            for m in order:
                try:
                    if m == "checkshot" and rec.get("checkshot") and rec["checkshot"]["ok"]:
                        cs = read_checkshot(rec["checkshot"]["path"])
                        z = _depth_to_tvdss(cs["z"], cs["z_kind"], dev, kb, notes, "checkshot")
                        td = TD(z, cs["twt"], f"checkshot ({rec['checkshot']['file']})")
                        if cs["note"]:
                            notes.add("checkshot: " + cs["note"])
                    elif m == "sonic" and rec.get("sonic"):
                        las = read_las(rec["sonic"]["path"])
                        d, slow, mn, unote = las_sonic(las)
                        z = _depth_to_tvdss(d, "md", dev, kb, notes, "sonic")
                        td = td_from_sonic(z, slow, v_replacement)
                        notes.add(f"sonic: {unote}")
                    elif m == "velocity":
                        if vel is not None and res["x"] is not None:
                            td = vel.td_at_xy(res["x"], res["y"])
                        elif v_average:
                            td = td_constant(v_average)
                    if td is not None:
                        res["method"], res["method_note"] = m, td.label
                        break
                except Exception as e:
                    notes.add(f"{m} failed: {e}")
            if td is None:
                res["status"] = "no-conversion"
                notes.add("no checkshot, usable sonic or seismic velocity: markers can't be shown in the seismic domain")

        for m in mk:
            r = dict(name=m["name"], md=m["md"], tvdss=None, twt=m["twt"], value=None)
            if m["tvdss"] is not None:
                r["tvdss"] = m["tvdss"]
            elif m["tvd"] is not None:
                r["tvdss"] = float(_depth_to_tvdss([m["tvd"]], "tvd", dev, kb, notes, "markers")[0])
            elif m["md"] is not None:
                r["tvdss"] = float(_depth_to_tvdss([m["md"]], "md", dev, kb, notes, "markers")[0])
            if seismic_domain == "time":
                if markers_domain == "time" and r["twt"] is not None:
                    r["value"] = r["twt"]
                elif td is not None and r["tvdss"] is not None:
                    r["value"] = r["twt"] = float(td.twt([r["tvdss"]])[0])
            else:
                if markers_domain == "depth" and r["tvdss"] is not None:
                    r["value"] = r["tvdss"]
                elif td is not None and r["twt"] is not None:
                    r["value"] = r["tvdss"] = float(td.depth([r["twt"]])[0])
            res["markers"].append(r)
        if not mk:
            notes.add("no markers file for this well")
        if res["x"] is None:
            notes.add("no location found (add a well list with X/Y)")
            if res["status"] == "ok":
                res["status"] = "no-location"
        res["notes"] = sorted(notes)
        out.append(res)
    def _order(n):  # shallow to deep (by median converted value, else by depth)
        v = [m["value"] if m["value"] is not None else m["tvdss"] for w in out for m in w["markers"] if m["name"] == n]
        v = [x for x in v if x is not None]
        return (float(np.median(v)) if v else float("inf"), n)
    names = sorted({m["name"] for w in out for m in w["markers"]}, key=_order)
    return dict(seismic_domain=seismic_domain, markers_domain=markers_domain, wells=out, marker_names=names,
                warnings=warnings, unit="ms" if seismic_domain == "time" else "m")


def field_td(scan, wells_dev_kb=True):
    """Average TVDSS->TWT relation from all checkshots in a scan (for horizon conversion)."""
    zs, ts = [], []
    used = []
    for rec in scan["wells"]:
        if rec.get("checkshot") and rec["checkshot"]["ok"]:
            cs = read_checkshot(rec["checkshot"]["path"])
            dev = read_deviation(rec["deviation"]["path"]) if rec.get("deviation") and rec["deviation"]["ok"] else None
            z = _depth_to_tvdss(cs["z"], cs["z_kind"], dev, rec.get("kb"), set(), "checkshot")
            zs.append(z)
            ts.append(cs["twt"])
            used.append(rec["name"])
    if not zs:
        return None
    grid = np.linspace(0, max(z.max() for z in zs), 200)
    curves = [TD(z, t, "").twt(grid) for z, t in zip(zs, ts)]
    return TD(grid, np.median(curves, axis=0), f"field average of {len(used)} checkshot(s): {', '.join(used)}")


def prepare_horizon(path, horizon_domain, seismic_domain, method="checkshots", scan=None, v_average=None,
                    velocity_path=None, out_path=None, il_byte=189, xl_byte=193) -> dict:
    hz = read_horizon(path)
    if hz is None:
        raise ValueError("Could not read the horizon: expected columns inline, crossline and a value.")
    v = hz["v"].copy()
    notes = []
    if horizon_domain == "time" and np.nanmax(v) < 20:
        v *= 1000
        notes.append("times in seconds converted to ms")
    if horizon_domain == "depth":
        if np.nanmedian(v) < 0:
            notes.append("negative depths read as TVDSS (sign flipped)")
        v = np.abs(v)
    label = "no conversion needed (same domain as the seismic)"
    if horizon_domain != seismic_domain:
        if method == "velocity" and velocity_path:
            vol = VelocityVolume(velocity_path, il_byte, xl_byte)
            res = np.full_like(v, np.nan)
            keys = (hz["il"].astype(np.int64) * 10_000_000 + hz["xl"].astype(np.int64))
            cache = {}
            for j, k in enumerate(keys):
                i = vol.key.get(int(k))
                if i is None:
                    continue
                td = cache.get(i) or cache.setdefault(i, vol.td_at_trace(i, ""))
                res[j] = td.twt([v[j]])[0] if seismic_domain == "time" else td.depth([v[j]])[0]
            miss = np.isnan(res).mean()
            if miss > 0:
                notes.append(f"{miss * 100:.0f}% of horizon points have no velocity trace and were dropped")
            v = res
            label = "seismic velocity volume (per trace)"
        else:
            if method == "constant":
                if not v_average:
                    raise ValueError("Enter an average velocity (m/s) to convert the horizon.")
                td = td_constant(float(v_average))
            else:
                td = field_td(scan) if scan else None
                if td is None:
                    raise ValueError("No checkshots available to build a field time-depth relation. "
                                     "Choose 'constant average velocity' or a velocity volume instead.")
            v = td.twt(v) if seismic_domain == "time" else td.depth(v)
            label = td.label
    ok = np.isfinite(v)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write("inline xline value\n")
            for a, b, c in zip(hz["il"][ok], hz["xl"][ok], v[ok]):
                fh.write(f"{int(a)} {int(b)} {c:.2f}\n")
    unit = "ms" if seismic_domain == "time" else "m"
    return dict(points=int(ok.sum()), method=label, notes=notes, unit=unit,
                vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)), median=float(np.nanmedian(v)),
                out_path=str(out_path) if out_path else None)
