"""Well trajectories, markers and checks, all in TWT (the seismic's domain), for the maps and the 3D viewer.

Same method as the 4D ChronoSeis well-overlay helper (team notebook, cell "WELL OVERLAY HELPER"):
  * wellhead X/Y from the LAS header, at TVDSS = -KB
  * deviation survey MDKB, TVD (below KB), inclination, azimuth  ->  minimum curvature -> east/north offsets
  * checkshot TVDSS (m), TWT (ms)  ->  TWT along the path, sampled every 5 m of TVDSS within the checkshot range
  * no deviation file -> vertical well; no checkshot -> no path (it cannot be placed in TWT), wellhead only
Differences: any number of wells and any well names (not only Well-1..8), and KB read from the LAS header
(KB / EKB / EDF / ...) with the notebook's KB table as fallback for Well-1..8.

Everything is placed in two-way time so wells can be drawn on the seismic in 2D and 3D:
  * horizontal: wellhead X/Y (LAS header, else the well list), deviation survey by minimum curvature
  * vertical:   KB/RT (well list > LAS EREF > LAS EKB > team notebook table), TVDSS = TVD - KB, then the checkshot
                (TVDSS -> TWT).  Above the first checkshot the time-depth is anchored at MSL (0 ms) and the seabed
                (water velocity 1500 m/s); below the last checkshot it continues at the last interval velocity.
                These extrapolated parts are flagged.  Markers (MDKB) go MD -> TVD -> TVDSS -> TWT the same way.
  * checks:     bottom-hole offset vs the LAS XOFFSET/YOFFSET_BOTTOM, KB sources, X/Y sources, survey vs TVD column,
                checkshot monotonic, wellhead / path inside the seismic X/Y footprint, TWT inside the seismic time range.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

STEP_M = 5.0
WATER_V = 1500.0          # m/s, water column (two-way time to the seabed)
MAX_TAIL_M = 250.0         # extrapolate at most this far below the last checkshot
# KB elevations typed into the team notebook (WELL_OVERLAY_KB_M), used only when the LAS header has no KB.
NOTEBOOK_KB = {1: 9.450, 2: 22.000, 3: 22.000, 4: 21.978, 5: 22.000, 6: 22.000, 7: 29.000, 8: 29.800}
AUX = re.compile(r"[_-](checkshot|deviation|dev|cs|survey)\b", re.I)
KB_KEYS = ("KB", "EKB", "KBE", "ELEV_KB", "KB_ELEV", "EDF", "DFE", "ELEV", "EREF", "APD")
X_KEYS = ("XWELL", "X", "XCOORD", "EAST", "X_LOC", "XLOC")
Y_KEYS = ("YWELL", "Y", "YCOORD", "NORTH", "Y_LOC", "YLOC")


def _header(p: Path) -> dict:
    hdr = {}
    with open(p, errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 400:
                break
            s = line.strip()
            if s.upper().startswith("~A"):
                break
            m = re.match(r"([^.\s]+)\s*\.(\S*)\s+(.*?):", s + " ")
            if m:
                hdr[m.group(1).upper()] = m.group(3).strip()
    return hdr


def _num(hdr: dict, keys) -> float | None:
    for k in keys:
        try:
            return float(str(hdr[k]).split()[0])
        except (KeyError, ValueError, IndexError):
            continue
    return None


def _numeric(p: Path) -> np.ndarray:
    """Numeric rows of a text or LAS table (same rules as the notebook's _numeric_file)."""
    lines = p.read_text(errors="replace").splitlines()
    if p.suffix.lower() == ".las":
        start = next((i for i, l in enumerate(lines) if l.strip().upper().startswith("~A")), None)
        if start is None:
            raise ValueError(f"no ~ASCII section in {p.name}")
        lines = lines[start + 1:]
    vals = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith(("#", "~")):
            continue
        try:
            row = [float(x) for x in line.replace(",", " ").split()]
        except ValueError:
            continue
        if len(row) >= 2 and np.isfinite(row[:2]).all() and -999.25 not in row:
            vals.append(row)
    if not vals:
        raise ValueError(f"no numeric rows in {p.name}")
    w = min(map(len, vals))
    a = np.array([r[:w] for r in vals], float)
    a = a[np.argsort(a[:, 0])]
    return a[np.r_[True, np.diff(a[:, 0]) > 0]]


def min_curvature(md, inc_deg, azi_deg):
    inc, azi = np.deg2rad(inc_deg), np.deg2rad(azi_deg)
    e, n = np.zeros(len(md)), np.zeros(len(md))
    for k in range(1, len(md)):
        dog = np.arccos(np.clip(np.cos(inc[k - 1]) * np.cos(inc[k]) +
                                np.sin(inc[k - 1]) * np.sin(inc[k]) * np.cos(azi[k] - azi[k - 1]), -1, 1))
        rf = 1.0 if dog < 1e-9 else 2 * np.tan(dog / 2) / dog
        dm = md[k] - md[k - 1]
        e[k] = e[k - 1] + dm / 2 * (np.sin(inc[k - 1]) * np.sin(azi[k - 1]) + np.sin(inc[k]) * np.sin(azi[k])) * rf
        n[k] = n[k - 1] + dm / 2 * (np.sin(inc[k - 1]) * np.cos(azi[k - 1]) + np.sin(inc[k]) * np.cos(azi[k])) * rf
    return e, n



def min_curvature_v(md, inc_deg, azi_deg):
    """East, north and TVD offsets (minimum curvature) from the first station."""
    inc, azi = np.deg2rad(inc_deg), np.deg2rad(azi_deg)
    e, n, v = np.zeros(len(md)), np.zeros(len(md)), np.zeros(len(md))
    for k in range(1, len(md)):
        dog = np.arccos(np.clip(np.cos(inc[k - 1]) * np.cos(inc[k]) +
                                np.sin(inc[k - 1]) * np.sin(inc[k]) * np.cos(azi[k] - azi[k - 1]), -1, 1))
        rf = 1.0 if dog < 1e-9 else 2 * np.tan(dog / 2) / dog
        dm = md[k] - md[k - 1]
        e[k] = e[k - 1] + dm / 2 * (np.sin(inc[k - 1]) * np.sin(azi[k - 1]) + np.sin(inc[k]) * np.sin(azi[k])) * rf
        n[k] = n[k - 1] + dm / 2 * (np.sin(inc[k - 1]) * np.cos(azi[k - 1]) + np.sin(inc[k]) * np.cos(azi[k])) * rf
        v[k] = v[k - 1] + dm / 2 * (np.cos(inc[k - 1]) + np.cos(inc[k])) * rf
    return e, n, v



# ----------------------------------------------------------------------------- file discovery
def _wkey(s: str) -> str:
    m = re.search(r"well[-_ ]?0*(\d+)", s, re.I)
    return f"well-{int(m.group(1))}" if m else re.split(r"[_\s]", s.strip().lower())[0]


FILE_TYPES = [("las", "LAS log"), ("las_alt", "LAS log (alternative: composite / interpreted)"), ("deviation", "Deviation survey"),
              ("checkshot", "Checkshot (depth–time)"), ("markers", "Markers (tops)"), ("welllist", "Well list"),
              ("other", "Other / not used"), ("ignore", "Ignore")]
_TEXT_EXT = (".txt", ".csv", ".dat", ".tsv", ".asc", ".prn")


def _head_lines(p: Path, n: int = 80) -> list[str]:
    try:
        with open(p, errors="replace") as fh:
            return [next(fh).rstrip("\n") for _ in range(n)]
    except StopIteration:
        pass
    except OSError:
        return []
    with open(p, errors="replace") as fh:
        return [l.rstrip("\n") for l in fh][:n]


def _sniff_text(lines: list[str]) -> tuple[str | None, str, str | None]:
    """-> (type from content, why, well named in a '#WELL-2' comment)."""
    com = [l.lstrip("#").strip() for l in lines if l.startswith("#")]
    lab = " ".join(com).lower()
    well = None
    for c in com[:2]:
        if re.search(r"well", c, re.I):
            well = _wkey(c)
            break
    rows, texty = [], 0
    for l in lines:
        if not l.strip() or l.startswith("#"):
            continue
        t = re.split(r"[\s,;]+", l.strip())
        nums = []
        for x in t:
            try:
                nums.append(float(x))
            except ValueError:
                break
        if len(nums) == len(t) and t:
            rows.append(nums)
        elif nums == [] or len(nums) < len(t):
            texty += 1
    w = min((len(r) for r in rows), default=0)
    if "checkshot" in lab or ("twt" in lab and ("tvd" in lab or "md" in lab)):
        return "checkshot", "header comments name TVDSS/TWT columns", well
    if "incl" in lab or "azim" in lab:
        return "deviation", "header comments name inclination/azimuth columns", well
    if "marker" in lab or "tops" in lab:
        return "markers", "header comments say markers", well
    if rows and w >= 4:
        a = np.array([r[:4] for r in rows])
        if (a[:, 2] >= 0).all() and (a[:, 2] <= 180).all() and (a[:, 3] >= 0).all() and (a[:, 3] <= 360).all() and np.all(np.diff(a[:, 0]) >= 0):
            return "deviation", "4 numeric columns shaped like MD, TVD, inclination, azimuth", well
    if rows and w == 2 and not texty:
        a = np.array([r[:2] for r in rows])
        if np.all(np.diff(a[:, 0]) > 0) and np.all(np.diff(a[:, 1]) > 0) and a[:, 1].max() > a[:, 0].min():
            return "checkshot", "two increasing numeric columns (depth, time)", well
    if texty and len(rows) == 0 and any(re.match(r"^\S+(\s+\S+)*\s+[-\d.]+\s*$", l) for l in lines if l.strip() and not l.startswith("#")):
        return "markers", "lines of 'name  depth'", well
    if texty >= 1 and any(re.match(r"^[A-Za-z][^\t]*\s+[-\d.]+(\s+[-\d.]+)?\s*$", l) for l in lines if l.strip() and not l.startswith("#")):
        return "markers", "lines of 'name  depth'", well
    return None, "", well


def classify_file(p: Path) -> dict:
    """Guess what a file is and which well it belongs to. -> dict(type, well, conf, why)."""
    n, ext = p.name.lower(), p.suffix.lower()
    why, by_name, by_content, well, wsrc = [], None, None, None, ""
    if ext == ".las":
        alt = bool(re.search(r"composite|interp", n) or AUX.search(p.stem))
        hdr = _header(p)
        m = re.search(r"well[-_ ]?0*(\d+)", p.stem, re.I)
        if m:
            well, wsrc = _wkey(p.stem), "file name"
        elif hdr.get("WELL"):
            well, wsrc = _wkey(hdr["WELL"]), "LAS header WELL"
        else:
            well, wsrc = _wkey(p.stem), "file name"
        if alt:
            why.append("name says composite / interpreted: kept as an alternative LAS")
        why.append("LAS file")
        return dict(type="las_alt" if alt else "las", well=well, conf="high", why="; ".join(why) + (f"; well from {wsrc}" if well else ""))
    if "checkshot" in n or re.search(r"[_-]cs\b", n):
        by_name = "checkshot"
    elif "deviation" in n or "survey" in n or re.search(r"[_-]dev\b", n):
        by_name = "deviation"
    elif "marker" in n or re.search(r"\btops?\b|picks?", n):
        by_name = "markers"
    if ext in (".xlsx", ".xlsm", ".xls") or (ext == ".csv" and "well" in n and re.search(r"list|information|info|summary", n)):
        if read_well_list(p):
            return dict(type="welllist", well=None, conf="high", why="table with Well Name, X, Y columns")
        if ext != ".csv":
            return dict(type="other", well=None, conf="medium", why="spreadsheet without Well Name / X / Y columns")
    cwell = None
    if ext in _TEXT_EXT:
        by_content, cwhy, cwell = _sniff_text(_head_lines(p))
        if cwhy:
            why.append(cwhy)
    m = re.search(r"well[-_ ]?0*(\d+)", p.stem, re.I)
    if m:
        well, wsrc = _wkey(p.stem), "file name"
    elif cwell:
        well, wsrc = cwell, "header comment"
    elif by_name or by_content:
        tok = re.split(r"[_\s]", p.stem.strip())[0]
        if tok and not re.fullmatch(r"(checkshot|deviation|survey|markers?|tops?|picks?)", tok, re.I):
            well, wsrc = tok.lower(), "file name"
    if by_name and by_content and by_name == by_content:
        t, conf = by_name, "high"
    elif by_name and by_content:
        t, conf = by_content, "low"; why.append(f"file name says {by_name} but the content looks like {by_content}")
    elif by_name or by_content:
        t, conf = by_name or by_content, "medium"; why.append("file name" if by_name else "content only")
    else:
        t, conf = "other", "low"; why.append("not recognised")
    if t in ("checkshot", "deviation", "markers") and not well:
        conf = "low" if conf == "high" else conf
    return dict(type=t, well=well, conf=conf, why="; ".join(why) + (f"; well from {wsrc}" if wsrc else ""))


def _scan(folder: Path, max_depth: int = 4, overrides: dict | None = None) -> dict:
    """Classify every file under the well folder (flat or sub-folders) and group them per well.
    overrides = {relative path: {"type": ..., "well": ...}} from the user's corrections."""
    overrides = overrides or {}
    out = dict(las={}, cs={}, dv={}, mk={}, lists=[], files=[])
    base = len(folder.parts)
    items = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        if len(Path(root).parts) - base >= max_depth:
            dirs[:] = []
        for fn in sorted(files):
            if fn.startswith((".", "~$")):
                continue
            p = Path(root) / fn
            if p.suffix.lower() not in (".las", ".xlsx", ".xlsm", ".xls") + _TEXT_EXT:
                continue
            rel = p.relative_to(folder).as_posix()
            info = classify_file(p)
            ov = overrides.get(rel) or {}
            t = ov.get("type") or info["type"]
            wk = (None if ov.get("well") == "-" else ov["well"]) if ov.get("well") else info["well"]
            out["files"].append(dict(path=rel, name=fn, type=info["type"], well=info["well"], conf=info["conf"], why=info["why"],
                                     type_final=t, well_final=wk, overridden=bool(ov)))
            items.append((bool(ov), p, t, wk))
    for _, p, t, wk in sorted(items, key=lambda x: not x[0]):      # corrected files first so they win over guesses
        if t == "las" and wk:
            out["las"].setdefault(wk, p)
        elif t == "checkshot" and wk:
            out["cs"].setdefault(wk, p)
        elif t == "deviation" and wk:
            out["dv"].setdefault(wk, p)
        elif t == "markers" and wk:
            out["mk"].setdefault(wk, p)
        elif t == "welllist":
            out["lists"].append(p)
    return out


def read_well_list(path) -> dict:
    """Well list table (Excel or CSV): name, X, Y, RT/KB, WD, WB, TD ... -> {well key: record}. {} if unreadable."""
    import pandas as pd
    p = Path(path)
    try:
        if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            rows = [list(r) for r in wb.worksheets[0].iter_rows(min_row=1, max_row=80, max_col=26, values_only=True)]
            raw = pd.DataFrame(rows)
        else:
            raw = pd.read_csv(p, header=None, sep=None, engine="python", nrows=80)
    except Exception:
        return {}
    hdr = None
    for i in range(len(raw)):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if any(c in ("x", "easting", "x (m)") for c in cells) and any(c in ("y", "northing", "y (m)") for c in cells) and any("well" in c for c in cells):
            hdr = i
            break
    if hdr is None:
        return {}
    cols = [str(c).strip().lower() for c in raw.iloc[hdr].tolist()]
    def ix(*names):
        for nme in names:
            for j, c in enumerate(cols):
                if c == nme or c.startswith(nme + " "):
                    return j
        return None
    i_n, i_x, i_y = ix("well name", "well", "name"), ix("x", "easting"), ix("y", "northing")
    i_rt, i_wd, i_wb, i_td = ix("rt", "kb", "rkb", "ekb"), ix("wd", "water depth"), ix("wb"), ix("td")
    out = {}
    for r in range(hdr + 1, len(raw)):
        row = raw.iloc[r].tolist()
        name = str(row[i_n]).strip() if i_n is not None else ""
        def g(j):
            try:
                v = float(row[j]); return v if np.isfinite(v) else None
            except (TypeError, ValueError, IndexError):
                return None
        if not name or name.lower() == "nan" or g(i_x) is None or g(i_y) is None:
            continue
        out[_wkey(name)] = dict(name=name, x=g(i_x), y=g(i_y), rt=g(i_rt) if i_rt is not None else None,
                                wd=g(i_wd) if i_wd is not None else None, wb=g(i_wb) if i_wb is not None else None,
                                td=g(i_td) if i_td is not None else None)
        i_ty = ix("well types", "well type", "type")
        if i_ty is not None and str(row[i_ty]).strip() not in ("", "None", "nan"):
            out[_wkey(name)]["type"] = str(row[i_ty]).strip()
        avail = {}
        for lab, keys in (("deviation", ("deviations", "deviation", "survey")), ("checkshot", ("checkshots", "checkshot")),
                          ("markers", ("markers", "marker", "tops")), ("las", ("las", "logs"))):
            j = ix(*keys)
            if j is not None and row[j] is not None:
                v = str(row[j]).strip().lower()
                if v in ("/", "y", "yes", "1", "true", "✓"):
                    avail[lab] = True
                elif v in ("x", "n", "no", "0", "false", "-"):
                    avail[lab] = False
        if avail:
            out[_wkey(name)]["listed_data"] = avail
    return out


def _comment_labels(p: Path) -> list[str]:
    """Column labels given as '#' comment lines at the top of a checkshot / deviation / marker file."""
    lab = []
    for line in p.read_text(errors="replace").splitlines()[:20]:
        if line.startswith("#"):
            lab.append(line.lstrip("#").strip())
        elif line.strip():
            break
    return lab


def _read_markers(p: Path) -> list[dict]:
    labels = [re.sub(r"\(.*?\)", "", l).strip().lower() for l in _comment_labels(p)][2:]
    labels = [l for l in labels if l and l != "name"]
    out = []
    for line in p.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks, nums = re.split(r"\s+", s), []
        while toks:
            try:
                nums.insert(0, float(toks[-1])); toks.pop()
            except ValueError:
                break
        if not toks or not nums:
            continue
        rec = dict(name=" ".join(toks), md=None, tvdss=None, tvd=None, twt=None)
        for lab, v in zip(labels or ["md"], nums):
            key = "twt" if "twt" in lab else "tvdss" if "tvdss" in lab else "tvd" if "tvd" in lab else "md"
            rec[key] = v
        if rec["md"] is None and rec["tvd"] is None and rec["tvdss"] is None and rec["twt"] is None:
            rec["md"] = nums[0]
        out.append(rec)
    return out


SED_V = 1800.0   # m/s, assumed average sediment velocity between seabed and the top of a sonic log (no checkshot nearby)


def _sonic_profile(las_p: Path, dev, kb):
    """(TVDSS m ascending, slowness s/m, mnemonic, unit note) from a LAS sonic curve, or None."""
    try:
        from wellio import las_sonic, read_las
        son = las_sonic(read_las(las_p))
    except Exception:
        return None
    if not son or kb is None:
        return None
    d, slow, mn, unote = son
    ok = np.isfinite(d) & np.isfinite(slow)
    if ok.sum() < 20:
        return None
    d, slow = d[ok], slow[ok]
    z = np.interp(d, dev["md"], dev["tvdss"]) if dev is not None else d - kb
    o = np.argsort(z, kind="stable")
    z, slow = z[o], slow[o]
    keep = np.r_[True, np.diff(z) > 1e-6]
    z, slow = z[keep], slow[keep]
    if len(z) < 20 or z[-1] - z[0] < 50:
        return None
    if len(slow) >= 5:                              # light despike: 5-sample running median
        pad = np.pad(slow, 2, mode="edge")
        slow = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    return z, slow, mn, unote


class TimeDepth:
    """TVDSS (m) -> TWT (ms).  Anchored at MSL / seabed above the first checkshot, last interval velocity below."""

    def __init__(self, z, t, wd=None):
        z, t = np.asarray(z, float), np.asarray(t, float)
        anc = []
        if z[0] > 0.5:
            anc = [(0.0, 0.0)]
            if wd and 0.5 < wd < z[0] and 2000.0 * wd / WATER_V < t[0]:
                anc.append((float(wd), 2000.0 * wd / WATER_V))
        self.z = np.r_[[a[0] for a in anc], z]
        self.t = np.r_[[a[1] for a in anc], t]
        self.z0, self.z1 = float(z[0]), float(z[-1])
        k = max(0, len(z) - 4)
        dz, dt = z[-1] - z[k], t[-1] - t[k]
        self.vtail = float(2000.0 * dz / dt) if dt > 0 and dz > 0 else 2500.0

    def __call__(self, zq):
        zq = np.atleast_1d(np.asarray(zq, float))
        t = np.interp(zq, self.z, self.t)
        hi = zq > self.z[-1]
        t = np.where(hi, self.t[-1] + (zq - self.z[-1]) * 2000.0 / self.vtail, t)
        return t, (zq < self.z0) | hi


def _kb_options(hdr: dict, lst: dict | None, name: str) -> list[tuple[str, float]]:
    opts = []
    if lst and lst.get("rt") is not None:
        opts.append(("well list (RT)", float(lst["rt"])))
    for lab, keys in (("LAS EREF", ("EREF",)), ("LAS EKB/KB", ("EKB", "KB", "KBE", "ELEV_KB", "KB_ELEV", "EDF", "DFE", "APD"))):
        v = _num(hdr, keys)
        if v is not None:
            opts.append((lab, v))
    m = re.fullmatch(r"well[-_ ]?(\d+)", name, re.I)
    if m and int(m.group(1)) in NOTEBOOK_KB:
        opts.append(("team notebook table", NOTEBOOK_KB[int(m.group(1))]))
    return opts


def well_paths(folder: str | Path, well_list: str | Path | None = None, seismic: dict | None = None,
               approx: bool = False, overrides: dict | None = None) -> dict:
    """seismic = dict(x=[min,max], y=[min,max], t=[t0,t1]) from the Baseline headers (optional, for the footprint checks)."""
    folder = Path(folder) if folder else None
    if not folder or not folder.is_dir():
        return dict(wells=[], notes=["No well folder selected."])
    sig = []
    for root, _, files in os.walk(folder):
        if len(Path(root).parts) - len(folder.parts) > 4:
            continue
        sig += [(Path(root, f).as_posix(), Path(root, f).stat().st_mtime) for f in files if not f.startswith(".")]
    lp = Path(well_list) if well_list and Path(well_list).is_file() else None
    skey = None if not seismic else (tuple(seismic.get("x") or ()), tuple(seismic.get("y") or ()), tuple(seismic.get("t") or ()))
    return _well_paths(str(folder), tuple(sorted(sig)), str(lp) if lp else "", lp.stat().st_mtime if lp else 0, skey, bool(approx), json.dumps(overrides or {}, sort_keys=True))


def heads(folder, well_list=None, overrides=None) -> list[dict]:
    """Well heads (name, X, Y) from the LAS headers, or from the well list when a LAS has none."""
    folder = Path(folder) if folder else None
    if not folder or not folder.is_dir():
        return []
    sc = _scan(folder, overrides=overrides)
    lp = Path(well_list) if well_list and Path(well_list).is_file() else (sc["lists"][0] if sc["lists"] else None)
    lst = read_well_list(lp) if lp else {}
    out = []
    for key in sorted(set(sc["las"]) | set(lst), key=_natural):
        hdr = _header(sc["las"][key]) if key in sc["las"] else {}
        x, y = _num(hdr, X_KEYS), _num(hdr, Y_KEYS)
        if (x is None or y is None) and key in lst:
            x, y = lst[key]["x"], lst[key]["y"]
        if x is not None and y is not None:
            out.append(dict(name=hdr.get("WELL") or (lst.get(key) or {}).get("name") or key, x=x, y=y))
    return out


@lru_cache(maxsize=8)
def _well_paths(folder: str, _sig, listp: str, _lmt, skey, approx: bool, ovjson: str = "{}") -> dict:
    folder = Path(folder)
    sc = _scan(folder, overrides=json.loads(ovjson))
    notes = []
    lp = Path(listp) if listp else (sc["lists"][0] if sc["lists"] else None)
    lst = read_well_list(lp) if lp else {}
    if lp and not lst:
        notes.append(f"Well list {lp.name} could not be read (needs columns Well Name, X, Y, RT).")
    elif lp:
        notes.append(f"Well list: {lp.name} ({len(lst)} wells).")
    if not sc["las"] and not lst:
        notes.append("No well LAS files or well list found in the well folder.")
    seis = None
    if skey and skey[0] and skey[1]:
        seis = dict(x=skey[0], y=skey[1], t=skey[2] if len(skey) > 2 else ())
    wells, tds = [], {}
    keys = sorted(set(sc["las"]) | set(lst), key=_natural)

    for key in keys:
        L = lst.get(key)
        las_p = sc["las"].get(key)
        hdr = _header(las_p) if las_p else {}
        label = (L or {}).get("name") or hdr.get("WELL") or key
        rec = dict(name=label, key=key, file=las_p.name if las_p else None, las_path=str(las_p) if las_p else None, warnings=[], checks=[], markers=[])
        chk = lambda lvl, txt: (rec["checks"].append(dict(level=lvl, text=txt)), lvl != "ok" and rec["warnings"].append(txt))
        # --- head X/Y ---
        lx, ly = _num(hdr, X_KEYS), _num(hdr, Y_KEYS)
        if lx is not None and ly is not None:
            x, y, xy_src = lx, ly, "LAS header"
            if L and np.hypot(L["x"] - lx, L["y"] - ly) > 1.0:
                chk("warn", f"X/Y differ between LAS ({lx:.1f}, {ly:.1f}) and the well list ({L['x']:.1f}, {L['y']:.1f}); LAS used")
        elif L:
            x, y, xy_src = L["x"], L["y"], "well list"
        else:
            rec.update(x=None, y=None, kind="no X/Y", path=None, xy_source="none")
            chk("bad", "no X/Y in the LAS header and no well list: not drawn"); wells.append(rec); continue
        rec.update(x=x, y=y, xy_source=xy_src)
        # --- KB ---
        opts = _kb_options(hdr, L, rec["name"])
        if opts:
            kb, kb_src = opts[0][1], opts[0][0]
            spread = max(v for _, v in opts) - min(v for _, v in opts)
            if spread > 0.1:
                chk("warn", "KB sources disagree (" + ", ".join(f"{n} {v:g}" for n, v in opts) + f"); {kb_src} used")
        else:
            kb, kb_src = 0.0, "not found (0 m used)"
            chk("warn", "KB not found; 0 m used, so depths are not tied to MSL")
        rec.update(kb=kb, kb_source=kb_src, kb_options=[dict(source=n, value=v) for n, v in opts])
        wd = (L or {}).get("wd")
        td_md = (L or {}).get("td")
        if td_md is None:
            td_md = _num(hdr, ("STOP",))
        # --- deviation ---
        dv_p, cs_p, mk_p = sc["dv"].get(key), sc["cs"].get(key), sc["mk"].get(key)
        rec.update(checkshot=cs_p.name if cs_p else None, deviation=dv_p.name if dv_p else None, markers_file=mk_p.name if mk_p else None)
        dev = None
        kind = "vertical (no deviation file)"
        if dv_p:
            try:
                dv = _numeric(dv_p)
                if dv.shape[1] < 4:
                    raise ValueError("needs MDKB, TVD, INCL, AZIM")
                md, tvd, inc, azi = dv[:, :4].T
                if md[0] > 0.5:                              # survey starts below the wellhead: vertical above it
                    md, tvd, inc, azi = np.r_[0, md], np.r_[0, tvd], np.r_[0, inc], np.r_[0, azi]
                e, n, v = min_curvature_v(md, inc, azi)
                tvdss = tvd - kb
                dev = dict(md=md, tvd=tvd, tvdss=tvdss, e=e, n=n)
                kind = "deviated"
                rec["max_inclination"] = float(np.nanmax(inc))
                rec["horizontal_offset_m"] = float(np.hypot(e, n).max())
                dtv = float(np.nanmax(np.abs(v - tvd)))
                chk("ok" if dtv <= 5 else "warn", f"survey: TVD column vs minimum curvature max difference {dtv:.1f} m" if dtv > 5 else "survey TVD consistent with MD/inclination/azimuth")
                bx, by = _num(hdr, ("XOFFSET_BOTTOM",)), _num(hdr, ("YOFFSET_BOTTOM",))
                if bx is not None and by is not None:
                    d = float(np.hypot(e[-1] - bx, n[-1] - by))
                    chk("ok" if d <= 2 else "warn", f"bottom-hole offset matches the LAS header (difference {d:.1f} m)" if d <= 2 else
                        f"bottom-hole offset ({e[-1]:.1f}, {n[-1]:.1f}) m differs from the LAS header ({bx:.1f}, {by:.1f}) m by {d:.1f} m")
            except ValueError as ex:
                chk("warn", f"deviation file unreadable ({ex}); drawn vertical")
        else:
            chk("ok" if key not in sc["dv"] else "warn", "no deviation survey: drawn as a vertical well from the wellhead")
        # --- checkshot -> time-depth ---
        td, td_note = None, None
        if cs_p:
            try:
                cs = _numeric(cs_p)[:, :2]
                lab = " ".join(_comment_labels(cs_p)).lower()
                z = cs[:, 0].copy()
                if "ft" in lab or "feet" in lab:
                    z *= 0.3048
                basis = "tvdss"
                if "tvdss" not in lab and "tvd" in lab:
                    z -= kb; basis = "tvd (KB subtracted)"
                elif "tvdss" not in lab and "md" in lab:
                    chk("warn", "checkshot is in MD; treated as TVDSS")
                t = cs[:, 1].copy()
                if np.nanmax(z) <= 0 < -np.nanmin(z):          # depths written negative-down (elevation sign)
                    z = -z
                    chk("ok", "checkshot depths are negative (elevation sign): used as positive depth below MSL")
                o = np.argsort(z, kind="stable")
                z, t = z[o], t[o]
                if np.nanmax(t) < 20:
                    t *= 1000.0
                if "one" in lab and "way" in lab and "two" not in lab:
                    t *= 2.0
                if np.any(np.diff(t) <= 0):
                    chk("warn", "checkshot TWT is not increasing with depth")
                td = TimeDepth(z, t, wd)
                rec["checkshot_range_tvdss"] = [float(z[0]), float(z[-1])]
                tds[key] = (td, x, y)
                chk("ok", f"checkshot {basis.split()[0].upper()} {z[0]:.0f}–{z[-1]:.0f} m → TWT {t[0]:.0f}–{t[-1]:.0f} ms ({len(z)} points)")
            except ValueError as ex:
                chk("bad", f"checkshot unreadable ({ex})")
        if td is None and las_p is not None:          # no checkshot: try the LAS sonic (integrated later, once checkshots are known)
            rec["_sonic"] = _sonic_profile(las_p, dev, kb)
        rec["_wd"] = wd
        rec["_td"], rec["_dev"], rec["_td_md"] = td, dev, td_md
        rec["_kind"] = kind
        rec["_mk"] = mk_p
        wells.append(rec)

    # wells with no checkshot: 1) integrated sonic from the LAS, 2) optionally the nearest well's time-depth (approximate)
    for rec in wells:
        if rec.get("x") is None or rec.get("_td") is not None:
            continue
        son = rec.pop("_sonic", None)
        if son is not None:
            z, slow, mn, unote = son
            ztop, wd = float(z[0]), rec.get("_wd")
            if tds:
                nk = min(tds, key=lambda k: np.hypot(tds[k][1] - rec["x"], tds[k][2] - rec["y"]))
                t_top = float(tds[nk][0](ztop)[0][0])
                anchor = f"anchored at the log top ({ztop:.0f} m TVDSS) to {next((w['name'] for w in wells if w['key'] == nk), nk)}'s checkshot"
            else:
                w = wd if wd and 0 < wd < ztop else 0.0
                t_top = 2000.0 * w / WATER_V + 2000.0 * (ztop - w) / SED_V
                anchor = f"above the log top ({ztop:.0f} m TVDSS): water {WATER_V:.0f} m/s, sediment {SED_V:.0f} m/s (assumed)"
            t = t_top + 2000.0 * np.r_[0.0, np.cumsum(0.5 * (slow[1:] + slow[:-1]) * np.diff(z))]
            rec["_td"] = TimeDepth(z, t, wd)
            rec["time_source"] = "sonic"
            txt = f"no checkshot: time from the integrated sonic {unote}, {z[0]:.0f}–{z[-1]:.0f} m TVDSS; {anchor}. Sonic times are not drift-corrected"
            rec["checks"].append(dict(level="warn", text=txt))
            rec["warnings"].append("time from integrated sonic (no checkshot; not drift-corrected)")
            continue
        if approx and tds:
            nk = min(tds, key=lambda k: np.hypot(tds[k][1] - rec["x"], tds[k][2] - rec["y"]))
            rec["_td"] = tds[nk][0]; rec["approx_time"] = True
            rec["approx_from"] = next((w["name"] for w in wells if w["key"] == nk), nk)
            rec["checks"].append(dict(level="warn", text=f"no checkshot: time-depth borrowed from {rec['approx_from']} (approximate)"))
            rec["warnings"].append(f"no checkshot: time-depth borrowed from {rec['approx_from']} (approximate)")
        else:
            rec["checks"].append(dict(level="bad", text="no checkshot: cannot be placed in TWT; only the wellhead is drawn"))
            rec["warnings"].append("no checkshot: cannot be placed in TWT, drawn at the wellhead only")

    for rec in wells:
        td, dev, td_md, kind = rec.pop("_td", None), rec.pop("_dev", None), rec.pop("_td_md", None), rec.pop("_kind", "")
        mk_p = rec.pop("_mk", None)
        if rec.get("x") is None:
            continue
        x, y, kb = rec["x"], rec["y"], rec["kb"]
        if not td:
            rec.update(kind="no checkshot", path=None)
            rec["status"] = "bad"
            if seis:
                if not (seis["x"][0] <= x <= seis["x"][1] and seis["y"][0] <= y <= seis["y"][1]):
                    rec["checks"].append(dict(level="warn", text="wellhead is outside the seismic X/Y footprint"))
            continue
        # depth range of the path (TVDSS)
        if dev is not None:
            z_end = float(dev["tvdss"][-1])
        elif td_md:
            z_end = float(td_md - kb)
        else:
            z_end = td.z1
        z_end = min(max(z_end, td.z0), td.z1 + MAX_TAIL_M) if z_end > td.z1 else max(z_end, td.z0)
        depths = np.unique(np.r_[np.arange(0.0, z_end, STEP_M), z_end])
        twt, est = td(depths)
        if dev is not None:
            ts = dev["tvdss"]
            east, north = np.interp(depths, ts, dev["e"]), np.interp(depths, ts, dev["n"])
        else:
            east = north = np.zeros(len(depths))
        keep = np.unique(np.r_[np.linspace(0, len(depths) - 1, min(len(depths), 700)).round().astype(int)])
        px, py = x + east, y + north
        md_all = np.interp(depths, dev["tvdss"], dev["md"]) if dev is not None else depths + kb
        rec.update(kind=kind, path=dict(x=px[keep].round(2).tolist(), y=py[keep].round(2).tolist(), twt=twt[keep].round(2).tolist(),
                                        tvdss=depths[keep].round(1).tolist(), md=np.asarray(md_all)[keep].round(1).tolist(),
                                        est=est[keep].astype(int).tolist()),
                   twt_range=[float(twt.min()), float(twt.max())])
        if est.any():
            ranges = []
            if (depths < td.z0).any():
                ranges.append(f"above {td.z0:.0f} m TVDSS (MSL/seabed anchored)")
            if (depths > td.z1).any():
                ranges.append(f"below {td.z1:.0f} m TVDSS (last interval velocity {td.vtail:.0f} m/s)")
            rec["extrapolated"] = ranges
            rec["checks"].append(dict(level="ok", text="time extrapolated " + " and ".join(ranges)))
        # markers
        if mk_p:
            try:
                ms = _read_markers(mk_p)
            except Exception as ex:
                ms = []; rec["checks"].append(dict(level="warn", text=f"markers unreadable ({ex})"))
            bad = 0
            for m in ms:
                if m["twt"] is not None:
                    m["est"] = False
                    tw = m["twt"]; zs = None
                else:
                    if m["tvdss"] is not None:
                        zs = m["tvdss"]
                    elif m["tvd"] is not None:
                        zs = m["tvd"] - kb
                    else:
                        tvd_m = float(np.interp(m["md"], dev["md"], dev["tvd"])) if dev is not None else m["md"]
                        zs = tvd_m - kb
                    tw_a, es = td(zs); tw, m["est"] = float(tw_a[0]), bool(es[0])
                m["twt"] = round(float(tw), 2)
                if zs is not None:
                    m["tvdss"] = round(float(zs), 2)
                    xs = float(np.interp(zs, depths, px)); ys = float(np.interp(zs, depths, py))
                    m["x"], m["y"] = round(xs, 2), round(ys, 2)
                else:
                    m["x"], m["y"] = x, y
            rec["markers"] = ms
            rec["checks"].append(dict(level="ok", text=f"{len(ms)} markers placed in TWT ({min((m['twt'] for m in ms), default=0):.0f}–{max((m['twt'] for m in ms), default=0):.0f} ms)"))
        # seismic checks
        if seis:
            inside = (px >= seis["x"][0]) & (px <= seis["x"][1]) & (py >= seis["y"][0]) & (py <= seis["y"][1])
            if not inside.any():
                rec["checks"].append(dict(level="bad", text="well is outside the seismic X/Y footprint (coordinate system or units differ?)"))
            elif not inside.all():
                rec["checks"].append(dict(level="warn", text=f"{100 * (1 - inside.mean()):.0f}% of the path is outside the seismic X/Y footprint"))
            else:
                rec["checks"].append(dict(level="ok", text="inside the seismic X/Y footprint"))
            if len(seis["t"]) == 2:
                t0, t1 = seis["t"]
                if rec["twt_range"][1] > t1 + 1e-6:
                    rec["checks"].append(dict(level="warn", text=f"well reaches {rec['twt_range'][1]:.0f} ms, below the seismic ({t1:.0f} ms); it is cut at the volume bottom"))
                else:
                    rec["checks"].append(dict(level="ok", text=f"TWT {rec['twt_range'][0]:.0f}–{rec['twt_range'][1]:.0f} ms lies inside the seismic time range"))
        rec["warnings"] = [c["text"] for c in rec["checks"] if c["level"] != "ok"]
    for rec in wells:
        lv = [c["level"] for c in rec["checks"]]
        rec["status"] = "bad" if "bad" in lv else "warn" if "warn" in lv else "ok"
    wells.sort(key=lambda w: _natural(w["key"]))
    return dict(wells=wells, notes=notes, domain="TWT (ms)", seismic=seis)


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def intersections(paths: dict, ws, max_dist: float | None = None) -> None:
    """Add, per well, where its path crosses each horizon: [{horizon, x, y, twt}]. Uses the horizon TWT points."""
    from scipy.spatial import cKDTree
    import results
    trees = {}
    for h in ws.horizons():
        f = ws.files(h)
        src = f.get("rms") or f.get("horizon")
        if not src:
            continue
        d = results.read(src)
        xy, t = d[["X", "Y"]].to_numpy(float), d["TWT"].to_numpy(float)
        ok = np.isfinite(xy).all(1) & np.isfinite(t)
        if ok.sum() < 3:
            continue
        tr = cKDTree(xy[ok])
        sp = float(np.median(tr.query(xy[ok][:: max(1, ok.sum() // 2000)], k=2)[0][:, 1]))
        trees[h] = (tr, t[ok], max_dist or max(3 * sp, 30.0))
    for w in paths["wells"]:
        w["crossings"] = []
        p = w.get("path")
        if not p:
            continue
        px, py, pt = (np.asarray(p[k], float) for k in ("x", "y", "twt"))
        for h, (tr, t, md) in trees.items():
            dist, idx = tr.query(np.c_[px, py])
            ht = np.where(dist <= md, t[idx], np.nan)
            diff = pt - ht
            s = np.where(np.isfinite(diff[:-1]) & np.isfinite(diff[1:]) & (np.sign(diff[:-1]) != np.sign(diff[1:])))[0]
            if len(s):
                k = s[0]
                a = diff[k] / (diff[k] - diff[k + 1]) if diff[k] != diff[k + 1] else 0.0
                w["crossings"].append(dict(horizon=h, x=float(px[k] + a * (px[k + 1] - px[k])),
                                           y=float(py[k] + a * (py[k + 1] - py[k])), twt=float(pt[k] + a * (pt[k + 1] - pt[k]))))


def file_table(folder: str | Path, well_list=None, overrides: dict | None = None) -> dict:
    """Detected files with their type / well guesses, plus the wells they can be assigned to."""
    folder = Path(folder)
    sc = _scan(folder, overrides=overrides)
    lp = Path(well_list) if well_list and Path(well_list).is_file() else (sc["lists"][0] if sc["lists"] else None)
    lst = read_well_list(lp) if lp else {}
    names = {}
    for k, p in sc["las"].items():
        names[k] = _header(p).get("WELL") or k
    for k, v in lst.items():
        names.setdefault(k, v["name"])
    for f in sc["files"]:                                   # wells only known from aux files
        if f["well_final"] and f["well_final"] not in names:
            k = f["well_final"]
            names[k] = re.sub(r"^well-", "Well-", k)
    wells = [dict(key=k, name=names[k]) for k in sorted(names, key=_natural)]
    kinds = dict(las="las", checkshot="cs", deviation="dv", markers="mk")
    for f in sc["files"]:
        k = kinds.get(f["type_final"])
        f["used"] = (sc[k].get(f["well_final"]) == folder / f["path"]) if k and f["well_final"] else (f["type_final"] == "welllist")
        f["preview"] = "\n".join(_head_lines(folder / f["path"], 8))[:500] if not f["path"].lower().endswith((".xlsx", ".xlsm", ".xls", ".las")) else ""
    return dict(files=sc["files"], wells=wells, types=[dict(id=i, label=l) for i, l in FILE_TYPES])
