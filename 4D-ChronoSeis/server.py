"""
4D ChronoSeis: web GUI for the team's 4D workflow (4D ChronoSeis engine), running on the HPC.

    export APP4D_REPO=/path/to/Shell_Geohackathon          # clone of the team notebooks (read-only)
    export APP4D_ROOTS=/data/epic20/APGCE/input_data:/data/epic20/APGCE/<team>   # folders the GUI may read/write
    python server.py                                        # open http://localhost:8050 in Firefox on the HPC desktop

The page collects the inputs, runs the notebook stages in Jupyter kernels on the HPC (nbrunner.py),
reads the notebook's output files (results.py) for the Summary and Detail views, and uses the hosted
Gemma model for plain-language interpretation.

Security: binds to 127.0.0.1 only; reads/writes only inside APP4D_ROOTS (the output working directory
must be inside them too); never runs shell commands.
Prototype assumption: seismic, wells and platforms share one coordinate system and units (not checked).
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import threading
import time
import os
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import advisor
import offsets
import nbrunner
import numpy as np
import wellpath
import results
import section as secmod
import slices
import wellio

HERE = Path(__file__).parent
ROOTS = [Path(p).resolve() for p in os.environ.get("APP4D_ROOTS", str(HERE / "demo_data")).split(":") if p]
OUT = Path(os.environ.get("APP4D_OUT", HERE / "runs")).resolve()          # app state (settings, kernel cwd)
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(os.environ.get("APP4D_REPO", HERE / "team_notebooks")).resolve()
ROOT_FROM = os.environ.get("APP4D_ROOT_FROM", "/home/apgce_team_08")
CHALLENGE = os.environ.get("APP4D_CHALLENGE", "/data/epic20/APGCE/input_data/challenge3")
GEMMA_URL = os.environ.get("GEMMA_URL", "http://172.28.34.1:8000/v1/chat/completions")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-26B-A4B-it")
SETTINGS = OUT / "settings.json"
SHOW = {".sgy", ".segy", ".seg", ".csv", ".txt", ".xlsx", ".xls", ".dat", ".las", ".json"}

app = FastAPI(title="4D ChronoSeis")
RUNNER = nbrunner.Runner(REPO, HERE / "pipeline.json", OUT / "nb_work", OUT)


# ------------------------------------------------------------------ helpers
def safe(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    for r in ROOTS:
        if p == r or r in p.parents:
            return p
    raise HTTPException(403, "Path is outside the allowed folders.")


def settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


_SET_LOCK = threading.RLock()


def _write_settings(s: dict):
    """Atomic write (temp file + rename) so a parallel request never reads a half-written settings.json."""
    tmp = SETTINGS.with_name(SETTINGS.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(s, indent=2))
    os.replace(tmp, SETTINGS)


def save_settings(**kw):
    with _SET_LOCK:                 # the page saves several inputs at once: serialise read-modify-write
        s = settings()
        s.update({k: v for k, v in kw.items() if v is not None})
        _write_settings(s)


def well_overrides(folder) -> dict:
    """The user's corrections (file type / well) for this well folder."""
    return (settings().get("well_map") or {}).get(str(folder), {})


def workspace_path(path: str | None = None) -> Path:
    w = path or settings().get("workspace")
    if not w:
        raise HTTPException(400, "Set the output working directory first.")
    return safe(w)


def ask_gemma(messages: list[dict]) -> str:
    body = json.dumps(dict(model=GEMMA_MODEL, messages=messages, temperature=0.3, max_tokens=900)).encode()
    req = urllib.request.Request(GEMMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def gemma_or_note(msgs):
    try:
        return dict(source="gemma", text=ask_gemma(msgs))
    except Exception as e:
        why = "took too long to answer" if "timed out" in str(e).lower() or isinstance(e, TimeoutError) else f"could not be reached ({type(e).__name__})"
        return dict(source="none", text=f"Gemma {why}. The results above are complete; try again, or ask a shorter question in the Advisor.")


# ------------------------------------------------------------------ settings & files
@app.get("/api/config")
def config():
    return dict(roots=[str(r) for r in ROOTS], gemma_url=GEMMA_URL, repo=str(REPO), repo_found=REPO.is_dir(),
                root_from=ROOT_FROM, settings=settings())


@app.post("/api/settings")
def set_settings(body: dict):
    clean = {}
    for k in ("workspace", "base", "monitor", "markers_csv", "platform_csv", "wells_dir", "well_list", "horizons_dir", "seismic_domain"):
        v = body.get(k)
        if v in (None, ""):
            continue
        clean[k] = v if k == "seismic_domain" else str(safe(v))
    save_settings(**clean)          # folders are created only when a run starts
    return settings()


@app.get("/api/browse")
def browse(path: str = ""):
    if not path:
        return dict(path="", parent=None, entries=[dict(name=str(r), type="dir", path=str(r)) for r in ROOTS])
    p = safe(path)
    if not p.is_dir():
        raise HTTPException(400, "Not a folder.")
    entries = []
    try:
        for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                entries.append(dict(name=c.name, type="dir", path=str(c)))
            elif c.suffix.lower() in SHOW:
                entries.append(dict(name=c.name, type="file", path=str(c), size=c.stat().st_size))
    except PermissionError:
        raise HTTPException(403, "No permission to list this folder.")
    return dict(path=str(p), parent="" if any(p == r for r in ROOTS) else str(p.parent), entries=entries)


READONLY = [Path(p).resolve() for p in os.environ.get("APP4D_READONLY", "/data/epic20").split(":") if p]


@app.post("/api/mkdir")
def make_dir(body: dict):
    """Create one new sub-folder (for an output working directory). Never inside the read-only input data."""
    parent = safe(body.get("parent") or "")
    name = (body.get("name") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,79}", name):
        raise HTTPException(400, "Use letters, numbers, spaces, dots, dashes or underscores (max 80), starting with a letter or number.")
    if not parent.is_dir():
        raise HTTPException(400, "The parent is not a folder.")
    new = (parent / name).resolve()
    if new.parent != parent:
        raise HTTPException(400, "Invalid folder name.")
    if any(new == r or r in new.parents for r in READONLY):
        raise HTTPException(403, f"Folders cannot be created inside the input data ({', '.join(map(str, READONLY))}).")
    safe(str(new))
    if new.exists():
        raise HTTPException(409, f"{name} already exists here.")
    try:
        new.mkdir()
    except OSError as e:
        raise HTTPException(400, f"Could not create the folder: {e.strerror or e}")
    return dict(path=str(new))


@app.get("/api/scan")
def scan(path: str):
    p = safe(path)
    if p.is_file():
        p = p.parent
    try:
        return wellio.scan_folder(p)
    except Exception as e:
        raise HTTPException(400, f"Could not scan this folder: {e}")


class InspectReq(BaseModel):
    path: str


_INSPECT_LOCK = threading.Lock()


def _inspect_cache_get(p: Path):
    f = OUT / "_cache" / "inspect.json"
    try:
        st = p.stat()
        rec = json.loads(f.read_text()).get(str(p))
        if rec and rec.get("_size") == st.st_size and rec.get("_mtime") == st.st_mtime:
            return {k: v for k, v in rec.items() if not k.startswith("_")}
    except Exception:
        pass
    return None


def _inspect_cache_put(p: Path, rec: dict):
    f = OUT / "_cache" / "inspect.json"
    with _INSPECT_LOCK:
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            try:
                allr = json.loads(f.read_text())
            except Exception:
                allr = {}
            st = p.stat()
            allr[str(p)] = dict(rec, _size=st.st_size, _mtime=st.st_mtime)
            tmp = f.with_name(f.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(allr))
            os.replace(tmp, f)
        except OSError:
            pass


@app.post("/api/inspect")
def inspect(req: InspectReq):
    """SEG-Y summary (traces, time range, IL/XL ranges). The first read scans every trace header (can take a minute on a
    6-7 GB file); the result is cached in <APP4D_OUT>/_cache/inspect.json keyed by path, size and modification time."""
    p = safe(req.path)
    if p.suffix.lower() not in (".sgy", ".segy", ".seg"):
        raise HTTPException(400, "Not a SEG-Y file (.sgy/.segy).")
    hit = _inspect_cache_get(p)
    if hit:
        return dict(hit, cached=True)
    slices.note(f"Reading the trace headers of {p.name} (first time only; cached afterwards)…")
    try:
        import segyio
        with segyio.open(str(p), "r", ignore_geometry=True) as f:
            il = f.attributes(segyio.TraceField.INLINE_3D)[:]
            xl = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            dt = segyio.tools.dt(f) / 1000.0
            try:
                txt = segyio.tools.wrap(f.text[0])
            except Exception:
                txt = ""
            hint, why = wellio.seismic_domain_hint(p, txt)
            rec = dict(file=p.name, traces=int(f.tracecount), samples=len(f.samples), dt_ms=dt,
                       t_start_ms=float(f.samples[0]), t_end_ms=float(f.samples[-1]),
                       il_min=int(il.min()), il_max=int(il.max()), xl_min=int(xl.min()), xl_max=int(xl.max()),
                       size_mb=round(p.stat().st_size / 1e6, 1), domain_hint=hint, domain_reason=why)
        _inspect_cache_put(p, rec)
        return rec
    except Exception as e:
        raise HTTPException(400, f"Could not read SEG-Y: {e}")


class FileCheck(BaseModel):
    path: str
    kind: str


@app.post("/api/check")
def check(req: FileCheck):
    """Light checks of the other inputs, so problems show before a long run."""
    p = safe(req.path)
    try:
        if req.kind == "platform":
            pl = results.read_platforms(p)
            if not pl:
                raise ValueError("needs X(m) and Y(m) columns")
            return dict(ok=True, text=f"{len(pl)} platform(s): " + ", ".join(x["name"] for x in pl[:6]), platforms=pl)
        if req.kind == "horizons":
            if not p.is_dir():
                raise ValueError("choose a folder with the horizon .txt files")
            import pandas as pd
            fs = sorted(p.glob("*.txt"))
            if not fs:
                raise ValueError("no .txt horizon files here")
            cols = pd.read_csv(fs[0], sep="\t", nrows=3).columns.map(str).str.strip()
            need = [c for c in ("X", "Y", "Inline", "Crossline", "TWT") if c not in set(cols)]
            if need:
                raise ValueError(f"{fs[0].name}: missing column(s) {', '.join(need)} (tab-separated X, Y, Inline, Crossline, TWT)")
            return dict(ok=True, text=f"{len(fs)} horizon file(s): " + ", ".join(f.stem for f in fs[:6]) + (" …" if len(fs) > 6 else "")
                        + ". Horizon picking will be skipped.")
        if req.kind == "markers":
            import pandas as pd
            df = pd.read_csv(p)
            cols = [c for c in df.columns if "twt" in c.lower()]
            if not cols:
                raise ValueError("no TWT column (expected e.g. 'Well-1 TWT (ms)' or 'TWT')")
            return dict(ok=True, text=f"{len(df)} marker(s), {len(cols)} TWT column(s). First: "
                        + ", ".join(map(str, df[df.columns[0]].head(5))))
        if req.kind == "wells":
            if not p.is_dir():
                raise ValueError("choose a folder")
            sc = wellpath._scan(p, overrides=well_overrides(p))
            w = wellpath.heads(p, settings().get("well_list"), well_overrides(p))
            lst = ", a well list" if sc["lists"] else ""
            note = "" if w else " No well X/Y found (LAS header or well list): wells won't appear on maps."
            return dict(ok=bool(w), text=f"{len(w)} well head(s) with X/Y; {len(sc['las'])} LAS, {len(sc['cs'])} checkshot, {len(sc['dv'])} deviation, {len(sc['mk'])} marker file(s){lst} (sub-folders are searched)." + note, wells=w)
        if req.kind == "welllist":
            lst = wellpath.read_well_list(p) if p.is_file() else {}
            if not lst:
                raise ValueError("no well table found (needs columns like Well Name, X, Y, RT)")
            return dict(ok=True, text=f"{len(lst)} wells: " + ", ".join(v['name'] for v in list(lst.values())[:6]) + ("…" if len(lst) > 6 else ""))
        if req.kind == "workspace":
            ws = results.Workspace(p)
            st = ws.status()
            if not p.exists():
                return dict(ok=True, text="New folder: it will be created when the run starts.", status=st)
            return dict(ok=True, text=(f"Existing results: {st['horizons']} horizon(s), {st['classes']} class file(s)."
                                       if st["horizons"] else "Empty folder: results will be written here."), status=st)
        raise ValueError("unknown kind")
    except HTTPException:
        raise
    except Exception as e:
        return dict(ok=False, text=str(e))


# ------------------------------------------------------------------ projects (save / open / relocate)
PROJ_FMT, PROJ_EXT = "4d-chronoseis-project", ".chronoseis.json"
PROJ_KEYS = ("base", "monitor", "markers_csv", "wells_dir", "well_list", "platform_csv", "horizons_dir")
PROJ_DIRS = {"wells_dir", "workspace", "horizons_dir"}


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._") or "project"


def _allowed(path: str | None) -> Path | None:
    if not path:
        return None
    try:
        return safe(path)
    except HTTPException:
        return None


def _proj_checks(proj: dict) -> dict:
    """Is every saved path still there? Results folder is checked for 4D ChronoSeis outputs."""
    out = {}
    for k in PROJ_KEYS + ("workspace",):
        v = (proj.get("workspace") if k == "workspace" else proj.get("inputs", {}).get(k)) or ""
        p = _allowed(v)
        if not v:
            out[k] = dict(path="", ok=None, text="not set")
        elif p is None:
            out[k] = dict(path=v, ok=False, text="outside the allowed folders")
        elif (k in PROJ_DIRS and not p.is_dir()) or (k not in PROJ_DIRS and not p.is_file()):
            out[k] = dict(path=v, ok=False, text="not found")
        elif k == "workspace":
            st = results.Workspace(p).status()
            has = st["horizons"] or st["classes"]
            had = (proj.get("results_snapshot") or {}).get("horizons", 0) or (proj.get("results_snapshot") or {}).get("classes", 0)
            if had and not has:
                out[k] = dict(path=v, ok=False, text="folder is there but empty (the saved results are not in it)")
            else:
                out[k] = dict(path=v, ok=True, has_results=bool(has), status=st,
                              text=f"{st['horizons']} horizon(s), {st['classes']} class file(s)" if has else "found, no results yet")
        else:
            out[k] = dict(path=v, ok=True, text="found")
    return out


def _proj_reply(proj: dict, path: str | None) -> dict:
    ch = _proj_checks(proj)
    missing = [k for k, c in ch.items() if c["ok"] is False]
    return dict(project=proj, path=path, checks=ch, missing=missing, workspace_ok=bool(ch["workspace"]["ok"]))


def _read_project(p: Path) -> dict:
    try:
        proj = json.loads(p.read_text())
    except Exception as e:
        raise HTTPException(400, f"Not a readable project file ({type(e).__name__}).")
    if proj.get("format") != PROJ_FMT:
        raise HTTPException(400, "This JSON file is not a 4D ChronoSeis project.")
    return proj


def _remember(name: str, path: str):
    rec = [r for r in settings().get("recent_projects", []) if r.get("path") != path]
    save_settings(recent_projects=[dict(name=name, path=path, opened=time.strftime("%Y-%m-%d %H:%M"))] + rec[:7], project=path)


@app.post("/api/project/save")
def project_save(body: dict):
    ws = workspace_path(body.get("workspace"))
    folder = safe(body["folder"]) if body.get("folder") else ws
    folder.mkdir(parents=True, exist_ok=True)
    name = (body.get("name") or ws.name).strip()[:80]
    inputs = {k: str(safe(body[k])) if body.get(k) else "" for k in PROJ_KEYS}
    st = results.Workspace(ws).status()
    proj = dict(format=PROJ_FMT, version=1, name=name, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                engine="4D ChronoSeis", seismic_domain=body.get("seismic_domain") or settings().get("seismic_domain"),
                workspace=str(ws), inputs=inputs, results_snapshot=dict(horizons=st["horizons"], classes=st["classes"]),
                note="Paths are absolute. If the folders move, open the project and pick the folder that holds the files.")
    path = folder / (_slug(name) + PROJ_EXT)
    path.write_text(json.dumps(proj, indent=2))
    _remember(name, str(path))
    return dict(path=str(path), project=proj)


@app.get("/api/project/recent")
def project_recent():
    out = []
    for r in settings().get("recent_projects", []):
        p = _allowed(r.get("path"))
        out.append(dict(r, available=bool(p and p.is_file())))
    return dict(recent=out, current=settings().get("project"))


@app.post("/api/project/open")
def project_open(body: dict):
    p = safe(body.get("path") or "")
    if p.is_dir():
        found = sorted(p.glob("*" + PROJ_EXT))
        if not found:
            raise HTTPException(404, "No 4D ChronoSeis project file (*.chronoseis.json) in this folder.")
        if len(found) > 1:
            return dict(choose=[str(x) for x in found])
        p = found[0]
    if not p.is_file():
        raise HTTPException(404, "Project file not found. Pick it again, or pick the folder that contains it.")
    return _proj_reply(_read_project(p), str(p))


def _walk(root: Path, max_depth=4, limit=30000):
    n, stack = 0, [(root, 0)]
    while stack and n < limit:
        d, dep = stack.pop()
        try:
            kids = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        for c in kids:
            n += 1
            yield c
            if c.is_dir() and not c.name.startswith(".") and dep < max_depth:
                stack.append((c, dep + 1))


@app.post("/api/project/relocate")
def project_relocate(body: dict):
    """The saved folders are gone: look in a folder the user picks for the results and any missing inputs."""
    proj, root = body["project"], safe(body["folder"])
    if not root.is_dir():
        raise HTTPException(400, "Choose a folder.")
    checks, moved = _proj_checks(proj), []
    want = {k: Path(c["path"]).name for k, c in checks.items() if c["ok"] is False and c["path"]}
    has_res = lambda d: (lambda st: st["horizons"] or st["classes"])(results.Workspace(d).status())
    if "workspace" in want and has_res(root):            # the picked folder is itself the results folder
        proj["workspace"] = str(root); moved.append("workspace"); want.pop("workspace")
    if want:
        best = {}
        for c in _walk(root):
            for k, nm in want.items():
                if c.name != nm or k in best:
                    continue
                if k == "workspace":
                    if c.is_dir() and has_res(c):
                        best[k] = c
                elif (k in PROJ_DIRS and c.is_dir()) or (k not in PROJ_DIRS and c.is_file()):
                    best[k] = c
            if len(best) == len(want):
                break
        if "workspace" in want and "workspace" not in best:   # any results folder will do
            for c in _walk(root, max_depth=2):
                if c.is_dir() and has_res(c):
                    best["workspace"] = c
                    break
        for k, c in best.items():
            if k == "workspace":
                proj["workspace"] = str(c)
            else:
                proj.setdefault("inputs", {})[k] = str(c)
            moved.append(k)
    rep = _proj_reply(proj, body.get("path"))
    rep["relocated"] = moved
    return rep


@app.post("/api/project/apply")
def project_apply(body: dict):
    """Make an opened project the current one (only paths that exist are applied)."""
    proj, path = body["project"], body.get("path")
    ch = _proj_checks(proj)
    if not ch["workspace"]["ok"]:
        raise HTTPException(400, "The output working directory of this project is not available. Pick the folder that holds its results.")
    upd = dict(workspace=ch["workspace"]["path"], seismic_domain=proj.get("seismic_domain"))
    for k in PROJ_KEYS:
        if ch[k]["ok"]:
            upd[k] = ch[k]["path"]
    s = settings()
    for k in PROJ_KEYS:                       # don't keep inputs from the previous project
        s.pop(k, None)
    with _SET_LOCK:
        _write_settings(s)
    save_settings(**upd)
    if path and _allowed(path):
        if body.get("update_file") and body.get("relocated"):
            proj["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            Path(path).write_text(json.dumps(proj, indent=2))
        _remember(proj.get("name") or Path(path).stem, path)
    return dict(settings=settings(), checks=ch)


# ------------------------------------------------------------------ team notebooks
@app.get("/api/nb")
def nb_catalogue():
    return dict(repo=str(REPO), repo_found=REPO.is_dir(), root_from=ROOT_FROM, notebooks=RUNNER.catalogue())


@app.get("/api/nb/{nb_id}/{stage_id}/code")
def nb_code(nb_id: str, stage_id: str):
    try:
        return RUNNER.stage_code(nb_id, stage_id)
    except (KeyError, StopIteration, FileNotFoundError):
        raise HTTPException(404, "Unknown notebook or stage.")


FOREIGN_OUTPUTS = ("Autopick_Horizon", "4D_RMS_NRMS_3", "Maps", "4D_Platform_Constrained_Maps",
                   "Sweetness", "4D_difference.sgy")
MARKER = ".chronoseis_workspace"


REUSE_ITEMS = ("Autopick_Horizon", "4D_Matched_Inline_Crossline.csv")   # copied (small)
REUSE_LINKS = ("4D_difference.sgy",)                                    # large: linked, never copied


def reuse_team_results(workspace: str | None = None, dry: bool = False) -> dict:
    """Bring the team's earlier engine results (horizons, RMS/NdRMS, classes, sweetness) into the results folder, so
    Summary, Detail and the Advisor work at once, without re-running the long GPU stages. The team's folder is only
    read; nothing existing in the results folder is overwritten."""
    ws = workspace_path(workspace)
    team = Path(ROOT_FROM)
    if ws == team or ws in team.parents:
        raise HTTPException(400, f"Choose a results folder that is not the team folder itself ({ROOT_FROM}).")
    found = [n for n in REUSE_ITEMS + REUSE_LINKS if (team / n).exists()]
    if not any((team / n).exists() for n in REUSE_ITEMS[:1]):
        raise HTTPException(404, f"No earlier horizons/results found in {ROOT_FROM} (expected Autopick_Horizon).")
    if dry:
        return dict(workspace=str(ws), source=str(team), items=found)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / MARKER).write_text("Created by 4D ChronoSeis. Runs started from the GUI write into this folder.\n")
    n_new = n_skip = 0
    nbytes = 0

    def cp(src, dst):
        nonlocal n_new, n_skip, nbytes
        if Path(dst).exists():
            n_skip += 1
            return dst
        shutil.copy2(src, dst); n_new += 1; nbytes += Path(dst).stat().st_size
        return dst

    for n in REUSE_ITEMS:
        src = team / n
        if src.is_dir():
            shutil.copytree(src, ws / n, copy_function=cp, dirs_exist_ok=True, symlinks=False)
        elif src.is_file():
            cp(src, ws / n)
    linked = []
    for n in REUSE_LINKS:
        src, dst = team / n, ws / n
        if src.exists() and not dst.exists():
            os.symlink(src, dst); linked.append(n)
    save_settings(workspace=str(ws))
    st = results.Workspace(ws).status()
    return dict(workspace=str(ws), source=str(team), copied_files=n_new, skipped_existing=n_skip, mb=round(nbytes / 1e6, 1),
                linked=linked, status=st)


@app.post("/api/ws/reuse")
def ws_reuse(body: dict):
    return reuse_team_results(body.get("workspace"), bool(body.get("dry")))


def _guard_ws(ws: Path, allow_overwrite: bool):
    """Protect results that already exist (e.g. the team's earlier HPC run)."""
    team = Path(ROOT_FROM)
    if ws == team or ws in team.parents:
        raise HTTPException(400, f"The output folder cannot be the team folder itself ({ROOT_FROM}), because the "
                                 f"notebook's earlier results live there. Use a new subfolder, e.g. {ROOT_FROM}/chronoseis_output.")
    if (ws / MARKER).exists() or (ws / "_app_runs").exists():
        return  # created by this app: re-running stages here is normal
    found = [n for n in FOREIGN_OUTPUTS if (ws / n).exists()]
    if found and not allow_overwrite:
        raise HTTPException(409, "EXISTS: this folder already contains results that were not created by the GUI "
                                 f"({', '.join(found)}). Running here would replace them. Choose a new folder, or confirm to overwrite.")


def _params(body: dict) -> dict:
    ws = workspace_path(body.get("workspace"))
    _guard_ws(ws, bool(body.get("allow_overwrite")))
    ws.mkdir(parents=True, exist_ok=True)
    (ws / MARKER).write_text("Created by 4D ChronoSeis. Runs started from the GUI write into this folder.\n")
    params = dict(base="", monitor="", markers_csv="", platform_csv="", wells_dir="", horizons_dir="",
                  workdir=str(ws), root_from=ROOT_FROM, root_to=str(ws))
    for k in ("base", "monitor", "markers_csv", "platform_csv", "wells_dir", "horizons_dir"):
        if body.get(k):
            params[k] = str(safe(body[k]))
    save_settings(workspace=str(ws), **{k: params[k] or None for k in ("base", "monitor", "markers_csv", "platform_csv", "wells_dir", "horizons_dir")})
    return params


GEN_STAGES = ("seeds", "picking", "refine", "export_h")                     # horizon generation + export
NEEDS_HORIZONS = ("import_horizons", "well_helper", "horizon_qc", "rms", "sweetness")   # need horizons in memory; classes/viewer3 read files


def _use_existing_horizons(stages: list[str], params: dict) -> list[str]:
    """With an 'Existing horizons' folder set: drop seeds/picking/refine/export and read the files instead."""
    hd = Path(params["horizons_dir"])
    if not hd.is_dir() or not (list(hd.glob("*.txt"))):
        raise HTTPException(400, f"No horizon .txt files in {hd}. Expected tab-separated files with X, Y, Inline, Crossline, TWT.")
    out = [s for s in stages if s not in GEN_STAGES]
    if "import_horizons" not in out and any(s in out for s in NEEDS_HORIZONS):
        k = min(i for i, s in enumerate(out) if s in NEEDS_HORIZONS)
        out.insert(k, "import_horizons")
    ws = Path(params["root_to"]); dst = ws / "Autopick_Horizon"
    if hd.resolve() != dst.resolve():                  # make them visible to Summary / Detail: link, never copy or overwrite
        dst.mkdir(parents=True, exist_ok=True)
        for f in list(hd.glob("*.txt")) + list(hd.glob("Autopick_Export_Summary.csv")):
            t = dst / f.name
            if not t.exists():
                try:
                    os.symlink(f.resolve(), t)
                except OSError:
                    shutil.copy2(f, t)
    return out


@app.post("/api/nb/run")
def nb_run(body: dict):
    nb_id, stages = body.get("nb"), body.get("stages") or []
    if nb_id not in RUNNER.nbs or not stages:
        raise HTTPException(400, "Choose a notebook and at least one stage.")
    if body.get("seismic_domain") == "depth":
        raise HTTPException(400, "4D ChronoSeis works in time (TWT). Depth-domain volumes are not supported.")
    params = _params(body)
    uses_sel = any("uses_selection" in st.get("flags", []) for st in RUNNER.nbs[nb_id]["stages"] if st["id"] in stages)
    if uses_sel and not (params["base"] and params["monitor"]):
        raise HTTPException(400, "Select both the Baseline and the Monitor SEG-Y first.")
    if nb_id == "4d" and params.get("horizons_dir"):
        stages = _use_existing_horizons(stages, params)
    if nb_id == "4d":   # stages that need the seismic loaded in the same kernel: refuse early, before asking SLURM for a GPU
        sess = RUNNER.sessions.get(nb_id) if hasattr(RUNNER, "sessions") else None
        loaded = bool(sess and sess.alive() and sess.stage_status.get("load_match") == "done")
        free = {"install", "gemma_test", "load_match", "classes", "viewer3"}
        if "load_match" not in stages and not loaded and any(s not in free for s in stages):
            raise HTTPException(400, "These stages need the seismic loaded in this session first. Run 'Load & match Baseline / Monitor' "
                                     "(stage 03) and the stages before the one you want, or use 'Run full workflow'. "
                                     "For the results already computed, open Summary and Detail instead.")
    try:
        jid = RUNNER.run(nb_id, stages, params, include_install=bool(body.get("include_install")),
                         log_dir=Path(params["root_to"]) / "_app_runs", url_prefix="/api/ws/file?path=_app_runs")
        return dict(job_id=jid)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/team/run_all")
def run_all(body: dict):
    stages = [s["id"] for s in RUNNER.nbs["4d"]["stages"] if s.get("default_selected", True)]
    return nb_run({**body, "nb": "4d", "stages": stages})


@app.get("/api/nb/jobs/{jid}")
def nb_job(jid: str, since: int = 0):
    if jid not in RUNNER.jobs:
        raise HTTPException(404, "Unknown run.")
    return RUNNER.job_view(jid, since)


@app.post("/api/nb/jobs/{jid}/stop")
def nb_stop(jid: str):
    if jid not in RUNNER.jobs:
        raise HTTPException(404, "Unknown run.")
    RUNNER.stop(jid)
    return dict(ok=True)


@app.post("/api/nb/{nb_id}/restart")
def nb_restart(nb_id: str):
    if nb_id not in RUNNER.nbs:
        raise HTTPException(404, "Unknown notebook.")
    s = RUNNER.session(nb_id)
    if s.busy:
        raise HTTPException(409, "Stop the running stage first.")
    if s.km is None:
        return dict(ok=True, note="Kernel not started yet.")
    try:
        s.restart()
        return dict(ok=True, kernel=None, note="Kernel stopped; the next run starts a fresh one.")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/nb/jobs/{jid}/explain")
def nb_explain(jid: str, body: dict | None = None):
    if jid not in RUNNER.jobs:
        raise HTTPException(404, "Unknown run.")
    j = RUNNER.jobs[jid]
    nb = RUNNER.nbs[j["nb"]]
    titles = {st["id"]: st["title"] for st in nb["stages"]}
    text = "".join(f"\n### {titles.get(e['stage'], e['stage'])}\n" if e["kind"] == "stage" else
                   (e["text"] if e["kind"] in ("stream", "text", "error", "error_stop") else "") for e in j["events"])[-12000:]
    q = (body or {}).get("question") or "Explain this run for the project team: what each step found, any warnings or errors, and what to do next."
    return gemma_or_note([
        dict(role="system", content="You are a 4D seismic expert reviewing the printed output of a team's notebooks ("
             + nb["title"] + "). Use only what the output shows; never invent numbers. Be concise and practical."),
        dict(role="user", content=f"Notebook output:\n{text}\n\n{q[:1000]}")])


# ------------------------------------------------------------------ results from the output working directory
def _ctx():
    s = settings()
    ws = results.Workspace(workspace_path())
    wells = results.read_wells(s.get("wells_dir"), s.get("well_list"), well_overrides(s.get("wells_dir"))) if s.get("wells_dir") else []
    plats = results.read_platforms(s.get("platform_csv")) if s.get("platform_csv") else []
    return ws, wells, plats


@app.get("/api/team/summary")
def team_summary():
    ws, wells, plats = _ctx()
    return results.summary(ws, wells, plats)


@app.get("/api/team/map")
def team_map(horizon: str, attr: str = "class"):
    ws, _, _ = _ctx()
    try:
        return results.raster(ws, horizon, attr)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------- time-slice QC (straight from the SEG-Y)
def _qc_vols(base: str = "", monitor: str = ""):
    st = settings()
    b, m = base or st.get("base"), monitor or st.get("monitor")
    return (safe(b) if b else None), (safe(m) if m else None)


def _qc_call(fn, base, monitor, vol, t, n, pid="", **kw):
    b, m = _qc_vols(base, monitor)
    slices.begin(pid)
    try:
        return fn(b, m, vol, t, n=n, cache_dir=OUT / "_cache", **kw)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    except ImportError:
        raise HTTPException(500, "segyio is not installed on this machine.")
    finally:
        slices.end()


@app.get("/api/qc/info")
def qc_info(base: str = "", monitor: str = ""):
    """Time range and size of the chosen volumes (reads only the headers)."""
    b, m = _qc_vols(base, monitor)
    out = {}
    for k, pth in (("baseline", b), ("monitor", m)):
        if pth:
            try:
                out[k] = slices.volume(pth, OUT / "_cache").meta()
            except Exception as e:
                out[k] = dict(error=str(e))
    return out


@app.get("/api/qc/progress")
def qc_progress(pid: str = ""):
    """What the running slice request is doing right now (stage text + seconds elapsed)."""
    return slices.progress(pid)


@app.get("/api/qc/slice")
def qc_slice(vol: str = "diff", t: float = 1000.0, base: str = "", monitor: str = "", n: int = 320, pid: str = "",
             t1: float | None = None, lag: float = 16.0, cc: float = 0.5, r0: float | None = None, r1: float | None = None):
    return _qc_call(slices.raster, base, monitor, vol, t, max(60, min(n, 700)), pid, t1=t1, lag=lag, cc=cc, r0=r0, r1=r1)


@app.get("/api/qc/slice3d")
def qc_slice3d(vol: str = "diff", t: float = 1000.0, base: str = "", monitor: str = "", n: int = 200, pid: str = "",
               t1: float | None = None, lag: float = 16.0, cc: float = 0.5, r0: float | None = None, r1: float | None = None):
    return _qc_call(slices.scene, base, monitor, vol, t, max(60, min(n, 400)), pid, t1=t1, lag=lag, cc=cc, r0=r0, r1=r1)


@app.post("/api/qc/slice/save")
def qc_slice_save(body: dict):
    """Write the shown slice (CSV of X, Y, value + PNG of the map) into <output folder>/QC_Slices."""
    ws = workspace_path(body.get("workspace"))
    team = Path(ROOT_FROM)
    if ws == team or ws in team.parents:
        raise HTTPException(400, f"Choose an output folder inside the team folder, not {ROOT_FROM} itself.")
    r = _qc_call(slices.raster, body.get("base", ""), body.get("monitor", ""), body.get("vol", "diff"), float(body.get("t", 1000)), 320, "",
                 **{k: body.get(k) for k in ("t1", "lag", "cc", "r0", "r1") if body.get(k) is not None})
    d = ws / "QC_Slices"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{body.get('vol', 'diff')}_{r['t_ms']:g}ms"
    if body.get("t1") is not None:
        stem = f"{body.get('vol')}_{float(body['t']):g}-{float(body['t1']):g}ms"
    slices.save_csv(r, d / f"{stem}.csv")
    saved = [f"{stem}.csv"]
    png = body.get("png") or ""
    if png.startswith("data:image/png;base64,"):
        import base64
        (d / f"{stem}.png").write_bytes(base64.b64decode(png.split(",", 1)[1]))
        saved.append(f"{stem}.png")
    return dict(folder=str(d), files=saved)


def _seis_bounds(base: str | None, compute: bool = False):
    """X/Y/time extent of the Baseline (from its trace headers). Only if already cached unless compute=True."""
    if not base:
        return None
    try:
        p = safe(base)
        m = (slices.volume(p, OUT / "_cache") if compute else slices.cached_volume(p, OUT / "_cache"))
        if m is None:
            return None
        m = m.meta()
        return dict(x=m["x_range"], y=m["y_range"], t=[m["t0_ms"], m["t1_ms"]])
    except Exception:
        return None


@app.get("/api/team/wellpaths")
def team_wellpaths(approx: bool = False):
    """Well trajectories and markers in TWT (checkshot + deviation, minimum curvature) and their horizon crossings."""
    st = settings()
    try:                                   # results folder only needed for horizon crossings
        ws = results.Workspace(workspace_path())
    except HTTPException:
        ws = None
    d, wl = st.get("wells_dir"), st.get("well_list")
    if not d:
        return dict(wells=[], notes=["No well folder is set (Wells card, or ask the Advisor to load the wells)."])
    out = wellpath.well_paths(safe(d) if d else None, safe(wl) if wl else None, _seis_bounds(st.get("base")), approx, well_overrides(safe(d)) if d else None)
    out = dict(wells=[dict(w) for w in out["wells"]], notes=list(out["notes"]))
    if ws is not None:
        try:
            wellpath.intersections(out, ws)
        except Exception as e:  # results incomplete: paths still useful
            out["notes"].append(f"Horizon crossings not computed ({type(e).__name__}).")
    for w in out["wells"]:
        w.get("crossings", []).sort(key=lambda c: c["twt"])
    return out


@app.get("/api/wells/files")
def wells_files(wells_dir: str = "", well_list: str = ""):
    """Every file in the well folder with what the app thinks it is and which well it belongs to (editable)."""
    st = settings()
    d = wells_dir or st.get("wells_dir")
    if not d:
        raise HTTPException(400, "Choose the well folder first.")
    p = safe(d)
    if not p.is_dir():
        raise HTTPException(400, "The well folder is not a folder.")
    wl = well_list or st.get("well_list")
    return wellpath.file_table(p, safe(wl) if wl else None, well_overrides(p))


@app.post("/api/wells/map")
def wells_map(body: dict):
    """Save a correction: body = {wells_dir, path, type?, well?}. Both empty (or equal to the guess) removes it."""
    p = safe(body.get("wells_dir") or settings().get("wells_dir") or "")
    rel = body.get("path") or ""
    if not rel or ".." in rel:
        raise HTTPException(400, "path required")
    with _SET_LOCK:
        s = settings()
        wm = s.setdefault("well_map", {})
        cur = wm.setdefault(str(p), {})
        guess = next((f for f in wellpath.file_table(p, None, {})["files"] if f["path"] == rel), None)
        ov = dict(cur.get(rel) or {})
        for k in ("type", "well"):
            if k in body:
                ov[k] = body[k]
        same_type = not ov.get("type") or (guess and ov["type"] == guess["type"])
        same_well = not ov.get("well") or (guess and ov["well"] == (guess["well"] or "-"))
        if guess and same_type and same_well:
            cur.pop(rel, None)
        else:
            cur[rel] = ov
        if not cur:
            wm.pop(str(p), None)
        _write_settings(s)
    return dict(ok=True, overrides=well_overrides(p))


@app.post("/api/wells/map/reset")
def wells_map_reset(body: dict):
    p = safe(body.get("wells_dir") or settings().get("wells_dir") or "")
    with _SET_LOCK:
        s = settings()
        (s.get("well_map") or {}).pop(str(p), None)
        _write_settings(s)
    return dict(ok=True)


@app.get("/api/wells/qc")
def wells_qc(wells_dir: str = "", well_list: str = "", base: str = "", approx: bool = False, seismic: bool = False):
    """Well tie check table: where each well's X/Y, KB, deviation, checkshot and markers come from, and whether it fits the seismic."""
    st = settings()
    d = wells_dir or st.get("wells_dir")
    if not d:
        raise HTTPException(400, "Choose the well folder first.")
    wl = well_list or st.get("well_list")
    b = base or st.get("base")
    try:
        out = wellpath.well_paths(safe(d), safe(wl) if wl else None, _seis_bounds(b, compute=seismic), approx, well_overrides(safe(d)))
    except OSError as e:
        raise HTTPException(400, str(e))
    wells = []
    for w in out["wells"]:
        w = dict(w); path = w.pop("path", None)
        w["has_path"] = bool(path)
        w["n_markers"] = len(w.pop("markers", []))
        wells.append(w)
    return dict(wells=wells, notes=out["notes"], seismic=out.get("seismic"), domain=out.get("domain"))


@app.get("/api/team/scene3d")
def team_scene3d(attr: str = "class", n: int | None = None):
    ws, _, _ = _ctx()
    try:
        return results.scene3d(ws, attr, min(max(n, 40), 400) if n else None)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


@app.get("/api/team/attrs")
def team_attrs(horizon: str):
    ws, _, _ = _ctx()
    return [dict(id=a, title=results.ATTRS[a][0]) for a in results.available_attrs(ws, horizon)]


@app.get("/api/team/figures")
def team_figures():
    ws, _, _ = _ctx()
    figs = ws.figures()
    runs = Path(ws.root) / "_app_runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
            for p in sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime):
                figs.append(dict(path=str(p.relative_to(ws.root)), title="Figure printed by 4D ChronoSeis", group=f"4D ChronoSeis run {d.name}", horizon=""))
    return figs


@app.get("/api/ws/file")
def ws_file(path: str):
    ws = workspace_path()
    p = (ws / path).resolve()
    if not (p == ws or ws in p.parents) or not p.is_file():
        raise HTTPException(404, "Not found in the output working directory.")
    return FileResponse(p)


def _team_context(sm: dict, limit: int = 14000) -> str:
    """Compact JSON of the results for Gemma: no well paths or per-bin arrays, so the prompt stays small and fast."""
    def r(d):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()
                if k not in ("score", "path", "md", "tvd", "twt", "points_xy", "markers", "logs", "fluid", "trajectory")
                and not (isinstance(v, list) and len(v) > 12)}
    hz = sorted(sm.get("horizons", []), key=lambda h: -(h.get("hardening_pct", 0) + h.get("softening_pct", 0) + h.get("sig_pct", 0)))
    wells = [dict(name=w.get("name"), x=round(w["x"]) if isinstance(w.get("x"), (int, float)) else w.get("x"),
                  y=round(w["y"]) if isinstance(w.get("y"), (int, float)) else w.get("y")) for w in (sm.get("wells") or [])]
    plats = [dict(name=p.get("name"), x=round(p["x"]), y=round(p["y"])) for p in (sm.get("platforms") or []) if "x" in p]
    ctx = dict(quality=sm.get("quality"), background_nrms=sm.get("background_nrms"), thresholds=sm.get("thresholds"),
               changed_horizons=sm.get("changed_horizons"), n_horizons=len(hz), horizons=[r(h) for h in hz],
               untapped=[r(u) for u in sm.get("untapped", [])][:8], platforms=plats, wells=wells,
               well_fluids=sm.get("wells_fluid") or None, rfi=sm.get("rfi"))
    out = json.dumps(ctx, default=str)
    while len(out) > limit and len(ctx["horizons"]) > 6:      # keep the most changed horizons
        ctx["horizons"] = ctx["horizons"][:-2]
        ctx["note"] = f"Only the {len(ctx['horizons'])} most changed of {len(hz)} horizons are listed."
        out = json.dumps(ctx, default=str)
    if len(out) > limit:
        ctx["well_fluids"] = None
        ctx["rfi"] = (sm.get("rfi") or {}) and {k: v for k, v in (sm.get("rfi") or {}).items() if k != "hotspots"}
        out = json.dumps(ctx, default=str)
    return out[:limit * 2]


SYSTEM = ("You are a careful 4D (time-lapse) seismic interpreter helping a reservoir team find changes and possibly "
          "bypassed oil and gas. You get per-horizon statistics from the team's 4D ChronoSeis engine: NdRMS = 200*(RMS_Baseline - "
          "RMS_Monitor)/(RMS_Baseline + RMS_Monitor) (positive = monitor dimmer), NRMS, and the notebook's classes "
          "(hardening / softening / unswept, platform-constrained). 'rfi' (when present) is the team's Phase 6 Remaining Fluid Index: a relative 0-100 screening score per horizon (not saturation or reserves), with per-horizon statistics and ranked hotspots (area, mean RFI, TWT, nearest well and platform). 'well_fluids' (when present) holds the wells' oil, gas and water "
          "intervals in TWT from the team's Phase 4 classification and Phase 5 correlation: counts and thickness in ms per well, the "
          "thickest hydrocarbon intervals with their nearest marker, and the number of proposed markers; these are machine-learning "
          "classes from logs, not tested fluids. The notebook labels positive NdRMS as hardening; that "
          "polarity is not yet calibrated. 'untapped' lists unswept, bright-on-baseline areas next to changed areas. "
          "Coordinates are assumed to share one system. Use ONLY these numbers; never invent wells, fluids or production. "
          "Treat causes as hypotheses. Be concise.")


@app.post("/api/team/interpret")
def team_interpret():
    ws, wells, plats = _ctx()
    sm = results.summary(ws, wells, plats)
    if not sm.get("ready"):
        raise HTTPException(400, "No results yet. Run the workflow first.")
    return gemma_or_note([dict(role="system", content=SYSTEM),
                          dict(role="user", content="Results (JSON):\n" + _team_context(sm) + "\n\nWrite a short interpretation "
                               "for a manager: (1) data quality, (2) which horizons changed and how, (3) possible untapped "
                               "areas and how to confirm them, (4) recommended next steps.")])


class ChatReq(BaseModel):
    question: str


class HzReq(BaseModel):
    horizon: str
    question: str = ""


def _horizon_context(sm: dict, ws, h: str) -> str:
    """One horizon in detail: its statistics, the companion-map numbers, its untapped areas and RFI hotspots."""
    row = next((x for x in sm.get("horizons", []) if x.get("horizon") == h), None)
    if row is None:
        raise HTTPException(404, f"Horizon {h} not found in the results.")
    maps = {}
    for a in ("RMS_Baseline", "NRMS", "dRMS", "NdRMS_pct", "RFI_Final", "class"):
        try:
            r = results.raster(ws, h, a)
            st = r.get("stats") or {}
            maps[a] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()} if st else {
                k: v for k, v in r.items() if k in ("counts", "classes", "legend")}
        except Exception:
            pass
    rf = sm.get("rfi") or {}
    rf_h = dict(per_horizon=[x for x in rf.get("horizons", []) if x.get("horizon") == h],
                hotspots=[x for x in rf.get("hotspots", []) if x.get("horizon") == h]) if rf else None
    rank = (sm.get("ranked") or []).index(h) + 1 if h in (sm.get("ranked") or []) else None
    ctx = dict(horizon=h, rank_by_change=rank, of_horizons=len(sm.get("horizons", [])), thresholds=sm.get("thresholds"),
               background_nrms=sm.get("background_nrms"), stats=row, companion_maps=maps,
               untapped_here=[u for u in sm.get("untapped", []) if u.get("horizon") == h], rfi=rf_h,
               platforms=[dict(name=p.get("name"), x=round(p["x"]), y=round(p["y"])) for p in sm.get("platforms") or [] if "x" in p],
               wells=[dict(name=w.get("name"), x=w.get("x"), y=w.get("y")) for w in sm.get("wells") or []])
    return json.dumps(ctx, default=lambda v: round(v, 2) if isinstance(v, float) else str(v))[:14000]


@app.post("/api/team/interpret_horizon")
def team_interpret_horizon(req: HzReq):
    ws, wells, plats = _ctx()
    sm = results.summary(ws, wells, plats)
    if not sm.get("ready"):
        raise HTTPException(400, "No results yet. Run the workflow first.")
    q = req.question.strip()[:1000] or (
        "Interpret this horizon for a reservoir team: (1) how reliable the 4D is here (NRMS), (2) where the sand is (Baseline RMS) "
        "and where change was seen (dRMS/NdRMS, hardening/softening), (3) where potential may remain (unswept + bright + RFI, "
        "untapped areas with coordinates), (4) what to check next. Read the companion maps together. 6-10 sentences.")
    return gemma_or_note([dict(role="system", content=SYSTEM),
                          dict(role="user", content=f"Horizon details (JSON):\n{_horizon_context(sm, ws, req.horizon)}\n\n{q}")])


@app.post("/api/team/chat")
def team_chat(req: ChatReq):
    ws, wells, plats = _ctx()
    sm = results.summary(ws, wells, plats)
    return gemma_or_note([dict(role="system", content=SYSTEM + "\nResults JSON:\n" + _team_context(sm)),
                          dict(role="user", content=req.question[:1000])])


# ------------------------------------------------------------------ A-A' section with wells and logs
def _wells_now(compute_bounds=False):
    st = settings()
    d, wl = st.get("wells_dir"), st.get("well_list")
    if not d:
        return []
    out = wellpath.well_paths(safe(d), safe(wl) if wl else None, _seis_bounds(st.get("base"), compute=compute_bounds), False, well_overrides(safe(d)))
    ws = results.Workspace(workspace_path())
    for w in out["wells"]:
        try:
            w["fluid"] = results.fluid_for(ws, w["name"])
        except Exception:
            w["fluid"] = []
    return out["wells"]


def _section_point(p: str):
    """'Well-5' (well name, any case/spacing) or 'X,Y' -> [x, y]."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*", p or "")
    if m:
        return [float(m.group(1)), float(m.group(2))]
    key = wellpath._wkey(p)
    for w in _wells_now():
        if wellpath._wkey(w["name"]) == key or w.get("key") == key:
            if w.get("x") is None:
                raise HTTPException(400, f"{w['name']} has no X/Y.")
            return [float(w["x"]), float(w["y"])]
    raise HTTPException(404, f"No well called {p!r} is loaded. Load the wells first, or give X,Y.")


def _horizon_grids():
    try:
        ws = results.Workspace(workspace_path())
        S = results.scene3d(ws, "TWT")
    except Exception:
        return None
    grids = []
    for h in S["horizons"]:
        z = np.frombuffer(base64.b64decode(h["z"]), np.float32).reshape(S["ny"], S["nx"])
        grids.append((h["name"], z))
    return dict(x0=S["x0"], y0=S["y0"], cell=S["cell"], nx=S["nx"], ny=S["ny"], grids=grids)


@app.get("/api/section")
def api_section(a: str, b: str, vol: str = "diff", base: str = "", monitor: str = "", t0: float | None = None, t1: float | None = None, pid: str = ""):
    B, M = _qc_vols(base, monitor)
    pa, pb = _section_point(a), _section_point(b)
    slices.begin(pid)
    try:
        r = secmod.section(B, M, vol, pa, pb, OUT / "_cache", t0, t1, wells=_wells_now(), horizons=_horizon_grids())
        r["a_name"], r["b_name"] = a, b
        return r
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    finally:
        slices.end()


@app.get("/api/wells/names")
def wells_names():
    return [dict(name=w["name"], x=w.get("x"), y=w.get("y"), las=bool(w.get("las_path"))) for w in _wells_now() if w.get("x") is not None]


# ------------------------------------------------------------------ ChronoSeis Advisor (Gemma drives the app via JSON actions)
SEGY_EXT = (".sgy", ".segy", ".seg")


def _default_dir(sub: str) -> Path:
    for c in (Path(CHALLENGE) / sub, Path(CHALLENGE)):
        try:
            p = safe(str(c))
            if p.is_dir():
                return p
        except HTTPException:
            pass
    raise HTTPException(404, f"The challenge folder {CHALLENGE}/{sub} is not available here; give a folder.")


def _role_guess(name: str):
    n = name.lower()
    if re.search(r"base|baseline|1995|_bl\b|pre", n):
        return "baseline"
    if re.search(r"monitor|_mon|2006|post|repeat", n):
        return "monitor"
    return None


def a_find_segy(folder: str = ""):
    root = safe(folder) if folder else _default_dir("volumes")
    files = []
    for c in _walk(root, max_depth=2, limit=5000):
        if c.is_file() and c.suffix.lower() in SEGY_EXT:
            files.append(dict(path=str(c), name=c.name, size_mb=round(c.stat().st_size / 1e6, 1), role_guess=_role_guess(c.name)))
    files.sort(key=lambda f: f["name"])
    guess = {}
    for r in ("baseline", "monitor"):
        hits = [f["path"] for f in files if f["role_guess"] == r]
        if len(hits) == 1:
            guess[r] = hits[0]
    if len(files) == 2 and not guess:
        guess = dict(baseline=files[0]["path"], monitor=files[1]["path"], note="guessed by name order")
    return dict(result=dict(folder=str(root), files=files, guess=guess))


def a_set_volumes(baseline: str, monitor: str):
    out, rows = {}, []
    for k, pth in (("base", baseline), ("monitor", monitor)):
        slices.note(f"Reading the {'Baseline' if k == 'base' else 'Monitor'} headers…")
        out[k] = inspect(InspectReq(path=pth))
        out[k]["path"] = str(safe(pth))
    a, b = out["base"], out["monitor"]
    upd = dict(base=a["path"], monitor=b["path"])
    if a["domain_hint"] != "depth" and b["domain_hint"] != "depth":
        upd["seismic_domain"] = "time"
    save_settings(**upd)
    find = []
    if a["samples"] == b["samples"] and a["dt_ms"] == b["dt_ms"] and a["t_start_ms"] == b["t_start_ms"]:
        find.append(f"Same time range ({a['t_start_ms']:g}-{a['t_end_ms']:g} ms) and sampling ({a['dt_ms']:g} ms): ready for differencing.")
    else:
        find.append("WARNING: different time sampling or range; the engine stops at the matching step.")
    if (a["il_min"], a["il_max"], a["xl_min"], a["xl_max"]) == (b["il_min"], b["il_max"], b["xl_min"], b["xl_max"]):
        find.append("Same inline/crossline grid.")
    else:
        find.append("The inline/crossline ranges differ: only common bins are compared.")
    if a["traces"] != b["traces"]:
        find.append(f"Trace counts differ ({a['traces']:,} vs {b['traces']:,}).")
    if "seismic_domain" not in upd:
        find.append("A file hints at depth: confirm the domain in the Seismic volumes card.")
    for lab, key in (("File", "file"), ("Size", "size_mb"), ("Traces", "traces"), ("Time (ms)", None), ("Sample (ms)", "dt_ms"), ("Inlines", None), ("Crosslines", None)):
        f = lambda v: (f"{v['t_start_ms']:g}–{v['t_end_ms']:g}" if lab.startswith("Time") else f"{v['il_min']}–{v['il_max']}" if lab == "Inlines"
                       else f"{v['xl_min']}–{v['xl_max']}" if lab == "Crosslines" else f"{v[key]:,} MB".replace(",", ",") if key == "size_mb"
                       else f"{v[key]:,}" if key == "traces" else str(v[key]))
        rows.append([lab, f(a), f(b)])
    mid = round((a["t_start_ms"] + a["t_end_ms"]) / 2 / max(a["dt_ms"], 1)) * max(a["dt_ms"], 1)
    return dict(result=dict(baseline={k: a[k] for k in a if k != "domain_reason"}, monitor={k: b[k] for k in b if k != "domain_reason"}, findings=find),
                card=dict(type="table", title="Seismic volumes", cols=["", "Baseline", "Monitor"], rows=rows,
                          checks=[dict(level="warn" if f.startswith(("WARNING", "The inline", "Trace", "A file")) else "ok", text=f.replace("WARNING: ", "")) for f in find]),
                ui=[dict(refresh_inputs=True)],
                buttons=[dict(label=f"Show Monitor − Baseline at {mid:g} ms", ui=dict(slice=dict(vol="diff", t=mid)))])


def _find_well_list(folder: Path):
    best = None
    for c in _walk(folder, max_depth=3, limit=5000):
        if c.is_file() and c.suffix.lower() in (".xlsx", ".xls", ".csv") and re.search(r"well.?list|well.?info", c.name, re.I):
            best = best or c
    return best


def a_find_wells(folder: str = ""):
    root = safe(folder) if folder else _default_dir("well_data")
    wl = _find_well_list(root)
    tab = wellpath.file_table(root, wl, well_overrides(root))
    counts = {}
    for f in tab["files"]:
        counts[f["type_final"]] = counts.get(f["type_final"], 0) + 1
    unsure = [f["name"] for f in tab["files"] if f.get("conf") == "low" and not f.get("overridden")]
    return dict(result=dict(folder=str(root), well_list=str(wl) if wl else None, wells=[w["name"] for w in tab["wells"]],
                            file_types=counts, unsure_files=unsure[:15]))


def a_load_wells(folder: str, well_list: str = ""):
    root = safe(folder)
    wl = safe(well_list) if well_list else _find_well_list(root)
    save_settings(wells_dir=str(root), **({"well_list": str(wl)} if wl else {}))
    slices.note("Reading the well files and converting to TWT…")
    qc = wells_qc(wells_dir=str(root), well_list=str(wl) if wl else "", seismic=False)
    listed = wellpath.read_well_list(wl) if wl else {}
    W = []
    for w in qc["wells"]:
        lw = listed.get(wellpath._wkey(w["name"]), {})
        W.append(dict(name=w["name"], status=w["status"], type=lw.get("type"), listed_data=lw.get("listed_data"),
                      x=w.get("x"), y=w.get("y"), kb=w.get("kb"), deviation=w.get("deviation"), checkshot=w.get("checkshot"),
                      markers=w.get("n_markers"), twt_range=w.get("twt_range"),
                      issues=[c["text"] for c in w["checks"] if c["level"] != "ok"][:4]))
    n = len(W)
    ok = sum(w["status"] == "ok" for w in W)
    warn = sum(w["status"] == "warn" for w in W)
    bad = n - ok - warn
    types = sorted({w["type"] for w in W if w["type"]})
    head = f"{n} wells loaded and converted to TWT: {ok} OK, {warn} with warnings, {bad} with problems." + (f" Types: {', '.join(types)}." if types else "")
    checks = [dict(level=("ok" if w["status"] == "ok" else w["status"]), text=f"{w['name']}: " + (
        "; ".join(w["issues"]) if w["issues"] else f"{w['markers'] or 0} markers" + (f", TWT {w['twt_range'][0]:.0f}–{w['twt_range'][1]:.0f} ms" if w["twt_range"] else "")))
        for w in W]
    return dict(result=dict(headline=head, wells=W, notes=qc.get("notes", [])[:5]),
                card=dict(type="checks", title="Wells in TWT", checks=checks, note=head),
                ui=[dict(refresh_inputs=True), dict(wellqc=True)])


def _is_platform_csv(p: Path) -> bool:
    try:
        return p.suffix.lower() == ".csv" and p.stat().st_size < 200_000 and len(results.read_platforms(p)) > 0
    except Exception:
        return False


def _is_horizon_dir(d: Path) -> int:
    try:
        fs = sorted(d.glob("*.txt"))[:50]
        if not fs:
            return 0
        import pandas as pd
        cols = set(pd.read_csv(fs[0], sep="\t", nrows=2).columns.map(lambda c: str(c).strip()))
        return len(list(d.glob("*.txt"))) if {"X", "Y", "Inline", "Crossline", "TWT"} <= cols else 0
    except Exception:
        return 0


def a_find_inputs(folder: str = ""):
    """Look for a platform coordinate CSV and folders of horizon .txt files."""
    roots = [safe(folder)] if folder else []
    if not roots:
        for c in (HERE / "inputs", Path(ROOT_FROM) / "HTML_TEST" / "inputs", Path(ROOT_FROM) / "HTML_TEST", Path(CHALLENGE),
                  Path(CHALLENGE).parent, Path("/data/epic20/APGCE/phase6_horizons"), Path(ROOT_FROM)):
            try:
                p = safe(str(c))
                if p.is_dir() and p not in roots:
                    roots.append(p)
            except HTTPException:
                pass
    plats, hdirs, seen = [], [], set()
    for root in roots:
        for c in _walk(root, max_depth=3, limit=6000):
            if str(c) in seen:
                continue
            seen.add(str(c))
            if c.is_file() and re.search(r"platform", c.name, re.I) and _is_platform_csv(c):
                plats.append(dict(path=str(c), platforms=[p["name"] for p in results.read_platforms(c)]))
            elif c.is_dir() and re.search(r"horizon|autopick", c.name, re.I) and (n := _is_horizon_dir(c)):
                hdirs.append(dict(path=str(c), files=n))
    return dict(result=dict(searched=[str(r) for r in roots], platform_files=plats[:6], horizon_folders=hdirs[:6],
                            hint="Use set_inputs with a path from these lists." if plats or hdirs else
                            "Nothing found. Ask the user for the path, or to upload Platforms.csv to HTML_TEST/inputs."))


def a_set_inputs(platform_csv: str = "", horizons_dir: str = ""):
    upd, msg = {}, []
    if platform_csv:
        p = safe(platform_csv)
        if not _is_platform_csv(p):
            raise HTTPException(400, f"{p.name} is not a platform CSV (needs X(m) and Y(m) columns).")
        upd["platform_csv"] = str(p); msg.append(f"Platforms: {len(results.read_platforms(p))} from {p.name}.")
    if horizons_dir:
        d = safe(horizons_dir)
        n = _is_horizon_dir(d)
        if not n:
            raise HTTPException(400, f"{d} has no horizon .txt files (tab-separated X, Y, Inline, Crossline, TWT).")
        upd["horizons_dir"] = str(d); msg.append(f"Existing horizons: {n} files in {d.name}; horizon picking will be skipped.")
    if not upd:
        raise HTTPException(400, "Give platform_csv and/or horizons_dir.")
    save_settings(**upd)
    return dict(result=dict(set=upd, notes=msg), ui=[dict(refresh_inputs=True)])


def a_search_papers(query: str):
    import knowledge
    hits = knowledge.search(query, 4)
    if not hits:
        return dict(result=dict(passages=[], note="No passage in the reference papers matches this. Answer from general knowledge and say so."))
    return dict(result=dict(passages=[dict(source=h["source"], page=h["page"], text=h["text"][:1100]) for h in hits]))


def a_list_projects():
    seen, out = set(), []
    for r in settings().get("recent_projects", []):
        p = _allowed(r.get("path"))
        if p and p.is_file() and str(p) not in seen:
            seen.add(str(p)); out.append(dict(name=r.get("name") or p.stem, path=str(p), saved=r.get("opened")))
    for root in [Path(ROOT_FROM), OUT]:
        try:
            root = safe(str(root))
        except HTTPException:
            continue
        for c in _walk(root, max_depth=3, limit=20000):
            if c.is_file() and c.name.endswith(PROJ_EXT) and str(c) not in seen:
                seen.add(str(c))
                try:
                    j = json.loads(c.read_text())
                    out.append(dict(name=j.get("name") or c.stem, path=str(c), saved=j.get("saved_at")))
                except Exception:
                    pass
    return dict(result=dict(projects=out[:20]))


def a_open_project(path: str):
    r = project_open(dict(path=path))
    if r.get("choose"):
        return dict(result=dict(choose=r["choose"], note="Several project files in that folder; pick one."))
    if not r.get("workspace_ok"):
        raise HTTPException(400, "The results folder of this project is not available; open it from Open project to relocate it.")
    P = r["project"]
    project_apply(dict(project=P, path=r["path"], relocated=r.get("relocated") or []))
    snap = P.get("results_snapshot") or {}
    return dict(result=dict(name=P.get("name"), saved_at=P.get("saved_at"), results_folder=P.get("workspace"),
                            horizons=snap.get("horizons"), classes=snap.get("classes"), missing_inputs=r.get("missing") or []),
                card=dict(type="checks", title=f"Project: {P.get('name')}", note=f"Saved {P.get('saved_at')} · results computed earlier",
                          checks=[dict(level="ok", text=f"Results folder: {P.get('workspace')}")] +
                                 [dict(level="warn", text=f"Not found: {m}") for m in (r.get("missing") or [])]),
                ui=[dict(project=dict(path=r["path"], name=P.get("name"))), dict(refresh_inputs=True), dict(view="summary")])


def a_reuse_team_results():
    r = reuse_team_results()
    st = r["status"]
    return dict(result=dict(copied_files=r["copied_files"], already_there=r["skipped_existing"], horizons=st.get("horizons"), classes=st.get("classes"),
                            note="Team's earlier results copied into the results folder (nothing overwritten); the team folder was only read."),
                card=dict(type="checks", title="Earlier team results brought in", note=f"From {r['source']} (read only)",
                          checks=[dict(level="ok", text=f"{r['copied_files']} files copied, {r['skipped_existing']} already present"),
                                  dict(level="ok" if st.get("horizons") else "warn", text=f"{st.get('horizons', 0)} horizons, {st.get('classes', 0)} class files in the results folder")]),
                ui=[dict(refresh_inputs=True), dict(view="summary")])


def a_get_results():
    try:
        ws, wells, plats = _ctx()
        sm = results.summary(ws, wells, plats)
    except HTTPException as e:
        return dict(result=dict(ready=False, reason=e.detail))
    if not sm.get("ready"):
        return dict(result=dict(ready=False, reason="No engine results in the output folder yet."))
    return dict(result=dict(ready=True, **json.loads(_team_context(sm))))


def a_show_view(view: str):
    v = view.lower()
    v = {"data viewer": "data", "viewer": "data", "data view": "data"}.get(v, v)
    if v not in ("inputs", "data", "summary", "detail"):
        raise HTTPException(400, "view must be inputs, data, summary or detail")
    return dict(result=dict(shown=v), ui=[dict(view=v)])


def a_show_slice(vol: str, t: float, t1: float | None = None):
    if vol not in ("diff", "baseline", "monitor", "tshift", "tcc"):
        raise HTTPException(400, "vol must be diff, baseline, monitor, tshift or tcc")
    if vol in ("tshift", "tcc") and not t1:
        t1 = t + 150
    return dict(result=dict(shown=dict(vol=vol, t=t, t1=t1)), ui=[dict(slice=dict(vol=vol, t=t, t1=t1))])


def a_show_section(a: str, b: str, vol: str = "diff"):
    pa, pb = _section_point(a), _section_point(b)
    return dict(result=dict(shown=dict(a=a, b=b, a_xy=pa, b_xy=pb, vol=vol)), ui=[dict(section=dict(a=a, b=b, vol=vol))])


def _match_horizon(ws, name: str) -> str:
    hs = ws.horizons()
    if not hs:
        raise HTTPException(400, "No horizons in the results folder yet.")
    if not name:
        return hs[0]
    k = results._norm(name)
    for h in hs:
        if results._norm(h) == k:
            return h
    hit = [h for h in hs if k in results._norm(h) or results._norm(h) in k]
    if len(hit) == 1:
        return hit[0]
    raise HTTPException(404, f"Horizon '{name}' not found. Available: {', '.join(hs)}.")


def a_scan_offsets(horizon: str = "", step_ms: float = 4.0, max_ms: float = 150.0):
    step_ms, max_ms = max(2.0, min(float(step_ms), 40.0)), max(10.0, min(float(max_ms), 400.0))
    b, m = _qc_vols()
    if not (b and m):
        raise HTTPException(400, "Set the Baseline and Monitor SEG-Y first (the scan reads both volumes).")
    ws, wells, plats = _ctx()
    h = _match_horizon(ws, horizon)
    B, M = slices.volume(b, OUT / "_cache"), slices.volume(m, OUT / "_cache")
    r = offsets.scan(B, M, ws, h, wells=wells, platforms=plats, step_ms=step_ms, max_ms=max_ms)
    xs, ys = r["offsets"], r["median_nrms"]
    brief = dict(horizon=h, step_ms=r["step_ms"], max_ms=r["max_ms"], window_ms=r["window_ms"], traces_used=r["traces"],
                 nrms_at_horizon=r["at_horizon_nrms"], nrms_background_away_from_horizon=r["background_nrms"],
                 peaks_in_median_nrms=r["peaks"], clusters_away_from_horizon=r["clusters"], cluster_threshold_nrms=r["cluster_threshold"],
                 profile=[[xs[i], ys[i]] for i in range(0, len(xs), max(1, len(xs) // 16))],
                 note=f"NRMS = 200*RMS(Monitor-Baseline)/(RMS_B+RMS_M) in {r['window_ms']} ms windows every {r['step_ms']} ms; the engine's horizon maps use +-10 ms "
                      f"around the pick, so anything beyond about {r['guard_ms']:.0f} ms from it is new. 'over_known_change_pct' is the share of the area already classed hardening/softening on the horizon. "
                      "Off-horizon response can be real, but noise, multiples or processing differences also cause it: check NRMS repeatability and the section view before interpreting.")
    rows = [[f"A{c['rank']}", f"{c['from_ms']:+.0f} to {c['to_ms']:+.0f}", f"{c['twt_ms']:.0f}", c["mean_nrms"], c["area_km2"],
             (f"{c['over_known_change_pct']:.0f}%" if "over_known_change_pct" in c else "–"), (f"{c['nearest_well']} {c['well_km']} km" if "nearest_well" in c else "–")]
            for c in r["clusters"]]
    card = dict(type="table", title=f"4D response every {r['step_ms']:g} ms, ±{r['max_ms']:g} ms around {h}",
                note=f"Median NRMS {r['at_horizon_nrms']} at the horizon, {r['background_nrms']} away from it. {len(r['peaks'])} offset band(s) stand out; "
                     f"{len(r['clusters'])} localized anomaly cluster(s) away from the horizon.",
                cols=["#", "Offset (ms)", "TWT (ms)", "Mean NRMS", "Area km²", "Already changed on horizon", "Nearest well"], rows=rows,
                profile=dict(x=xs, y=ys, p90=r["p90_nrms"], bg=r["background_nrms"], sigma=r["sigma"], peaks=r["peaks"], guard=r["guard_ms"]))
    return dict(result=brief, card=card)


def a_propose_run(what: str):
    return dict(result=dict(offered=what, note="Shown as a button; nothing runs until the user presses it."),
                buttons=[dict(label="Go to Run the workflow", ui=dict(view="inputs", scroll="run"))])


ADVISOR_HANDLERS = dict(find_segy=a_find_segy, set_volumes=a_set_volumes, find_wells=a_find_wells, load_wells=a_load_wells,
                        list_projects=a_list_projects, open_project=a_open_project, get_results=a_get_results,
                        show_view=a_show_view, show_slice=a_show_slice, show_section=a_show_section, propose_run=a_propose_run,
                        reuse_team_results=a_reuse_team_results, scan_offsets=a_scan_offsets,
                        find_inputs=a_find_inputs, set_inputs=a_set_inputs,
                        search_papers=a_search_papers)


def _advisor_state(view: str = ""):
    s = settings()
    nm = lambda k: Path(s[k]).name if s.get(k) else None
    st = dict(view=view or None, challenge_folder=CHALLENGE, baseline=nm("base"), monitor=nm("monitor"),
              seismic_domain=s.get("seismic_domain"), wells_folder=s.get("wells_dir"), well_list=nm("well_list"),
              results_folder=s.get("workspace"), project=None, results_ready=False,
              markers_csv=s.get("markers_csv"), platform_file=s.get("platform_csv"), horizon_folder=s.get("horizons_dir"))
    try:
        if s.get("platform_csv"):
            st["platforms"] = [f'{p["name"]} (X {p["x"]:.0f}, Y {p["y"]:.0f})' for p in results.read_platforms(safe(s["platform_csv"]))]
        hd = Path(s["horizons_dir"]) if s.get("horizons_dir") else None
        if hd and hd.is_dir():
            st["horizon_files"] = sorted(f.stem for f in hd.glob("*.txt"))
    except Exception:
        pass
    if s.get("project"):
        rr = next((x for x in s.get("recent_projects", []) if x.get("path") == s["project"]), None)
        st["project"] = (rr or {}).get("name") or Path(s["project"]).stem
    try:
        if s.get("workspace"):
            stt = results.Workspace(safe(s["workspace"])).status()
            st["results_ready"] = bool(stt.get("horizons"))
            st["horizons"] = stt.get("horizons")
    except Exception:
        pass
    return st


def _ask_json(messages):
    body = json.dumps(dict(model=GEMMA_MODEL, messages=messages, temperature=0.2, max_tokens=900)).encode()
    req = urllib.request.Request(GEMMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


@app.post("/api/advisor")
def advisor_chat(body: dict):
    msg = (body.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "Empty message.")
    slices.begin(body.get("pid") or "")
    try:
        return advisor.run(msg, body.get("history") or [], _advisor_state(body.get("view", "")), ADVISOR_HANDLERS, _ask_json, note=slices.note)
    finally:
        slices.end()


@app.get("/api/advisor/status")
def advisor_status():
    url = GEMMA_URL.rsplit("/chat/completions", 1)[0] + "/models"
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            r.read(2000)
        return dict(ok=True, model=GEMMA_MODEL, ms=round(1000 * (time.time() - t0)))
    except Exception as e:
        return dict(ok=False, model=GEMMA_MODEL, reason=type(e).__name__)


# ------------------------------------------------------------------ page
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.on_event("shutdown")
def _shutdown():
    RUNNER.shutdown()


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("APP4D_PORT", "8050"))
    print(f"Allowed folders: {[str(r) for r in ROOTS]}\nNotebooks: {REPO}\nOpen http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
