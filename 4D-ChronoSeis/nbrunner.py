"""
Run the team's GitHub notebooks stage by stage in Jupyter kernels on the HPC.

- The .ipynb files are only READ. Cells are sent unchanged to a kernel, except for the
  run-time substitutions declared in pipeline.json ('replace' literals, optional output-root
  mapping) and the extra 'inject_after' lines (e.g. the Baseline/Monitor chosen in the GUI).
  Every substitution is reported in the run log.
- One kernel per notebook, kept alive between stages so variables carry over (like JupyterLab).
- Kernels are started from the notebook's own kernelspec (e.g. the "Python 3.12 GPU via SLURM"
  kernel for the GPU notebook), falling back to python3 if that kernelspec isn't installed.
"""
from __future__ import annotations

import base64
import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _build_overlay(ov, params):
    """Make <workdir>/team_root: one symlink per entry of the team folder (inputs, read-only use),
    except this notebook's own output names, which become real folders/files here. The team's
    existing results are never written to; the notebooks stay unmodified."""
    import os
    src, dst = Path(ov["from"]), Path(params["workdir"]) / ov.get("dir", "team_root")
    dst.mkdir(parents=True, exist_ok=True)
    outs = set(ov.get("outputs", []))
    if not src.is_dir():
        return [f"Path check: team folder {src} not found on this machine. The notebook will stop at its first read."]
    linked, seeded = 0, []
    import shutil
    for name in ov.get("seed", []):   # outputs this notebook also reads (e.g. a trained model): start from a COPY of the team's
        s, t = src / name, dst / name
        if t.is_symlink():
            t.unlink()
        if s.exists() and not t.exists():
            (shutil.copytree(s, t, symlinks=False) if s.is_dir() else shutil.copy2(s, t)); seeded.append(name)
    for p in sorted(src.iterdir()):
        t = dst / p.name
        if p.name in outs:
            if t.is_symlink():
                t.unlink()
            continue
        if t.is_symlink() and not t.exists():
            t.unlink()
        if not (t.exists() or t.is_symlink()):
            os.symlink(p, t); linked += 1
    return ([f"Copied the team's {', '.join(seeded)} into {dst} as a starting point (the team's copy is not touched)."] if seeded else []) + [
            f"Path check: inputs are read from {src} (linked {linked} new entries). "
            f"Outputs ({', '.join(sorted(outs))}) are written to {dst}. The team's own copies are not touched."]


class Session:
    def __init__(self, nb_cfg: dict, repo: Path, work: Path):
        self.cfg = nb_cfg
        self.path = repo / nb_cfg["file"]
        self.work = work / nb_cfg["id"]
        self.work.mkdir(parents=True, exist_ok=True)
        self.km = None
        self.kc = None
        self.kernel_name = None
        self.kernel_note = ""
        self.stage_status: dict[str, str] = {}
        self.lock = threading.Lock()
        self.busy = False

    # ---------------------------------------------------------------- kernel
    def start(self, cpu: bool = False):
        from jupyter_client.kernelspec import NoSuchKernel
        from jupyter_client.manager import KernelManager
        if self.km is not None and self.km.is_alive():
            return
        wanted = self.cfg.get("kernel") or "python3"
        order = [wanted, self.cfg.get("fallback_kernel", "python3")]
        if cpu:   # stages that need no GPU: don't wait for a SLURM GPU slot
            order = order[::-1]
        for name in order:
            try:
                km = KernelManager(kernel_name=name)
                km.start_kernel(cwd=str(self.work))
                kc = km.client()
                kc.start_channels()
                kc.wait_for_ready(timeout=600)  # a SLURM GPU kernel may queue for a while
                self.km, self.kc, self.kernel_name = km, kc, name
                self.kernel_note = "" if name == wanted else (
                    f"Using the CPU kernel '{name}' (these stages need no GPU)." if cpu else
                    f"Kernel '{wanted}' is not available here; using '{name}' instead. GPU cells may fail.")
                self.stage_status = {}
                return
            except NoSuchKernel:
                continue
        raise RuntimeError(f"No usable Jupyter kernel found (tried {wanted!r}).")

    def restart(self):
        if self.km is not None:
            try:
                self.kc.stop_channels()
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass
        self.km = self.kc = None
        self.stage_status = {}   # the next run starts a fresh kernel (CPU or GPU, as its stages need)

    def interrupt(self):
        if self.km is not None:
            self.km.interrupt_kernel()

    def alive(self):
        return self.km is not None and self.km.is_alive()

    # ---------------------------------------------------------------- execute one cell
    def execute(self, code: str, emit, cancel: threading.Event, img_dir: Path, img_url: str) -> bool:
        kc = self.kc
        msg_id = kc.execute(code, store_history=True, allow_stdin=False, stop_on_error=True)
        ok = True
        while True:
            if cancel.is_set():
                self.interrupt()
                cancel.clear()
                emit("note", "Stop requested: kernel interrupted.")
            try:
                msg = kc.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                if not self.alive():
                    emit("error", "The kernel died (often: out of memory). Restart the kernel and run again.")
                    return False
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c.get("execution_state") == "idle":
                break
            if t == "stream":
                emit("stream", ANSI.sub("", c.get("text", "")), name=c.get("name"))
            elif t in ("display_data", "execute_result"):
                d = c.get("data", {})
                if "image/png" in d:
                    fn = f"img_{uuid.uuid4().hex[:10]}.png"
                    (img_dir / fn).write_bytes(base64.b64decode(d["image/png"]))
                    emit("image", f"{img_url}/{fn}")
                elif "application/vnd.jupyter.widget-view+json" in d:
                    emit("note", "Interactive widget: open this notebook in JupyterLab to use it.")
                elif "text/plain" in d:
                    txt = d["text/plain"]
                    if not re.fullmatch(r"<Figure size .*>", txt.strip()):
                        emit("text", ANSI.sub("", txt))
            elif t == "error":
                ok = False
                emit("error", ANSI.sub("", "\n".join(c.get("traceback", [])) or f"{c.get('ename')}: {c.get('evalue')}"))
        return ok


class Runner:
    def __init__(self, repo: Path, manifest: Path, work: Path, out: Path):
        self.repo, self.work, self.out = Path(repo), Path(work), Path(out)
        self.manifest = json.loads(Path(manifest).read_text())
        self.nbs = {n["id"]: n for n in self.manifest["notebooks"]}
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, dict] = {}

    # ---------------------------------------------------------------- catalogue
    def _cells(self, nb_id):
        p = self.repo / self.nbs[nb_id]["file"]
        return json.loads(p.read_text())["cells"]

    def catalogue(self):
        out = []
        for nb in self.manifest["notebooks"]:
            p = self.repo / nb["file"]
            s = self.sessions.get(nb["id"])
            item = dict(id=nb["id"], title=nb["title"], summary=nb.get("summary", ""), file=nb["file"],
                        found=p.is_file(), kernel=nb.get("kernel"),
                        kernel_running=bool(s and s.alive()), kernel_used=s.kernel_name if s else None,
                        kernel_note=s.kernel_note if s else "", busy=bool(s and s.busy), stages=[],
                        overrides=dict(inject_after=nb.get("inject_after", {}), replace=nb.get("replace", {})))
            cells = self._cells(nb["id"]) if p.is_file() else []
            for st in nb["stages"]:
                if st.get("code_file"):
                    item["stages"].append(dict(id=st["id"], title=st["title"], cells=[], code_cells=1, flags=st.get("flags", []),
                                               note=st.get("note", ""), default_selected=st.get("default_selected", True),
                                               status=(s.stage_status.get(st["id"], "not run") if s else "not run")))
                    continue
                code = [i for i in st["cells"] if i < len(cells) and cells[i]["cell_type"] == "code"
                        and "".join(cells[i]["source"]).strip()]
                item["stages"].append(dict(id=st["id"], title=st["title"], cells=st["cells"], code_cells=len(code),
                                           flags=st.get("flags", []), note=st.get("note", ""),
                                           default_selected=st.get("default_selected", True),
                                           status=(s.stage_status.get(st["id"], "not run") if s else "not run")))
            out.append(item)
        return out

    def stage_code(self, nb_id, stage_id):
        cells = self._cells(nb_id)
        st = next(s for s in self.nbs[nb_id]["stages"] if s["id"] == stage_id)
        if st.get("code_file"):
            return [dict(index=0, type="code", source=(Path(__file__).parent / st["code_file"]).read_text())]
        return [dict(index=i, type=cells[i]["cell_type"], source="".join(cells[i]["source"]))
                for i in st["cells"] if i < len(cells)]

    # ---------------------------------------------------------------- run
    def session(self, nb_id) -> Session:
        if nb_id not in self.sessions:
            self.sessions[nb_id] = Session(self.nbs[nb_id], self.repo, self.work)
        return self.sessions[nb_id]

    def _prepare(self, nb, idx, src, params, log):
        import string
        fmt = {k: (v or "") for k, v in params.items()}
        keep = {}   # input paths put in by the app: protected from the output-folder redirect below
        for lit, repl in nb.get("replace", {}).items():
            keys = [f for _, f, _, _ in string.Formatter().parse(repl) if f]
            if any(not fmt.get(k.split("!")[0]) for k in keys):
                continue  # input not given: keep the notebook's own path
            new = repl.format(**fmt)
            if lit in src and new and new != lit:
                tok = f"\x00APP4D{len(keep)}\x00"
                keep[tok] = new
                src = src.replace(lit, tok)
                log.append(f"cell {idx}: '{lit}' → '{new}'")
        ov = nb.get("overlay")
        if ov and params.get("workdir"):
            ovdir = str(Path(params["workdir"]) / ov.get("dir", "team_root"))
            if ov["from"] in src:
                n = src.count(ov["from"])
                src = src.replace(ov["from"], ovdir)
                log.append(f"cell {idx}: {n}× '{ov['from']}' → '{ovdir}' (reads follow links to the team folder, writes stay here)")
            for tok, v in keep.items():
                src = src.replace(tok, v)
            return src  # no blanket redirect for overlay notebooks
        root_from, root_to = params.get("root_from"), params.get("root_to")
        if nb.get("redirect", True) is False:   # this notebook writes only where its replace map sends it
            root_from = None
        if root_from and root_to and root_from != root_to and root_from in src:
            n = src.count(root_from)
            src = src.replace(root_from, root_to)
            log.append(f"cell {idx}: {n}× '{root_from}' → '{root_to}'")
        for tok, v in keep.items():
            src = src.replace(tok, v)
        return src

    def run(self, nb_id: str, stage_ids: list[str], params: dict, include_install=False,
            log_dir: Path | None = None, url_prefix: str | None = None) -> str:
        if nb_id not in self.nbs:
            raise KeyError(nb_id)
        s = self.session(nb_id)
        if s.busy:
            raise RuntimeError("A run is already in progress on this engine. Stop it or wait.")
        jid = time.strftime("nb-%H%M%S-") + uuid.uuid4().hex[:4]
        img_dir = (Path(log_dir) if log_dir else self.out / "nb") / jid
        img_dir.mkdir(parents=True, exist_ok=True)
        img_url = (url_prefix or "/runs/nb") + "/" + jid
        job = dict(id=jid, nb=nb_id, status="queued", stages=stage_ids, current=None, cell=None,
                   events=[], cancel=threading.Event(), started=time.time(), workspace=params.get("root_to"))
        self.jobs[jid] = job
        nb = self.nbs[nb_id]

        def emit(kind, text, **kw):
            job["events"].append(dict(t=round(time.time() - job["started"], 1), kind=kind, text=text,
                                      stage=job["current"], cell=job["cell"], **kw))

        def work():
            s.busy = True
            job["status"] = "running"
            try:
                if nb.get("overlay"):
                    for line in _build_overlay(nb["overlay"], params):
                        emit("note", line)
                if not s.alive():
                    cpu = all("cpu_ok" in next(x for x in nb["stages"] if x["id"] == sid).get("flags", []) for sid in stage_ids)
                    emit("note", "Starting a CPU Jupyter kernel (no GPU needed for these stages)…" if cpu else
                         f"Starting Jupyter kernel '{nb.get('kernel')}'… (a SLURM GPU kernel may wait in the queue)")
                    s.start(cpu=cpu)
                    emit("note", f"Kernel ready: {s.kernel_name}." + (f" {s.kernel_note}" if s.kernel_note else ""))
                cells = self._cells(nb_id)
                for sid in stage_ids:
                    st = next(x for x in nb["stages"] if x["id"] == sid)
                    job["current"] = sid
                    if "install" in st.get("flags", []) and not include_install and set(st["cells"]) <= _install_cells(cells, st):
                        emit("note", f"Skipped '{st['title']}' (install-only stage; tick 'include installs' to run it).")
                        s.stage_status[sid] = "skipped"
                        continue
                    if "gpu" in st.get("flags", []) and s.kernel_name != (nb.get("kernel") or "python3"):
                        s.stage_status[sid] = "error"
                        emit("error_stop", f"Stopped before '{st['title']}': it needs a GPU, but the app is running the "
                             f"'{s.kernel_name}' kernel because '{nb.get('kernel')}' is not visible to the app's Python. "
                             f"Start the app server after 'module load jupyter/16.0.12' (or with ./start_app.sh), then press "
                             f"Restart kernel and run again from 'Load & match'.")
                        job["status"] = "error"
                        return
                    s.stage_status[sid] = "running"
                    emit("stage", st["title"])
                    ok = True
                    if st.get("prep_file"):
                        ok = s.execute((Path(__file__).parent / st["prep_file"]).read_text(), emit, job["cancel"], img_dir, img_url)
                    if st.get("code_file"):   # app-supplied step (not a notebook cell)
                        code = (Path(__file__).parent / st["code_file"]).read_text().replace(
                            "__HZ_DIR__", params.get("horizons_dir") or str(Path(params.get("root_to") or ".") / "Autopick_Horizon"))
                        job["cell"] = None
                        emit("cell", f"Running app step '{st['title']}'")
                        ok = s.execute(code, emit, job["cancel"], img_dir, img_url)
                        s.stage_status[sid] = "done" if ok else "error"
                        if not ok:
                            emit("error_stop", f"Stopped: '{st['title']}' failed.")
                            job["status"] = "error"
                            return
                        continue
                    for idx in (st["cells"] if ok else []):
                        if idx >= len(cells) or cells[idx]["cell_type"] != "code":
                            continue
                        src = "".join(cells[idx]["source"])
                        if not src.strip():
                            continue
                        if not include_install and _is_install(src):
                            emit("note", f"cell {idx}: package install skipped ({src.strip().splitlines()[0][:70]}).")
                            continue
                        job["cell"] = idx
                        log = []
                        src = self._prepare(nb, idx, src, params, log)
                        for line in log:
                            emit("app", "Run-time substitution: " + line)
                        emit("cell", f"Running cell {idx}")
                        ok = s.execute(src, emit, job["cancel"], img_dir, img_url)
                        if ok and str(idx) in nb.get("inject_after", {}):
                            code = nb["inject_after"][str(idx)].format(**{k: (v or "") for k, v in params.items()})
                            emit("app", f"Applying GUI selection after cell {idx}:\n{code}")
                            ok = s.execute(code, emit, job["cancel"], img_dir, img_url)
                        if not ok:
                            break
                    s.stage_status[sid] = "done" if ok else "error"
                    if not ok:
                        emit("error_stop", f"Stopped: '{st['title']}' failed at cell {job['cell']}.")
                        job["status"] = "error"
                        return
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                emit("error", f"{type(e).__name__}: {e}")
                job["status"] = "error"
            finally:
                job["current"] = job["cell"] = None
                job["finished"] = time.time()
                s.busy = False
                try:  # keep a plain-text log of the run next to its figures
                    (img_dir / "run_log.json").write_text(json.dumps(dict(
                        id=jid, notebook=nb["file"], stages=stage_ids, status=job["status"],
                        params={k: v for k, v in params.items()}, events=job["events"]), indent=1))
                except Exception:
                    pass

        threading.Thread(target=work, daemon=True).start()
        return jid

    def job_view(self, jid, since=0):
        j = self.jobs[jid]
        s = self.sessions.get(j["nb"])
        return dict(id=j["id"], nb=j["nb"], status=j["status"], stage=j["current"], cell=j["cell"], stages=j["stages"],
                    stage_status={k: (s.stage_status.get(k, "not run") if s else "not run") for k in j["stages"]},
                    workspace=j.get("workspace"),
                    events=j["events"][since:], next=len(j["events"]),
                    elapsed=round((j.get("finished") or time.time()) - j["started"], 1))

    def stop(self, jid):
        self.jobs[jid]["cancel"].set()

    def shutdown(self):
        for s in self.sessions.values():
            try:
                s.kc.stop_channels()
                s.km.shutdown_kernel(now=True)
            except Exception:
                pass


INSTALL_RE = re.compile(r"^\s*[%!]\s*pip\s+install", re.M)


def _is_install(src: str) -> bool:
    lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    return bool(lines) and all(INSTALL_RE.match(l) for l in lines)


def _install_cells(cells, st):
    return {i for i in st["cells"] if i < len(cells) and cells[i]["cell_type"] == "code" and _is_install("".join(cells[i]["source"]))} | \
           {i for i in st["cells"] if i < len(cells) and (cells[i]["cell_type"] != "code" or not "".join(cells[i]["source"]).strip())}
