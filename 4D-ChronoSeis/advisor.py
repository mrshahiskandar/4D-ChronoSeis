"""ChronoSeis Advisor: Gemma 4 drives the app through a fixed list of actions, answered in strict JSON.

Gemma never touches files. Each turn it replies with ONE JSON object:
    {"action": "<name>", "args": {...}}     -> the app runs that action and returns the result to Gemma
    {"reply": "<text>", "next": ["..", ..]}  -> final answer shown to the user (next = suggested follow-ups)
The app validates every reply against ACTIONS. Malformed replies get one correction request; after that the
text is shown as plain prose (or a clear message). If Gemma is unreachable a small keyword router runs the
same actions so the demo still works, clearly labelled as the app's built-in commands.
"""
from __future__ import annotations

import json
import re
import time

MAX_STEPS = 6

# name -> (args spec {arg: (type, required)}, description for Gemma, needs_confirm)
ACTIONS = {
    "find_segy": ({"folder": (str, False)},
                  "List SEG-Y volumes in a folder (default: the challenge 'volumes' folder) with sizes and a Baseline/Monitor guess.", False),
    "set_volumes": ({"baseline": (str, True), "monitor": (str, True)},
                    "Set the Baseline and Monitor SEG-Y in the app and read their headers (traces, time range, sampling, IL/XL, size) and compare them.", False),
    "find_wells": ({"folder": (str, False)},
                   "Look in a well folder (default: the challenge 'well_data' folder): file types found, well list file, wells named.", False),
    "load_wells": ({"folder": (str, True), "well_list": (str, False)},
                   "Load the wells into the app: detect files, assign them to wells, convert to TWT and run the well tie check.", False),
    "list_projects": ({}, "List saved 4D ChronoSeis projects (name, path, saved date).", False),
    "open_project": ({"path": (str, True)},
                     "Open a saved project: sets all inputs and the results folder and shows the Summary. Results were computed earlier.", False),
    "get_results": ({}, "Get the computed 4D results (per-horizon statistics, classes, untapped areas, wells, data quality).", False),
    "show_view": ({"view": (str, True)}, "Switch the page: 'inputs' (setup and runs), 'data' (Data viewer: time slices and A-A' sections), 'summary' or 'detail'.", False),
    "show_slice": ({"vol": (str, True), "t": (float, True), "t1": (float, False)},
                   "Show a time slice in the Data QC viewer. vol: 'diff' (Monitor - Baseline), 'baseline', 'monitor', 'tshift' (time shift, needs t1 = window end), 'tcc'.", False),
    "show_section": ({"a": (str, True), "b": (str, True), "vol": (str, False)},
                     "Show a vertical section A-A' between two wells (names, e.g. 'Well-5') or 'X,Y' points; vol 'diff' (default), 'baseline' or 'monitor'. Wells, horizons and logs are drawn on it.", False),
    "reuse_team_results": ({}, "Copy the team's earlier engine results (horizons, RMS/NdRMS, classes) into the results folder so Summary and Detail work at once, without a GPU run. Read-only on the team folder, never overwrites; needs the results folder to be set.", False),
    "scan_offsets": ({"horizon": (str, False), "step_ms": (float, False), "max_ms": (float, False)},
                     "Scan the 4D response (Monitor - Baseline, NRMS) at fixed steps (default every 4 ms) up to +-max_ms (default 150) above and below a horizon, straight from the two volumes, and flag anomalies away from the horizon that the horizon maps (+-10 ms) cannot show. Use for 'anything we missed above or below the horizon?' questions. Takes seconds to a minute.", False),
    "find_inputs": ({"folder": (str, False)},
                    "Search for the platform coordinate CSV (X(m), Y(m)) and for folders of existing horizon .txt files (default: the app's inputs folder, HTML_TEST, the challenge folder, phase6_horizons).", False),
    "set_inputs": ({"platform_csv": (str, False), "horizons_dir": (str, False)},
                   "Put the platform CSV and/or the existing-horizons folder into the app's input boxes (after find_inputs, or a path the user gave). You cannot upload files; the user places them on the HPC.", False),
    "search_papers": ({"query": (str, True)},
                      "Search the user's reference papers (4D seismic, ML, geoscience LLMs, hackathon handbook/FAQ) for background; returns passages with source and page.", False),
    "propose_run": ({"what": (str, True)},
                    "Offer to run the ChronoSeis engine (e.g. 'horizons from markers', 'full workflow'). Never runs by itself: the user confirms with a button.", True),
}

FACTS = """Facts about the workflow (use them when advising):
- The state lists the input files in use: platform_file (with platforms: names and X/Y), horizon_folder (with horizon_files: existing horizon names; when set, horizon picking is skipped) and markers_csv. Quote these paths when asked where an input is. If platform_file or horizon_folder is empty, call find_inputs, then set_inputs; never offer to upload (files are placed on the HPC by the user).
- Order: 1 seismic volumes -> 2 wells -> 3 horizons -> 4 engine run (RMS, NdRMS, classes) -> 5 interpretation.
- Without wells these still work: time-slice QC, Monitor - Baseline difference, 4D time-shift map.
- Without wells these are missing: horizons (the engine builds them from well markers), so RMS/NdRMS along horizons,
  hardening/softening/unswept classes and untapped-area screening cannot run; also no well paths on maps and no well tie check.
- If no horizon interpretation is given, the ChronoSeis engine generates horizons from the well markers
  (seed at markers -> NCC auto-pick -> cycle-skip refinement -> smoothing). This is a long GPU run: only propose it.
- Hardening/softening polarity is not calibrated. Fluid causes (water sweep, gas, pressure) are hypotheses: no production
  or injection data and no simulation model are available. All challenge wells are exploration wells (no injectors).
- Well results (Phase 4/5): per-well oil/gas/water intervals in TWT (ML classes from logs, not tested fluids), the nearest
  correlated marker for each, and proposed markers; they appear in get_results as well_fluids and in the Summary.
- To skip the long GPU run, the team's earlier results can be brought in with reuse_team_results (copies, no overwrite).
- Remaining Fluid Index (RFI, Phase 6 Part 3D): a relative screening score 0-100 per horizon from 4D response, sweetness and RMS, lowered near
  oil wells with strong 4D change and capped outside structural highs; hotspots are in get_results as rfi. Not a saturation or reserve estimate.
- Off-horizon check: scan_offsets samples the Baseline/Monitor difference every few ms above and below a horizon (the engine only looks within +-10 ms of the pick). It can reveal 4D response the horizon maps miss; always state that it may also be noise, multiples or processing, and point to NRMS and the section view for checking. It cannot say what fluid causes it.
- Reference papers: background passages may be given as PAPERS. Use them to explain methods, physics and context and cite as
  [short source, p. N]. They describe other fields and studies: never present their numbers as results for this field, and do not cite
  a paper that was not given to you. If no passage fits, say the papers don't cover it and answer from general knowledge, marked as such.
- Saved projects hold results computed earlier; say so and give the saved date when opening one."""


def system_prompt(state: dict) -> str:
    acts = "\n".join(f'- {n}({", ".join(a + ("" if r else "?") for a, (_, r) in spec.items())}): {d}'
                     for n, (spec, d, _) in ACTIONS.items())
    return f"""You are the ChronoSeis Advisor inside the 4D ChronoSeis web app (4D time-lapse seismic, GeoHackathon 2026 Challenge 3),
running as Gemma 4 on the HPC. You help the user load data, check it, and interpret the computed 4D results.

You can only act through these actions:
{acts}

Reply with exactly ONE JSON object and nothing else, in one of two forms:
{{"action": "<name>", "args": {{...}}}}
{{"reply": "<short answer for the user>", "next": ["<suggested follow-up>", ...]}}

Rules:
- Use actions to get facts; never invent file names, paths, numbers, wells or results. Quote numbers only from action results.
- Paths must come from action results or the state below. If two choices are equally likely, ask in a reply instead of guessing.
- Do quick actions straight away. For anything that runs the engine use propose_run; never claim it ran.
- Keep replies short (2-6 sentences), plain language, for geoscientists. End with the most useful next step.
- "next": 1-3 short follow-ups the user could click (e.g. "Load the wells").

{FACTS}

Current app state (JSON):
{json.dumps(state, default=str)}"""


# ------------------------------------------------------------------ parsing / validation
def parse(text: str):
    """First JSON object in the text (handles ```json fences and leading prose)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth, ins, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if ins:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    ins = False
                continue
            if c == '"':
                ins = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(t[start:i + 1])
                        if isinstance(o, dict):
                            return o
                    except json.JSONDecodeError:
                        break
                    break
        start = t.find("{", start + 1)
    return None


def validate(o: dict):
    """-> (kind, payload, error). kind 'reply' or 'action'."""
    if "reply" in o:
        nxt = o.get("next") or []
        if not isinstance(nxt, list):
            nxt = [str(nxt)]
        return "reply", dict(text=str(o["reply"]).strip(), next=[str(x)[:80] for x in nxt[:3]]), None
    name = o.get("action")
    if name not in ACTIONS:
        return None, None, f"Unknown action {name!r}. Allowed: {', '.join(ACTIONS)}."
    spec = ACTIONS[name][0]
    args = o.get("args") or {}
    if not isinstance(args, dict):
        return None, None, "args must be an object."
    clean = {}
    for k, (typ, req) in spec.items():
        if k not in args or args[k] in (None, ""):
            if req:
                return None, None, f"Action {name} needs argument {k!r}."
            continue
        try:
            clean[k] = typ(args[k])
        except (TypeError, ValueError):
            return None, None, f"Argument {k!r} of {name} must be {typ.__name__}."
    return "action", dict(name=name, args=clean), None


# ------------------------------------------------------------------ the loop
def run(message: str, history: list[dict], state: dict, handlers: dict, ask, note=lambda s: None) -> dict:
    """ask(messages) -> text (raises if Gemma unreachable). handlers[name](**args) -> dict(result, card?, ui?, buttons?).
    Common load requests take the fast path: the app runs the actions, then ONE Gemma call writes the answer."""
    intent = fast_intent(message)
    if intent:
        r = fast(intent, message, handlers, ask, note)
        if r is not None:
            return r
    msgs = [dict(role="system", content=system_prompt(state))]
    steps, cards, ui, buttons = [], [], [], []
    try:   # background from the user's papers, when the question matches them well
        import knowledge
        hits = knowledge.search(message, 3, min_score=7.0)
    except Exception:
        hits = []
    if hits:
        msgs.append(dict(role="system", content="PAPERS (background for this question; cite as [source, p. N] if you use them):\n" + "\n\n".join(
            f"[{h['source']}, p. {h['page']}] {h['text']}" for h in hits)))
        steps.append(dict(action="search_papers", args=dict(query=message[:60]), ok=True))
    for h in history[-8:]:
        if h.get("role") in ("user", "assistant") and h.get("text"):
            msgs.append(dict(role=h["role"], content=str(h["text"])[:2000]))
    msgs.append(dict(role="user", content=message[:2000]))
    retried = False
    t0 = time.time()
    for _ in range(MAX_STEPS + 2):
        note("Gemma is choosing the next step…" if steps else "Gemma is reading your request…")
        try:
            raw = ask(msgs)
        except Exception as e:
            if steps:   # partial work done: report it
                return _out("app", "Gemma stopped answering mid-way (" + type(e).__name__ + "). The steps above were done.", steps, cards, ui, buttons, [], t0)
            return fallback(message, handlers, reason=type(e).__name__, t0=t0)
        o = parse(raw)
        kind, payload, err = validate(o) if o is not None else (None, None, "Reply was not a JSON object.")
        if kind is None:
            if not retried:
                retried = True
                msgs += [dict(role="assistant", content=raw[:3000]),
                         dict(role="user", content=f"FORMAT ERROR: {err} Reply again with exactly one JSON object as specified.")]
                continue
            text = raw.strip() if raw and o is None and len(raw.strip()) > 0 else "Gemma's answer did not follow the expected format."
            return _out("gemma", text, steps, cards, ui, buttons, [], t0, note="free text (format not followed)")
        if kind == "reply":
            return _out("gemma", payload["text"], steps, cards, ui, buttons, payload["next"], t0)
        name, args = payload["name"], payload["args"]
        if len(steps) >= MAX_STEPS:
            msgs += [dict(role="assistant", content=raw[:3000]),
                     dict(role="user", content="STEP LIMIT reached. Reply now with {\"reply\": ...} summarising what was done.")]
            continue
        step = dict(action=name, args=args)
        note(f"Running {name}…")
        try:
            res = handlers[name](**args)
            step["ok"] = True
        except Exception as e:  # report to Gemma, let it recover
            res = dict(result=dict(error=getattr(e, "detail", None) or str(e)))
            step["ok"] = False
            step["error"] = res["result"]["error"]
        steps.append(step)
        if res.get("card"):
            cards.append(res["card"])
        ui += res.get("ui", [])
        buttons += res.get("buttons", [])
        msgs += [dict(role="assistant", content=json.dumps(dict(action=name, args=args))),
                 dict(role="user", content=f"RESULT of {name}: " + json.dumps(res.get("result"), default=str)[:6000]
                      + "\nContinue: another action, or the final reply.")]
    return _out("gemma", "I could not finish this request in the allowed number of steps.", steps, cards, ui, buttons, [], t0)


def _out(source, text, steps, cards, ui, buttons, nxt, t0, note=None):
    return dict(source=source, text=text, steps=steps, cards=cards, ui=ui, buttons=buttons, next=nxt,
                seconds=round(time.time() - t0, 1), note=note)


# ------------------------------------------------------------------ fast path + built-in commands
_VERB = re.compile(r"\b(load|open|put|bring|set|use|import|add|read|get|pull)\b")


def fast_intent(message: str):
    m = message.lower()
    if len(m) > 220 or not _VERB.search(m):
        return None
    if any(w in m for w in ("section", "slice", "why", "where", "how", "compare", "explain", "?")):
        return None
    if "project" in m:
        return "project"
    if "well" in m:
        return "wells"
    if any(w in m for w in ("seismic", "segy", "seg-y", "volume", "baseline", "monitor", "survey")):
        return "seismic"
    return None


def _execute(intent: str, message: str, handlers: dict, note=lambda s: None):
    """Run the actions for a load intent. -> dict(steps, cards, ui, buttons, results{}, lines[]) or None if unclear."""
    steps, cards, ui, buttons, results, lines = [], [], [], [], {}, []

    def do(name, **a):
        note(f"Running {name}…")
        try:
            r = handlers[name](**a)
            steps.append(dict(action=name, args=a, ok=True))
        except Exception as e:
            steps.append(dict(action=name, args=a, ok=False, error=getattr(e, "detail", None) or str(e)))
            results[name] = dict(error=steps[-1]["error"])
            return None
        if r.get("card"):
            cards.append(r["card"])
        ui.extend(r.get("ui", []))
        buttons.extend(r.get("buttons", []))
        results[name] = r.get("result")
        return r.get("result")

    m = message.lower()
    if intent == "project":
        projs = (do("list_projects") or {}).get("projects", [])
        pick = next((p for p in projs if p["name"].lower() in m), None) or (projs[0] if len(projs) == 1 else None)
        if not pick:
            if len(projs) > 1:
                return None          # let Gemma ask which one
            lines.append("No saved projects found.")
        else:
            r2 = do("open_project", path=pick["path"])
            if r2:
                lines.append(f"Opened project {pick['name']} (saved {r2.get('saved_at', '?')}). These results were computed earlier.")
    elif intent == "seismic":
        g = (do("find_segy") or {}).get("guess") or {}
        if not (g.get("baseline") and g.get("monitor")):
            return None              # ambiguous: Gemma asks
        r2 = do("set_volumes", baseline=g["baseline"], monitor=g["monitor"])
        if r2:
            lines.append("Loaded the Baseline and Monitor volumes. " + " ".join(r2.get("findings", [])))
    elif intent == "wells":
        r = do("find_wells")
        if r and r.get("folder"):
            r2 = do("load_wells", folder=r["folder"], **({"well_list": r["well_list"]} if r.get("well_list") else {}))
            if r2:
                lines.append(r2.get("headline", "Wells loaded."))
        else:
            lines.append("No well folder found.")
    return dict(steps=steps, cards=cards, ui=ui, buttons=buttons, results=results, lines=lines)


DEFAULT_NEXT = {"seismic": ["Load all the wells", "What do I lose if I skip the wells?"],
                "wells": ["Open our working project", "Show a section between two wells"],
                "project": ["Where could oil have been missed?", "Which horizon changed most?"]}


def fast(intent, message, handlers, ask, note=lambda s: None):
    t0 = time.time()
    ex = _execute(intent, message, handlers, note)
    if ex is None:
        return None
    note("Gemma is writing the summary…")
    sysmsg = ("You are the ChronoSeis Advisor (Gemma 4 on the HPC) in the 4D ChronoSeis app. The app has just run the actions "
              "below for the user. Write a short answer (2-5 sentences) for geoscientists: what was done, the key facts and any "
              "warnings, then the most useful next step. Use only numbers in the results. Reply with exactly one JSON object: "
              '{"reply": "...", "next": ["short follow-up", ...]}\n\n' + FACTS)
    user = (f"User asked: {message}\nActions run: " + json.dumps([dict(action=s['action'], ok=s['ok']) for s in ex['steps']])
            + "\nResults: " + json.dumps(ex["results"], default=str)[:7000])
    try:
        raw = ask([dict(role="system", content=sysmsg), dict(role="user", content=user)])
        o = parse(raw)
        kind, payload, _ = validate(o) if o else (None, None, None)
        if kind == "reply" and payload["text"]:
            return _out("gemma", payload["text"], ex["steps"], ex["cards"], ex["ui"], ex["buttons"], payload["next"] or DEFAULT_NEXT[intent], t0,
                        note="fast path: app ran the actions, Gemma wrote the answer")
        text = raw.strip() if raw and raw.strip() and not o else " ".join(ex["lines"])
        return _out("gemma", text, ex["steps"], ex["cards"], ex["ui"], ex["buttons"], DEFAULT_NEXT[intent], t0, note="fast path")
    except Exception as e:
        return _out("app", " ".join(ex["lines"]) + f"\n\n(Gemma is not reachable ({type(e).__name__}); summary written by the app.)",
                    ex["steps"], ex["cards"], ex["ui"], ex["buttons"], DEFAULT_NEXT[intent], t0)


def fallback(message: str, handlers: dict, reason: str = "", t0=None) -> dict:
    """Gemma unreachable: run the load intents with the app's built-in commands."""
    t0 = t0 or time.time()
    intent = fast_intent(message) or ("project" if "project" in message.lower() else "wells" if "well" in message.lower()
                                      else "seismic" if any(w in message.lower() for w in ("seismic", "segy", "volume", "baseline", "monitor")) else None)
    ex = _execute(intent, message, handlers) if intent else None
    if ex is None:
        text = ("I can load the seismic, load the wells or open a saved project without Gemma. Questions about the results need Gemma."
                if not intent else "That needs a choice (several files or projects match); pick it in the page, or try again when Gemma is back.")
        ex = dict(steps=[], cards=[], ui=[], buttons=[], lines=[text])
    text = " ".join(ex["lines"]) + f"\n\n(Gemma is not reachable{' (' + reason + ')' if reason else ''}; handled by the app's built-in commands.)"
    return _out("app", text, ex["steps"], ex["cards"], ex["ui"], ex["buttons"], ["Load the seismic", "Load the wells", "Open the working project"], t0)
