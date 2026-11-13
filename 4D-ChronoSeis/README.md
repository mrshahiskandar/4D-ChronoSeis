# 4D ChronoSeis (GeoHackathon 2026, Challenge 3)

A web GUI on the HPC that runs **4D ChronoSeis**, the team's 4D engine (GitHub `dankkl89/Shell_Geohackathon`), on two time-lapse
SEG-Y volumes, then shows the results: a high-level **Summary** and a **Detail analysis** with maps, statistics and
every figure the notebook made. Gemma (the hosted LLM) explains the results; the Jupyter kernel does the computing.

The notebooks are **never edited**. The GUI passes its inputs in at run time (see *How inputs reach the notebook*).

```
server.py           FastAPI server: path browser (allow-listed folders), input checks, notebook runs, results, Gemma
nbrunner.py         runs notebook stages in Jupyter kernels (one kernel per notebook), captures text + figures
pipeline.json       notebook cells grouped into named stages, plus the GUI-input mappings
results.py          reads the notebook's output files in the working directory → summary, X/Y map rasters, stats
wellio.py           folder scan, seismic domain hint, LAS / checkshot / marker readers
static/index.html   the web UI: ONE self-contained file (fonts, icons, artwork embedded; no external requests)
ui/index.src.html   editable source of the UI      ui/build_ui.py  embeds ui/fonts/*.woff2 and three.js → static/index.html
ui/vendor/          three.js 0.187 subset + OrbitControls (MIT, THREE_LICENSE.txt), bundled with esbuild
make_demo_data.py   synthetic Baseline/Monitor SEG-Y (+ make_demo_wells.py) for testing without the real data
tests/make_fixture_outputs.py   TEST ONLY: writes fake notebook-format outputs so the result tabs can be checked
```

Editing the UI: change `ui/index.src.html`, then `python ui/build_ui.py`. Don't edit `static/index.html` by hand.

## The three tabs

1. **Inputs & run**
   - Seismic: Baseline and Monitor SEG-Y paths, each inspected (traces, inline/crossline range, TWT).
     The domain must be **Time**; depth volumes are refused because the notebook works in TWT.
   - Wells & platforms: the marker CSV (horizon seeds), the flat well folder (`Well-N.las` + checkshot/deviation files)
     and `Platform.csv` (`X(m)`, `Y(m)`). Each has a *Check* button.
   - **Output working directory**: every run writes all intermediate and final products here (horizons, RMS/NdRMS
     tables, maps, 4D class CSVs, run logs in `_app_runs/`). Pick a fresh folder per scenario.
   - **Run full workflow** runs all default stages of the 4D notebook in order and stops at the first error. Single
     stages can be run or their code viewed. The petrophysics notebooks are optional and run separately.
2. **Summary**: confidence, headline, most-changed horizons, possible untapped areas (ranked U1…), data quality,
   next steps, a 4D-class map in real X/Y, and *Ask Gemma*. **Download report** saves an offline HTML report.
3. **Detail analysis**: map explorer (any horizon × attribute: NdRMS, NRMS, RMS Baseline/Monitor, classes,
   sweetness…), per-horizon statistics table, and a gallery of all notebook figures.

### Maps: 2D / 3D, pop-out, full screen
Every map panel (Summary and Detail) has a **2D | 3D** switch (remembered), **Pop out** and **Full screen**.
- **2D**: one horizon, any attribute, real X/Y.
- **3D** (three.js, embedded in the page; no internet needed): all horizons stacked by TWT in real X/Y, coloured by the
  chosen attribute (grey = not classified, i.e. outside the platform radius). Wells as vertical sticks with their
  horizon intersections, platforms with their 3000 m radius ring, U pins for possible untapped areas, frame with X/Y/TWT
  labels and a north arrow. Controls: show/hide horizons, vertical exaggeration (set automatically; TWT drawn at
  ~2,500 m/s, display only), opacity, overlay toggles, reset/top view, save image. Hover shows horizon, X, Y, TWT, value.
- **Well paths** (`wellpath.py`, `/api/team/wellpaths`): same method as the 4D ChronoSeis overlay helper — LAS X/Y as
  wellhead (TVDSS = −KB), deviation survey (MDKB, TVD, INCL, AZIM) by minimum curvature, checkshot (TVDSS, TWT) for TWT
  every 5 m. Works for any number of wells and any names: `<well>.las` + `<well>_Checkshot*` + `<well>_Deviation*`
  (`-`/`_` variants accepted). KB from the LAS header (KB/EKB/EDF…), else the notebook's table for Well-1..8, else 0 m
  with a warning. No deviation → vertical; no checkshot → wellhead only. Checked against the notebook's own code on
  the same files: identical within 5 mm. 2D shows each path's track on the map and where it crosses the shown horizon;
  3D draws the path as a tube with a dot at every horizon crossing.
- **Pop out** opens a separate viewer window (`/?viewer=1…`) that can be moved to another screen and maximised; it
  has its own horizon/attribute selectors, 2D/3D switch and Full screen button.
- 3D needs WebGL in the browser. If it is off, the panel says so; 2D always works.
- Server: `/api/team/scene3d?attr=…` returns all horizons on one shared X/Y grid (≈200 × 200 cells, float32).

### Projects (Save / Open, top bar)
- **Save project** writes `<name>.chronoseis.json` (default: in the output working directory). It stores the seismic,
  well and platform paths, the seismic domain, the output working directory and a snapshot of how many results it held.
  Data and results are **not** copied.
- **Open project** lists recent projects (with *Available* / *Moved or missing*) or browses to a project file (or a
  folder that contains one). Every saved path is checked.
- If the results folder or input files have moved, **Pick a folder to search**: ChronoSeis looks inside that folder
  (up to 4 levels deep) for a folder holding 4D ChronoSeis results and for files/folders with the same names. Found
  items are marked *found here*; tick *Save the new locations* to update the project file. A results folder that
  exists but is empty (while the project had results) counts as missing.
- Results can be viewed even if some inputs are missing; running again needs them.
- The app never creates folders when it opens a project or saves settings; the working directory is created only when
  a run starts.

### Coordinates (prototype assumption)
Maps use the real X/Y (metres) from the notebook's output tables. **The prototype does not ask for a coordinate
system: it assumes the seismic, wells and platforms use the same CRS and units.** This is shown as a reminder in the
GUI. Check it before trusting well/platform positions on the maps.

### Possible untapped areas (screening, not a prediction)
Unswept cells that are bright on the Baseline (≥ P70 and ≥ 1.3 × median), next to changed areas, within the platform
radius. Ranked with High/Medium/Low confidence. The hardening/softening polarity is **uncalibrated**.

## Run on the HPC (login node)

```bash
python3 -m venv ~/venv4d && source ~/venv4d/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

git clone https://github.com/dankkl89/Shell_Geohackathon team_notebooks   # or point to an existing clone
export APP4D_REPO=$PWD/team_notebooks
export APP4D_ROOTS=/data/epic20/APGCE/input_data:/home/apgce_team_08      # folders the GUI may read/write
export APP4D_OUT=/home/apgce_team_08/app4d                               # settings + default working directory
jupyter kernelspec list        # should list python3 and slurm-py312-gpu
python server.py               # → http://localhost:8050 in Firefox on the remote desktop
```

- Port 8000 is JupyterHub, so the app uses 8050 (`APP4D_PORT`). It listens on 127.0.0.1 only.
- Gemma: `http://172.28.34.1:8000/v1` (override with `GEMMA_URL`).
- The 4D notebook asks for the **`slurm-py312-gpu`** kernel. If the server can't see it, it falls back to `python3`
  and says so; GPU stages (preconditioning, refinement, sweetness) then fail. Add JupyterHub's kernel folder to
  `JUPYTER_PATH` if needed.

### How inputs reach the notebook (from `pipeline.json`, applied at run time and printed in the console)
- After cell 2: `baseline_path` / `monitor_path` = the selected surveys.
- Literal swaps: `/home/apgce_team_08/Baseline_Matched.sgy` → Baseline; the marker CSV, `Platform.csv` and
  `To_Overlay_Horizon/` → the selected files/folder.
- The `/home/apgce_team_08` prefix → the output working directory, so all outputs land there.
- After the class cell: the notebook's own `_export(None)` is called so the class CSVs/maps are written.
- `%pip install` cells are skipped unless *Include package installs* is ticked. Interactive widget viewers don't render
  in the GUI; their figures appear only when the notebook saves them.

### Check with the notebook author before the real run
- The notebook was built on `Baseline_Matched.sgy` (prepared outside the notebook). Using the raw Challenge 3
  Baseline may give different results.
- The marker CSV, `Platform.csv` and the flat well folder are not in the repo; they must exist on the HPC.
- Both full volumes are loaded into memory (~6.5 GB each, several copies). Use a kernel with enough memory.

## Try it without the HPC

```bash
python make_demo_data.py demo_data
APP4D_REPO=../team_notebooks APP4D_ROOTS=$PWD python server.py
```
Without CuPy/GPU the run stops at the first GPU stage (expected). To check the Summary and Detail tabs, generate
**fake** outputs: `python tests/make_fixture_outputs.py demo_data ws_fixture`, then pick `ws_fixture` as the working
directory. Fixture results are synthetic and must never be presented as findings.

## Safety
- The server reads and writes only inside `APP4D_ROOTS`. It runs only the team notebook cells listed in
  `pipeline.json` (plus the logged input assignments), never arbitrary commands from the page.
- Don't copy raw data off the HPC. Check with the organisers before exporting reports or figures.

## Protecting existing results

- The output folder defaults to `<team folder>/chronoseis_output` (e.g. `/home/apgce_team_08/chronoseis_output`).
- A run is refused if the output folder is the team folder itself (where the notebook's earlier results live).
- A run is refused (with an "overwrite?" confirmation in the GUI) if the folder holds notebook outputs
  (`Autopick_Horizon`, `4D_RMS_NRMS_3`, `Maps`, …) that this app did not create.
- Folders created by the app carry a `.chronoseis_workspace` marker, so re-running stages there is allowed.
- The notebooks themselves are never modified; input data is only read.

## Data QC viewer (time slices with wells)

- **Inputs & run** tab → "Data QC viewer", and **Detail analysis** → Layer: Time slice.
- Shows one time slice of the Baseline, the Monitor, or Monitor − Baseline in real X/Y, 2D or 3D (a flat plane
  inside a frame spanning the volume's time range, with wells and platforms in the Detail tab).
- Reads ONE sample per trace straight from the SEG-Y (strided read, no full volume in memory; IBM float, IEEE float,
  int32/16/8, fixed-length traces). Trace headers (X, Y, inline, crossline) are cached in the app's state folder
  (`APP4D_OUT/_cache`). The first slice on a ~6.7 GB volume takes about a minute or two; later slices reuse the headers.
- Nothing is written unless you press **Save QC slice**: that writes `QC_Slices/<volume>_<time>ms.csv` (X, Y, value)
  and a PNG into the output working directory.
- Monitor − Baseline is computed on the common X/Y bins only (both volumes are averaged onto the same grid first).
- Code: `slices.py`, endpoints `/api/qc/info`, `/api/qc/slice`, `/api/qc/slice3d`, `/api/qc/slice/save`.

## Wells in the seismic's domain (time)

All wells are drawn in two-way time (TWT), so they sit correctly on the seismic in 2D and 3D.

- **Well folder:** LAS, deviation surveys, checkshots and marker files are found anywhere under the folder (flat or in `las/`, `checkshots/`, `deviations/`, `markers/` sub-folders). An optional **well list** (Excel or CSV: Well Name, X, Y, RT, WD, TD) is found automatically in the folder, or chosen separately.
- **Horizontal:** wellhead X/Y (LAS header, else the well list), then the deviation survey by minimum curvature.
- **Vertical:** KB from the well list (RT), then LAS EREF, then LAS EKB; TVDSS = TVD − KB; checkshot TVDSS → TWT. Above the first checkshot the time-depth is anchored at MSL and the seabed (1500 m/s water); below the last checkshot it continues at the last interval velocity. Extrapolated parts are flagged. Markers go MD → TVD → TVDSS → TWT and appear in 3D.
- **Well tie check** (Inputs page, below the well list): per well, where X/Y and KB come from, disagreements between sources, bottom-hole offset versus the LAS `XOFFSET/YOFFSET_BOTTOM`, survey consistency, checkshot monotonicity, and (optionally) whether the well lies inside the Baseline X/Y footprint and time range.
- Wells without a checkshot are drawn at the wellhead only, unless "Estimate time … from the nearest well" is ticked (approximate, flagged).
- **Detected files (editable):** every LAS, table and text file under the well folder is classified (LAS log, deviation survey, checkshot, markers, well list, other) from its name, its `#` header comments and its columns, and assigned to a well from its name, header comment or LAS header. The table shows how sure the guess is. Change the type or the well in the dropdowns and the tie check re-runs; "Reset to detected" undoes your changes. Corrections are saved per well folder in the app's settings. If two files of one type are assigned to one well, only one is used and the other is marked.

## 4D time shift (Stage A)

In the Data QC viewer and in Detail → Layer: Time slice, choose **4D time shift** or **Time-shift confidence**. The app pairs Baseline and
Monitor traces (by inline/crossline, else nearest X/Y), cross-correlates a Hann-weighted window (start–end ms) over ±lag ms and reports the
shift with sub-sample precision. **Positive = the Monitor event arrives later.** Cells below the minimum correlation are left blank.
Optionally give a window with no expected change (e.g. above the reservoir) to measure the noise floor: the status line then reports
its robust σ and the share of the map that shifts by more than 2σ. Reads only the traces' window, writes nothing except when you press Save.

## ChronoSeis Advisor (Gemma 4)

The Gemma tab on the right edge (or the bar at the bottom, or the `/` key) opens the advisor. Ask in your own words:
"load the baseline and monitor for this hackathon", "load all the wells", "open our working project", "where could oil
have been missed?", "show a section from Well-5 to Well-3".

- `advisor.py` runs the loop: Gemma answers with one JSON object per turn, either an action
  (`find_segy, set_volumes, find_wells, load_wells, list_projects, open_project, get_results, show_view, show_slice,
  show_section, propose_run`) or a final reply. Every reply is validated; a malformed one gets one correction request.
  Max 6 actions per question. Native tool calling is not needed.
- Actions run in the app (`server.py`, section "ChronoSeis Advisor") inside the allowed folders only. They read files
  in place; settings and outputs go to `APP4D_OUT`. Engine runs are only offered (`propose_run` shows a button).
- Defaults: the challenge folder is `APP4D_CHALLENGE` (default `/data/epic20/APGCE/input_data/challenge3`, with
  `volumes/` and `well_data/`). Saved projects are found in the recent list and under `APP4D_ROOT_FROM` (3 levels).
- If Gemma is unreachable, the app's built-in commands still load the seismic, load the wells and open a project
  (clearly labelled). Status light: `/api/advisor/status` (GET `<GEMMA_URL base>/models`, 4 s timeout).

## Section A–A′ with wells and logs

Inputs view → "Section A–A′ with wells and logs" (or ask the advisor). Give two well names or `X,Y` points. The app
reads only the traces on the line (nearest Baseline trace every bin; Monitor paired by inline/crossline), draws
Monitor − Baseline / Baseline / Monitor in TWT with horizons (from the results folder), well paths and markers for wells
within ~250 m of the line, and a log panel per well: the 4D trace at the well plus GR, Sw / resistivity / porosity
when the LAS has them, converted MD → TWT with the same path as the maps. Checkshots written with negative depths
(elevation sign) are now accepted.

### Speed

- **Fast path:** "load the seismic / the wells / open the project" are recognised by the app, which runs the actions
  itself and asks Gemma once to write the answer (instead of one Gemma call per step). Ambiguous cases (several
  projects, unclear Baseline/Monitor) still go to Gemma, which asks.
- **SEG-Y header cache:** the first inspection of a volume scans every trace header (about a minute for a 6-7 GB file
  on shared disk); the result is cached in `<APP4D_OUT>/_cache/inspect.json` (keyed by path, size, modification time),
  so later loads take seconds. Load the data once before a demo.
- The tray shows what is running ("Reading the Baseline headers…", "Gemma is writing the summary…") and the seconds.

### Wells without a checkshot: integrated sonic

If a well has no checkshot file but its LAS has a compressional sonic (DT, DTC, DTCO, AC, DTCOM ... including suffixed
names such as `DTC_P`; µs/ft or µs/m), the app integrates it (light 5-sample median despike) to build the TVDSS -> TWT
relation. The TWT at the top of the log is taken from the nearest well's checkshot at that depth; if no well has a
checkshot, water 1500 m/s to the seabed and 1800 m/s sediment are assumed above the log. Flagged in the well tie check
("time from integrated sonic ... not drift-corrected"). In Challenge 3 this places Well-3 (DTC_P); its times run
~20-40 ms earlier than neighbouring checkshot wells at the same depth, the usual sonic drift.
Log curves with suffixes (`GR_P`, `RT_P`) are also recognised in the section log panel.

### New folder

The folder picker has a **New folder** button when choosing a folder (e.g. the output working directory): type a
name, press Create, then "Use this folder". Names: letters, numbers, spaces, dots, dashes, underscores (max 80).
Folders cannot be created inside the read-only input data (`APP4D_READONLY`, default `/data/epic20`).

## Starting on the APGCE HPC (GPU kernel)

The SLURM GPU kernel `slurm-py312-gpu` (partition petronas_apgce, 1× H200 slice, 8 CPU, 128 GB) is defined in
`/cm/shared/apps/jupyter/current/share/jupyter/kernels/` and launched by Bright's kernel provisioner, which is only
available after `module load jupyter/16.0.12`. Without it the app falls back to `python3` on the login node and GPU
stages stop. Start the app with `./start_app.sh` (does the module load and the usual APP4D_* settings), or run
`module load jupyter/16.0.12` before the usual uvicorn command.

## Phase 4 / Phase 5 well results (Summary, Detail, Advisor)
Run the optional **Wells: Phase 4** and **Wells: Phase 5** notebooks (outputs go to `<output folder>/team_root`, the team's
inputs are read through links). Summary then shows "Wells: oil, gas and water intervals (TWT)" and each untapped area lists
its nearest well's classes; Detail's A-A' section gets a **Fluid** track (blue water, green oil, red gas) beside the logs;
the Advisor sees them as `well_fluids`. If this project has no run of its own, the team's existing Phase 5 results in
`/home/apgce_team_08/link-to-apgce_team_08` (override with `APP4D_TEAM_ROOT`) are read, read-only, and labelled as such.

## Remaining Fluid Index (Phase 6 Part 3D)
The team's `Phase6_Part3D_...RMS_RFI` notebook writes one `<horizon>_RFI.csv` per horizon (X, Y, TWT, RFI_Final, closure flag,
oil-proximity penalty). The app reads them (own run in `<output folder>/phase6_3D/RFI`, else the team's
`/data/epic20/APGCE/phase6_3D/RFI`, read only; override with `APP4D_RFI_DIR`) and shows: a Summary card with ranked RFI hotspots
(connected areas with mean RFI >= 40: area, TWT, closure share, nearest well and platform), per-horizon statistics, and
**Remaining Fluid Index** / **Oil-proximity 4D penalty** as map attributes in Detail. Gemma sees the same numbers as `rfi`.
The RFI is a relative 0-100 screening score scored per horizon, not a saturation or reserve estimate. The Phase 6 notebooks
themselves are not run from the app yet.

## Off-horizon scan (Advisor: "anything we missed above or below the horizon?")
`offsets.py` samples the Monitor - Baseline difference every N ms (default 4) up to +-M ms (default 150) above and below a
picked horizon, straight from the two SEG-Y volumes (NRMS in short windows, up to 6000 horizon traces). The engine's horizon
attributes only use +-10 ms around the pick. The scan reports the profile of median NRMS against offset, offset bands that
stand out from the background, and localized clusters away from the horizon (offset range, TWT, area, how much is already
classed hardening/softening, nearest well and platform). It is a screening aid: an off-horizon response can also be noise,
multiples or processing differences, so check NRMS and a section before interpreting it.

## Use existing horizons (skip horizon generation)

If the horizons already exist (for example `/data/epic20/APGCE/phase6_horizons` or the team's `Autopick_Horizon`), put that folder in
step 2, **Existing horizons**. The files are tab-separated `.txt` with columns `X, Y, Inline, Crossline, TWT` (optionally with
`Autopick_Export_Summary.csv`: `Horizon, File`). "Run full workflow" then skips seeds, picking, refinement and export, and a small app
step ("Use existing horizons") builds `processed_horizons` and `horizon_order` from the files so the notebook's RMS / NdRMS, sweetness
and class stages run unchanged. Files are linked (not copied) into `<results>/Autopick_Horizon` so Summary and Detail see them.
The well markers CSV is not needed in this mode.

## Platform file

`Platforms.csv` (example in `examples/`): columns `Platform, X(m), Y(m)`, plain numbers, same coordinate system as the seismic.

## Reference papers for the Advisor

`knowledge/*.jsonl` holds page-numbered passages from the team's reference papers. For each question the app finds the best
matching passages (offline keyword ranking, no internet, no extra packages) and gives them to Gemma, which cites them as
`[source, p. N]`; Gemma can also call `search_papers`. Add PDFs on the HPC with `python knowledge/build_index.py <pdf folder>`
and restart. The paper text is not part of the GitHub upload (see `.gitignore`).
