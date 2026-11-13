# 4D ChronoSeis app: reuse the team's saved Part 3D horizon attributes (RMS / NdRMS / NRMS / sweetness per pick)
# instead of re-reading every SEG-Y trace for each horizon (Cell 3, hours). Builds `attr_files` exactly as Cell 3 does.
import re
attr_files, _missing = {}, []
for _name in horizons:
    _p = ATTR_OUT / (re.sub(r'[^A-Za-z0-9_.-]+', '_', _name) + '_3D_Attributes.csv')
    if _p.is_file():
        attr_files[_name] = _p
    else:
        _missing.append(_name)
if _missing:
    print('[app] No saved attributes for:', ', '.join(_missing), '- these horizons are skipped (run Cell 3 in JupyterLab to make them).')
    horizons = {k: v for k, v in horizons.items() if k in attr_files}
if not attr_files:
    raise FileNotFoundError(f'No saved *_3D_Attributes.csv in {ATTR_OUT}')
print(f'[app] Reusing {len(attr_files)} saved horizon attribute files from {ATTR_OUT}')
