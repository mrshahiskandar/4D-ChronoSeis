# 4D ChronoSeis app: lets the class stage run on its own (it reads the saved RMS/NdRMS files, not the seismic).
# The notebook's class cell draws wells with overlay_well_map() from the 'Well overlay helper' stage; when that stage
# was not run in this session, wells are simply left off the saved PNGs (the app's own maps still show them).
if "overlay_well_map" not in globals():
    def overlay_well_map(*args, **kwargs):
        return None
    print("[app] Well overlay helper not loaded: class PNGs are saved without wells (the app's maps show them).")
