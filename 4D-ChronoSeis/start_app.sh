#!/bin/bash
# Start 4D ChronoSeis on the APGCE HPC login node.
# 'module load jupyter' is needed so the app can start the SLURM GPU kernel (slurm-py312-gpu):
# that kernel is launched by Bright's kernel provisioner, which only exists inside the Jupyter module.
module load jupyter/16.0.12 2>/dev/null || echo "WARNING: could not load jupyter/16.0.12; GPU stages will not run."
cd "$(dirname "$0")"
export APP4D_REPO=${APP4D_REPO:-/home/apgce_team_08/Downloads}
export APP4D_ROOTS=${APP4D_ROOTS:-/home/apgce_team_08:/data/epic20}
export APP4D_OUT=${APP4D_OUT:-/home/apgce_team_08/chronoseis_output}
echo "Python: $(which python3)"
jupyter kernelspec list 2>/dev/null | grep -q slurm-py312-gpu && echo "GPU kernel slurm-py312-gpu: found" || echo "WARNING: slurm-py312-gpu not visible"
exec python3 -m uvicorn server:app --port ${PORT:-8050}
