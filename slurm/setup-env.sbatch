#!/bin/bash
# Builds the client virtualenv on scratch (login nodes kill pip). Run after syncing the repo:
#   rsync -a ./ explorer:/scratch/$USER/llmbench/repo/ && sbatch slurm/setup-env.sbatch
#SBATCH --job-name=llmbench-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/llmbench/logs/env-%j.out
set -euo pipefail
ROOT=/scratch/$USER/llmbench
module load python/3.13.5
python3 -m venv "$ROOT/env"
"$ROOT/env/bin/pip" install -q --upgrade pip
"$ROOT/env/bin/pip" install -q -e "$ROOT/repo[hf,fast,plot]"
"$ROOT/env/bin/llmbench" --version
"$ROOT/env/bin/python" -c "import uvloop, orjson, tokenizers, matplotlib; print('ok')"
