#!/usr/bin/env bash
#SBATCH --job-name=ocpq_diag
#SBATCH --output=logs/ocpq-diag-%J.out
#SBATCH --partition=c23ms
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=thes2358
#SBATCH --export=ALL

# Measure one already-generated query's result export. NO GPU is requested: the query was saved
# to disk before execution, so nothing needs a model. That keeps this out of the GPU queue.
#
# The environment block below is lifted verbatim from slurm/submit_eval_array.sh. It is not optional:
# without the GCC module and the Intel runtime on LD_LIBRARY_PATH, `cargo run` cannot link
# ocpq_cli (rust-lld: undefined symbol _intel_fast_memset, from libmimalloc_sys) and the CLI
# never runs. An interactive shell with only the conda env activated hits exactly that.
#
#   sbatch slurm/submit_diagnose.sh <shard-dir> <case-number>
#   sbatch slurm/submit_diagnose.sh eval_runs/_abandoned/shard_m5_s0_e0_c0of2_20260827_125642 37

set -u
SHARD_DIR="${1:?usage: sbatch slurm/submit_diagnose.sh <shard-dir> <case-number>}"
CASE_NO="${2:?usage: sbatch slurm/submit_diagnose.sh <shard-dir> <case-number>}"

# Resolve the repo root from this script's own location, so the job works whether it was
# submitted from the repo root (`sbatch slurm/<this>.sh`) or from inside `slurm/`, and
# regardless of where $SLURM_SUBMIT_DIR points. Everything below is repo-root-relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

# Same stack as the fine-tuning runs, so the adapters are loaded by what wrote them.
# A conda env, not the old $HOME/envs/nlxocpq-venv virtualenv.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# The eval environment is a clone of the fine-tuning env plus scikit-learn and openai,
# which the eval path needs and training does not. Built by scripts/make_eval_env.sh, which
# verifies no package shared with training changed version. Falls back to the training
# env only if you have installed those two into it yourself.
conda activate "${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

# Node-local scratch: the export may be several GB and is deleted on exit either way.
export OCPQ_CLI_SCRATCH_DIR="${OCPQ_CLI_SCRATCH_DIR:-${TMPDIR:-/tmp}/ocpq-diag-$SLURM_JOB_ID}"
mkdir -p "$OCPQ_CLI_SCRATCH_DIR"
trap 'rm -rf "$OCPQ_CLI_SCRATCH_DIR"' EXIT

echo "Diagnosing ${SHARD_DIR} case ${CASE_NO} at: $(date)"
python -u "$PROJECT_DIR/src/diagnose_result_size.py"     --shard-dir "$SHARD_DIR" --case "$CASE_NO"
echo "diagnose exit code: $?"
