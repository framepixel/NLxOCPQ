#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval
#SBATCH --output=logs/nlxocpq-eval-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
# 64G, matching the training jobs: parsing an ocpq_cli export costs a measured 11.7x its size in
# RAM, and the largest gold export is 745 MB (~8.5 GB resident).
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
# Required: the personal allocation is exhausted (-101%) while the project holds ~945 GPU-h.
#SBATCH --account=thes2358
# Site default may be --export=NONE; options are passed as arguments regardless, this is belt-and-braces.
#SBATCH --export=ALL

# Single-job form: all 36 conditions x 169 cases in one process. For the real matrix prefer
# slurm/submit_eval_array.sh, which shards it across 72 tasks and supports --resume.

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

# Fail here rather than hours in: LLM.py imports sklearn and openai unguarded at module level.
if [ ! -f "src/check_eval_deps.py" ]; then
  echo "FATAL: ./check_eval_deps.py not found — copy the current NLxOCPQ files to the cluster." >&2
  exit 1
fi
python "src/check_eval_deps.py" --verbose || exit 1

echo "Starting eval_testset at: $(date)"
echo "Python: $(which python)"; python --version

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set — ensure HF provider runs are disabled or token provided in config."
fi

# Offline by design, as in the training jobs. Populate the cache first with:
#     bash scripts/prefetch_models.sh && VERIFY=1 bash scripts/prefetch_models.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [ "${OCPQ_ALLOW_HUB:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
fi

# Node-local scratch for ocpq_cli result exports.
export OCPQ_CLI_SCRATCH_DIR="${OCPQ_CLI_SCRATCH_DIR:-${TMPDIR:-/tmp}/ocpq-$SLURM_JOB_ID}"
mkdir -p "$OCPQ_CLI_SCRATCH_DIR"
trap 'rm -rf "$OCPQ_CLI_SCRATCH_DIR"' EXIT

# The full schema block reaches 10,493 estimated tokens (angular_github_commits_ocel.xml); the
# old 3500 default silently dropped sections for that log and for bpic2017.
export OCPQ_SCHEMA_TOKEN_BUDGET="${OCPQ_SCHEMA_TOKEN_BUDGET:-17000}"

# bash, not zsh: `${PIPESTATUS[0]}` below is a bashism (zsh spells it `$pipestatus`), so under
# the previous zsh shebang it expanded to nothing, rc defaulted to 0, and a failed evaluation
# reported success -- `set -e` does not catch it either, because the pipeline's last command is
# `tee`, which succeeds. `pipefail` now makes that belt-and-braces.
set -eo pipefail
python -u src/eval_testset.py --config configs/eval_config.json 2>&1 | tee logs/eval_testset.full.log
rc=${PIPESTATUS[0]:-0}
echo "eval_testset exit code: $rc"
exit $rc
