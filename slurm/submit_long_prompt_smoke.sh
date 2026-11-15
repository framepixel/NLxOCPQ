#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_longprompt
#SBATCH --output=logs/nlxocpq-longprompt-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --account=thes2358
#SBATCH --export=ALL
# LONG-PROMPT MEMORY SMOKE (notes 44): one task per model of eval_config.unleashed_gate.json. Loads the
# model as a run does and reports peak GPU memory for prompts of 16k-64k tokens (src/long_prompt_smoke.py).
# Run before evaluation 2's gate; Gemma 4 (task 3) is the one that ran out of memory (notes 43.19).
#     sbatch slurm/submit_long_prompt_smoke.sh            # all four
#     sbatch --array=3 slurm/submit_long_prompt_smoke.sh  # Gemma only
set -u
PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit from the repo root: sbatch slurm/submit_long_prompt_smoke.sh}"
[ -f "$PROJECT_DIR/src/eval_testset.py" ] || { echo "FATAL: submit from the NLxOCPQ repo root" >&2; exit 1; }
cd "$PROJECT_DIR"
CONFIG="$PROJECT_DIR/configs/eval_config.unleashed_gate.json"
module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH
MODEL_IDX="${SLURM_ARRAY_TASK_ID}"
RUNTIME="$(python3 "$PROJECT_DIR/src/model_runtime.py" "$CONFIG" "$MODEL_IDX")" || exit 1
export HF_HOME="${RUNTIME#*|}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${RUNTIME%%|*}"
echo "Runtime: conda env ${RUNTIME%%|*}, HF cache ${HF_HUB_CACHE}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
python -u "$PROJECT_DIR/src/long_prompt_smoke.py" "$CONFIG" "$MODEL_IDX"
