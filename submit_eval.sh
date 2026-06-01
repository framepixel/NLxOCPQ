#!/usr/bin/env zsh
#SBATCH --job-name=nlxocpq_eval
#SBATCH --output=nlxocpq-eval-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

source "$HOME/envs/nlxocpq-venv/bin/activate"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:$HOME/envs/nlxocpq-venv/lib:${LD_LIBRARY_PATH:-}"

echo "Starting eval_testset at: $(date)"
echo "Python: $(which python)"; python --version

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set — ensure HF provider runs are disabled or token provided in config."
fi

set -e
python -u eval_testset.py --config eval_config.json 2>&1 | tee eval_testset.full.log
rc=${PIPESTATUS[0]:-0}
echo "eval_testset exit code: $rc"
exit $rc
