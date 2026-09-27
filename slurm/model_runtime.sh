# Which runtime a model index runs in: its conda environment and its Hugging Face cache.
# Sourced by the agentic submit scripts AFTER the model index is known (notes 43.16).
#
# Why per model: Muse Glimmer (index 8) needs transformers >= 5.15 (its model type did not exist
# before), while every earlier model -- and every recorded run -- used nlxocpq-eval-v5 on 5.5.0.
# Upgrading that environment would change the stack under the completed Ministral arm, so Glimmer
# gets its own clone (scripts/make_eval_env.sh, OCPQ_ENV_VARIANT=glimmer). Its bf16 weights
# (~60 GB) do not fit the 150 GB $HOME quota beside everything else, so its cache is on $HPCWORK
# (scripts/prefetch_glimmer.sh).
#
# The choice follows the INDEX, not an exported variable: sbatch has dropped exported variables
# before (the --case-shard lesson, 2026-09-12), and a Glimmer job in the wrong environment fails
# at load, or worse, a Ministral job in the new one runs on an unrecorded stack.
#
# Sets OCPQ_RUNTIME_ENV, OCPQ_RUNTIME_HF_HOME and OCPQ_RUNTIME_HF_HUB_CACHE; returns 1 on a
# problem, with the reason on stderr.
ocpq_model_runtime() {
  local idx="$1"
  case "$idx" in
    8)
      if [ -z "${HPCWORK:-}" ]; then
        echo "FATAL: model index 8 (Muse Glimmer) reads its weights from \$HPCWORK, which is unset." >&2
        return 1
      fi
      if [ -n "${OCPQ_EVAL_ENV:-}" ] && [ "${OCPQ_EVAL_ENV}" != "nlxocpq-eval-glimmer" ]; then
        echo "NOTE: ignoring OCPQ_EVAL_ENV=${OCPQ_EVAL_ENV} -- model index 8 runs in nlxocpq-eval-glimmer." >&2
      fi
      OCPQ_RUNTIME_ENV="nlxocpq-eval-glimmer"
      OCPQ_RUNTIME_HF_HOME="$HPCWORK/nlxocpq-hf"
      OCPQ_RUNTIME_HF_HUB_CACHE="$OCPQ_RUNTIME_HF_HOME/hub"
      ;;
    *)
      # Exactly what the scripts did before 2026-09-27, so no earlier model's runtime moves.
      OCPQ_RUNTIME_ENV="${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
      OCPQ_RUNTIME_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
      OCPQ_RUNTIME_HF_HUB_CACHE="${HF_HUB_CACHE:-$OCPQ_RUNTIME_HF_HOME/hub}"
      ;;
  esac
  return 0
}
