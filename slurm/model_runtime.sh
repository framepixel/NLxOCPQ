# Which runtime a model index runs in: its conda environment and its Hugging Face cache.
# Sourced by the agentic submit scripts AFTER the model index is known (notes 43.16, 43.18).
#
# The rule: a model runs on the stack of record -- nlxocpq-eval-v5, transformers 5.5.0, on which every
# recorded run so far was made -- unless its model type does not exist there. Muse Glimmer
# (`muse_glimmer`, index 8) and Qwen3.8 (`qwen3_5`, index 9) need a newer transformers, so they run in
# nlxocpq-eval-glimmer (a clone of the eval env with transformers 5.17.0 and nothing else moved;
# scripts/make_eval_env.sh, OCPQ_ENV_VARIANT=glimmer -- the name predates Qwen). Gemma 4 (`gemma4`,
# index 10) was written against 5.5.0 and stays on the stack of record. Upgrading nlxocpq-eval-v5 itself
# would change the stack under the completed Ministral run.
#
# The weights of 8, 9 and 10 (55-63 GB each in bf16) do not fit the 150 GB $HOME quota, so their cache
# is on $HPCWORK (scripts/prefetch_pinned_models.sh). Every other index keeps ~/.cache/huggingface.
#
# The choice follows the INDEX, not an exported variable: sbatch has dropped exported variables
# before (the --case-shard lesson, 2026-09-12), and a model in the wrong environment fails at load,
# or worse, runs on an unrecorded stack.
#
# Sets OCPQ_RUNTIME_ENV, OCPQ_RUNTIME_HF_HOME and OCPQ_RUNTIME_HF_HUB_CACHE; returns 1 on a
# problem, with the reason on stderr.
ocpq_model_runtime() {
  local idx="$1"
  case "$idx" in
    8|9|10)
      if [ -z "${HPCWORK:-}" ]; then
        echo "FATAL: model index ${idx} reads its weights from \$HPCWORK, which is unset." >&2
        return 1
      fi
      OCPQ_RUNTIME_HF_HOME="$HPCWORK/nlxocpq-hf"
      OCPQ_RUNTIME_HF_HUB_CACHE="$OCPQ_RUNTIME_HF_HOME/hub"
      if [ "$idx" = "10" ]; then
        # Gemma 4: the stack of record, exactly as for the models before it.
        OCPQ_RUNTIME_ENV="${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
      else
        if [ -n "${OCPQ_EVAL_ENV:-}" ] && [ "${OCPQ_EVAL_ENV}" != "nlxocpq-eval-glimmer" ]; then
          echo "NOTE: ignoring OCPQ_EVAL_ENV=${OCPQ_EVAL_ENV} -- model index ${idx} runs in nlxocpq-eval-glimmer." >&2
        fi
        OCPQ_RUNTIME_ENV="nlxocpq-eval-glimmer"
      fi
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
