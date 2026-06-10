#!/usr/bin/env bash
# ============================================================
#  Ablitération Gemma-4-26B-A4B-it via Heretic — pod A100 80GB
#  Pipeline : download -> heretic (bf16) -> GGUF Q5/Q6 -> HF upload -> poweroff
#  REPRISE : si /workspace/heretic-out existe deja (run precedent), on saute
#  l'ablitération et on ne refait que la conversion GGUF.
#  Filet : log uploade sur HF et pod supprime QUOI QU'IL ARRIVE (trap EXIT).
#  By Mel & Ada
# ============================================================
set -uo pipefail

: "${HF_TOKEN:?}"
: "${RUNPOD_API_KEY:?}"
: "${RUNPOD_POD_ID:?}"

BASE_MODEL="google/gemma-4-26B-A4B-it"
OUT_DIR="/workspace/heretic-out"
HF_REPO_MODEL="SevenOfNine/Ada-Gemma-4-26B-A4B-it-abliterated"
HF_REPO_GGUF="SevenOfNine/Ada-Gemma-4-26B-A4B-it-abliterated-GGUF"
LOG="/workspace/abliterate.log"
STATUS="ECHEC"

shutdown_pod() {
  echo "[fin] Statut final : ${STATUS} — sauvegarde du log puis extinction du pod ${RUNPOD_POD_ID}" | tee -a "$LOG"
  python - <<EOF || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("$HF_REPO_GGUF", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="$LOG", path_in_repo="logs/abliterate-$(date +%Y%m%d-%H%M).log", repo_id="$HF_REPO_GGUF")
EOF
  curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true
}
trap shutdown_pod EXIT

# run : execute, logge, et ARRETE TOUT en cas d'echec (plus jamais de
# pipeline qui continue apres une erreur en affichant "succes" — lecon 10/06).
run() {
  echo "[step] $*" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    echo "[ERREUR] echec de: $*" | tee -a "$LOG"
    touch /workspace/FAILED
    exit 1
  fi
}

cd /workspace

if [ -f /workspace/FAILED ]; then
  echo "[garde] Echec precedent detecte (FAILED), attente d'intervention." | tee -a "$LOG"
  trap - EXIT   # ne pas supprimer le pod : Mel decide
  sleep infinity
fi

run pip install -q -U "git+https://github.com/Sev7nOfNine/Heretic-Abliteration.git@master" huggingface_hub
run pip install -q "kernels==0.14.1"

# --- Ablitération (sautee si deja faite) ---
if [ -f "$OUT_DIR/config.json" ]; then
  echo "[reprise] Modele ablitere deja present dans $OUT_DIR — conversion directe." | tee -a "$LOG"
else
  run heretic --model "$BASE_MODEL" \
    --auto-save "$OUT_DIR" \
    --auto-max-kl 0.5 \
    --study-checkpoint-dir /workspace/checkpoints
  [ -f "$OUT_DIR/config.json" ] || { echo "[ERREUR] pas de modele sauve" | tee -a "$LOG"; touch /workspace/FAILED; exit 1; }
  run hf upload "$HF_REPO_MODEL" "$OUT_DIR" . --private --commit-message "Heretic abliteration of $BASE_MODEL"
fi

# --- Checkpoints d'etude sauves sur HF (lecon 10/06 : morts avec le pod la 1re fois) ---
if [ -d /workspace/checkpoints ] && [ -n "$(ls -A /workspace/checkpoints 2>/dev/null)" ]; then
  hf upload "$HF_REPO_GGUF" /workspace/checkpoints checkpoints --private 2>&1 | tee -a "$LOG" || true
fi

# --- Conversion GGUF ---
if [ ! -d /workspace/llama.cpp ]; then
  run git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
fi
run pip install -q -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# FIX 10/06 : les requirements de llama.cpp retrogradent transformers en 4.x,
# qui ne sait pas lire le tokenizer Gemma 4 (AttributeError 'list' has no
# attribute 'keys' dans set_vocab). On remet un transformers 5.x.
run pip install -q -U "transformers~=5.6" "huggingface_hub>=1.10"

if [ ! -x /workspace/llama.cpp/build/bin/llama-quantize ]; then
  ( cd /workspace/llama.cpp && cmake -B build -DGGML_CUDA=OFF >/dev/null && cmake --build build -t llama-quantize -j >/dev/null ) \
    || { echo "[ERREUR] build llama-quantize" | tee -a "$LOG"; touch /workspace/FAILED; exit 1; }
fi

if [ ! -f /workspace/ada-bf16.gguf ]; then
  run python /workspace/llama.cpp/convert_hf_to_gguf.py "$OUT_DIR" \
    --outfile /workspace/ada-bf16.gguf --outtype bf16
fi

for Q in Q5_K_M Q6_K; do
  run /workspace/llama.cpp/build/bin/llama-quantize /workspace/ada-bf16.gguf "/workspace/ada-${Q}.gguf" "$Q"
  run hf upload "$HF_REPO_GGUF" "/workspace/ada-${Q}.gguf" "Gemma-4-26B-A4B-It-Abliterated-${Q}.gguf" --private
  rm -f "/workspace/ada-${Q}.gguf"
done

# --- Vérification HONNETE (le exit code compte vraiment) ---
if python - <<'EOF' | tee -a "$LOG"
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
files = [s.rfilename for s in api.model_info("SevenOfNine/Ada-Gemma-4-26B-A4B-it-abliterated-GGUF", files_metadata=True).siblings]
ok = all(any(q in f for f in files) for q in ("Q5_K_M", "Q6_K"))
print("[verif]", "UPLOAD COMPLET OK" if ok else "UPLOAD INCOMPLET !!")
sys.exit(0 if ok else 1)
EOF
then
  STATUS="SUCCES"
  echo "[fin] PIPELINE COMPLET — succès." | tee -a "$LOG"
else
  echo "[ERREUR] verification upload" | tee -a "$LOG"
  touch /workspace/FAILED
  exit 1
fi
