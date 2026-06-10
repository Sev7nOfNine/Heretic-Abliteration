#!/usr/bin/env bash
# ============================================================
#  Conversion GGUF seule — pod CPU (pas de GPU requis)
#  Telecharge le modele ablitere depuis HF -> GGUF bf16 -> Q5_K_M + Q6_K -> HF
#  Filet : log uploade et pod supprime quoi qu'il arrive (trap EXIT).
#  By Mel & Ada
# ============================================================
set -uo pipefail

: "${HF_TOKEN:?}"
: "${RUNPOD_API_KEY:?}"
: "${RUNPOD_POD_ID:?}"

HF_REPO_MODEL="SevenOfNine/Ada-Gemma-4-26B-A4B-it-abliterated"
HF_REPO_GGUF="SevenOfNine/Ada-Gemma-4-26B-A4B-it-abliterated-GGUF"
LOG="/workspace/convert.log"
STATUS="ECHEC"

shutdown_pod() {
  echo "[fin] Statut final : ${STATUS} — log puis extinction du pod ${RUNPOD_POD_ID}" | tee -a "$LOG"
  python - <<EOF || true
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj="$LOG", path_in_repo="logs/convert-$(date +%Y%m%d-%H%M).log", repo_id="$HF_REPO_GGUF")
EOF
  curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true
}
trap shutdown_pod EXIT

run() {
  echo "[step] $*" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    echo "[ERREUR] echec de: $*" | tee -a "$LOG"
    touch /workspace/FAILED
    exit 1
  fi
}

cd /workspace

# Garde anti-boucle : RunPod relance le conteneur a chaque exit. Sans ca,
# un echec se rejoue a l'infini et brule du credit (lecon 10/06, ~95 boucles).
if [ -f /workspace/FAILED ]; then
  echo "[garde] Echec precedent (FAILED) — attente, pas de re-boucle." | tee -a "$LOG"
  trap - EXIT
  curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true
  sleep infinity
fi

run pip install -q -U huggingface_hub

# --- 1. Telecharger le modele ablitere depuis HF ---
# Download COMPLET tant que le tokenizer manque (lecon 10/06 : un download
# partiel laissait config.json mais pas tokenizer.json -> conversion en boucle).
# hf download est idempotent : il complete les fichiers manquants, ne re-DL pas
# ce qui est deja la et valide.
if [ ! -f /workspace/heretic-out/tokenizer.json ] || [ ! -f /workspace/heretic-out/config.json ]; then
  run hf download "$HF_REPO_MODEL" --local-dir /workspace/heretic-out
fi
# Garde-fou : on n'avance pas sans le tokenizer.
[ -f /workspace/heretic-out/tokenizer.json ] || {
  echo "[ERREUR] tokenizer.json absent apres download" | tee -a "$LOG"; exit 1; }

# --- 2. llama.cpp : converter + quantize ---
if [ ! -d /workspace/llama.cpp ]; then
  run git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
fi
run pip install -q -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
# FIX 10/06 : ces requirements retrogradent transformers en 4.x qui ne lit pas
# le tokenizer Gemma 4 ("'list' object has no attribute 'keys'"). On remonte.
run pip install -q -U "transformers~=5.6" "huggingface_hub>=1.10"

if [ ! -x /workspace/llama.cpp/build/bin/llama-quantize ]; then
  ( cd /workspace/llama.cpp && cmake -B build -DGGML_CUDA=OFF >/dev/null \
      && cmake --build build -t llama-quantize -j >/dev/null ) \
    || { echo "[ERREUR] build llama-quantize" | tee -a "$LOG"; exit 1; }
fi

# --- 3. Conversion bf16 ---
if [ ! -f /workspace/ada-bf16.gguf ]; then
  run python /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/heretic-out \
    --outfile /workspace/ada-bf16.gguf --outtype bf16
fi
# Libere 52 GB : les safetensors restent sur HF de toute facon.
rm -rf /workspace/heretic-out

# --- 4. Quantizations + upload ---
for Q in Q5_K_M Q6_K; do
  run /workspace/llama.cpp/build/bin/llama-quantize /workspace/ada-bf16.gguf "/workspace/ada-${Q}.gguf" "$Q"
  run hf upload "$HF_REPO_GGUF" "/workspace/ada-${Q}.gguf" "Gemma-4-26B-A4B-It-Abliterated-${Q}.gguf" --private
  rm -f "/workspace/ada-${Q}.gguf"
done

# --- 5. Verification honnete ---
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
  echo "[fin] CONVERSION COMPLETE — succès." | tee -a "$LOG"
else
  echo "[ERREUR] verification upload" | tee -a "$LOG"
  exit 1
fi
