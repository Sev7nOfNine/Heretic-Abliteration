#!/usr/bin/env bash
# ============================================================
#  Ablitération Gemma-4-26B-A4B-it via Heretic — pod A100 80GB
#  Pipeline : download -> heretic (bf16) -> GGUF Q5/Q6 -> HF upload -> poweroff
#  Filet : le pod se coupe QUOI QU'IL ARRIVE (trap EXIT).
#  By Mel & Ada
# ============================================================
set -uo pipefail

# --- Env requis (injectés via template pod) ---
: "${HF_TOKEN:?}"
: "${RUNPOD_API_KEY:?}"
: "${RUNPOD_POD_ID:?}"

BASE_MODEL="google/gemma-4-26B-A4B-it"
OUT_DIR="/workspace/heretic-out"
HF_REPO_MODEL="SevenOfNine/Gemma-4-26B-A4B-It-Abliterated"
HF_REPO_GGUF="SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-GGUF"
LOG="/workspace/abliterate.log"

shutdown_pod() {
  echo "[fin] Sauvegarde du log puis extinction du pod ${RUNPOD_POD_ID}" | tee -a "$LOG"
  # Le log survit au pod : uploadé sur HF quoi qu'il arrive.
  hf upload "$HF_REPO_GGUF" "$LOG" "logs/abliterate-$(date +%Y%m%d-%H%M).log" --private 2>/dev/null || \
    python -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo('$HF_REPO_GGUF', private=True, exist_ok=True)
api.upload_file(path_or_fileobj='$LOG', path_in_repo='logs/abliterate-last.log', repo_id='$HF_REPO_GGUF')
" || true
  curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true
}
trap shutdown_pod EXIT

run() { echo "[step] $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

cd /workspace

# Garde anti-boucle : si un run précédent a échoué sur ce volume, ne pas
# recommencer en boucle (RunPod redémarre le conteneur à chaque exit).
if [ -f /workspace/FAILED ]; then
  echo "[garde] Echec précédent détecté, attente d'intervention manuelle."
  sleep infinity
fi

# --- 1. Dépendances (fork Mel avec le mode --auto-save) ---
# kernels ÉPINGLÉ à 0.14.1 : kernels 0.15.x casse transformers 5.x
# (ValueError "Either a revision or a version must be specified" dans hub_kernels).
# Reproduit et validé en venv local le 2026-06-10.
run pip install -q -U "git+https://github.com/Sev7nOfNine/Heretic-Abliteration.git@master" huggingface_hub
run pip install -q "kernels==0.14.1"
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

# --- 2. Heretic (bf16, A100 80GB) ---
# Logs live des trials (refus + KL) dans $LOG.
# Checkpoints d'étude dans /workspace/checkpoints (rien n'est perdu en cas de crash).
# --model passé EXPLICITEMENT : sans ça, l'heuristique CLI de Heretic insère
# --model devant le dernier argument et vole la valeur du flag précédent.
run heretic --model "$BASE_MODEL" \
  --auto-save "$OUT_DIR" \
  --auto-max-kl 0.5 \
  --study-checkpoint-dir /workspace/checkpoints

if [ ! -d "$OUT_DIR" ] || [ -z "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "[ERREUR] Pas de modèle sauvé par Heretic — voir $LOG" | tee -a "$LOG"
  touch /workspace/FAILED
  exit 1
fi

# --- 3. Upload du modèle ablitéré (safetensors) ---
run hf upload "$HF_REPO_MODEL" "$OUT_DIR" . --private --commit-message "Heretic abliteration of $BASE_MODEL"

# --- 4. Conversion GGUF + quants ---
run git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
run pip install -q -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cd /workspace/llama.cpp && cmake -B build -DGGML_CUDA=OFF >/dev/null && cmake --build build -t llama-quantize -j >/dev/null
cd /workspace

run python /workspace/llama.cpp/convert_hf_to_gguf.py "$OUT_DIR" \
  --outfile /workspace/ada-bf16.gguf --outtype bf16

# Libérer le disque (safetensors déjà uploadés)
rm -rf "$OUT_DIR" ~/.cache/huggingface/hub 2>/dev/null

for Q in Q5_K_M Q6_K; do
  run /workspace/llama.cpp/build/bin/llama-quantize /workspace/ada-bf16.gguf "/workspace/ada-${Q}.gguf" "$Q"
  run hf upload "$HF_REPO_GGUF" "/workspace/ada-${Q}.gguf" "Gemma-4-26B-A4B-It-Abliterated-${Q}.gguf" --private
  rm -f "/workspace/ada-${Q}.gguf"
done

# --- 5. Vérification upload ---
python - <<'EOF' | tee -a "$LOG"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
files = [s.rfilename for s in api.model_info("SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-GGUF", files_metadata=True).siblings]
ok = all(any(q in f for f in files) for q in ("Q5_K_M", "Q6_K"))
print("[verif] fichiers HF:", files)
print("[verif]", "UPLOAD COMPLET OK" if ok else "UPLOAD INCOMPLET !!")
raise SystemExit(0 if ok else 1)
EOF

echo "[fin] PIPELINE COMPLET — succès." | tee -a "$LOG"
# trap EXIT coupe le pod.
