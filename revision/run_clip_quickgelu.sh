#!/usr/bin/env bash
# Re-run the CLIP encoder control (Table 5, Tables S23-S25) with the activation the OpenAI
# CLIP weights were trained with (QuickGELU). The published CLIP rows used open_clip
# 'ViT-L-14' (GELU); this script extracts features with 'ViT-L-14-quickgelu' and runs the
# same encoder experiment (all seven predictors, both gates, 7 closed + 42 open configurations).
#
# Usage (from the repository root, after run_all.sh has created $OUT/inputs.json):
#   GPU=0 bash revision/run_clip_quickgelu.sh
# Env: PY (python), OUT (revision_outputs), GPU (0), JOBS (8)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}; OUT=${OUT:-revision_outputs}; GPU=${GPU:-0}; JOBS=${JOBS:-8}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$PWD/src:$PWD/revision:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if ! curl -sI --max-time 6 https://huggingface.co >/dev/null 2>&1; then export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}; fi
ENC=clip-quickgelu
LOG=$OUT/logs; mkdir -p "$LOG"; ST=$LOG/clip_quickgelu_status.txt; echo $$ > "$LOG/clip_quickgelu.pid"
J() { $PY -c "import json;d=json.load(open('$OUT/inputs.json'));print(d$1)"; }
TCMP_ROOT=$(J "['tcmp_root']"); IND_ROOT=$(J "['indian_root']")
echo "start $(date '+%F %T')  GPU=$GPU JOBS=$JOBS HF_ENDPOINT=${HF_ENDPOINT:-default}" | tee -a "$ST"
$PY -c "import open_clip,torch;print('open_clip',open_clip.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" | tee -a "$ST"
if ! $PY -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"; then
  { echo "FAILED cuda not available to python ($(command -v $PY)) with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"; echo "  nvidia-smi: $(command -v nvidia-smi || echo not on PATH)"
    nvidia-smi --query-gpu=index,name,compute_mode,memory.used,memory.total --format=csv 2>&1 | sed 's/^/  /'
    $PY -c "import torch;torch.zeros(1).cuda()" 2>&1 | tail -n 3 | sed 's/^/  /'; } | tee -a "$ST"
  exit 1
fi

for ds in indian tcmp; do
  root=$TCMP_ROOT; [ $ds = indian ] && root=$IND_ROOT
  F=$OUT/features/${ENC}_$ds
  if [ ! -f "$F/features.npy" ]; then
    echo "== extract $ENC $ds  $(date +%T)" | tee -a "$ST"
    $PY revision/extract_features.py --encoder $ENC --dataset $ds --data-root "$root" --output "$F" --device cuda:0 --workers ${WORKERS:-4} \
      > "$LOG/extract_${ENC}_$ds.log" 2>&1; rc=$?
    if [ $rc -ne 0 ]; then
      sig=""; [ $rc -gt 128 ] && sig=" (killed by signal $((rc-128)))"
      echo "FAILED extract $ds exit=$rc$sig at $(date +%T) (see $LOG/extract_${ENC}_$ds.log)" | tee -a "$ST"; tail -n 20 "$LOG/extract_${ENC}_$ds.log"; exit 1
    fi
  fi
  if grep -q "QuickGELU" "$LOG/extract_${ENC}_$ds.log" 2>/dev/null; then
    echo "WARNING: QuickGELU warning still present in extract_${ENC}_$ds.log" | tee -a "$ST"
  else
    echo "OK extract $ds (no activation warning)" | tee -a "$ST"
  fi
  echo "== w5 $ENC $ds  $(date +%T)" | tee -a "$ST"
  $PY revision/exp_w5_encoders.py --dataset $ds --encoder $ENC --features "$F/features.npy" --ids "$F/ids.json" --output "$OUT" --jobs $JOBS \
    > "$LOG/w5_${ENC}_$ds.log" 2>&1; rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED w5 $ds exit=$rc at $(date +%T) (see $LOG/w5_${ENC}_$ds.log)" | tee -a "$ST"; tail -n 20 "$LOG/w5_${ENC}_$ds.log"; exit 1; fi
  echo "OK w5 $ds" | tee -a "$ST"
done

ZIP=$OUT/clip_quickgelu_results.zip; rm -f "$ZIP"
$PY - "$OUT" "$ZIP" <<'EOF'
import sys, zipfile
from pathlib import Path
o, z = Path(sys.argv[1]), sys.argv[2]; n = 0
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
    for p in [*o.glob('w5/clip-quickgelu/*/*.json'), *o.glob('w5/clip-quickgelu_*_summary.json'),
              *o.glob('features/clip-quickgelu_*/*.json'), *o.glob('logs/*clip_quickgelu*'), *o.glob('logs/*clip-quickgelu*')]:
        f.write(p, p.relative_to(o)); n += 1
print('wrote', z, n, 'files')
EOF
echo "finished $(date '+%F %T')" | tee -a "$ST"
