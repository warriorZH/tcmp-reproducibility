#!/usr/bin/env bash
# Additional PropGate experiments (OOD-filtered self-training, zero-shot and
# fine-tuning baselines, k x alpha sensitivity, other encoders, development grid).
# Stage names: w2 = OOD-filtered self-training, w3 = zeroshot/finetune,
#              w4 = k x alpha sensitivity, w5 = other encoders, dev = development grid.
# Usage (from the repository root):
#   TCMP_ROOT=/data/tcmp-300 INDIAN_ROOT="/data/Medicinal Leaf dataset" bash revision/run_all.sh
#   STAGES="w2 w4" bash revision/run_all.sh   # selected stages only (resumable; re-run to continue)
# Inputs (env): TCMP_ROOT, INDIAN_ROOT, optional BIOCLIP2_CKPT, TCMP_FEATURES, INDIAN_FEATURES
# Other overrides: PY, JOBS (parallel CPU contexts), DEVICE (cuda:0), OUT (revision_outputs)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${OUT:-revision_outputs}
DEVICE=${DEVICE:-cuda:0}
NCPU=$(nproc 2>/dev/null || echo 8)
JOBS=${JOBS:-$(( NCPU / 4 > 8 ? 8 : (NCPU / 4 < 1 ? 1 : NCPU / 4) ))}
STAGES=${STAGES:-"inputs w2 w4 zeroshot w5 dev finetune"}
export PYTHONPATH="$PWD/src:$PWD/revision:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT/logs" "$OUT/features"
STATUS="$OUT/logs/status.txt"
if ! curl -sI --max-time 6 https://huggingface.co >/dev/null 2>&1; then export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}; fi
echo "start $(date)  PY=$PY JOBS=$JOBS DEVICE=$DEVICE HF_ENDPOINT=${HF_ENDPOINT:-default}" | tee -a "$STATUS"
$PY -c "import sys,numpy,scipy,sklearn,torch;print(sys.version);print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee "$OUT/logs/env.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/logs/env.txt" 2>&1
lscpu 2>/dev/null | grep -m1 'Model name' >> "$OUT/logs/env.txt"

has() { [[ " $STAGES " == *" $1 "* ]]; }
step() { # step <name> <cmd...>
  local name=$1; shift
  if grep -q "^OK $name\$" "$STATUS" 2>/dev/null; then echo "skip $name (done)"; return 0; fi
  echo "== $name  $(date +%T)"; local t0=$(date +%s)
  if "$@" > "$OUT/logs/$name.log" 2>&1; then echo "OK $name" >> "$STATUS"; echo "   ok $name ($(( $(date +%s) - t0 ))s)"
  else echo "FAILED $name (see $OUT/logs/$name.log)" | tee -a "$STATUS"; tail -n 15 "$OUT/logs/$name.log"; fi
}
J() { $PY -c "import json,sys;d=json.load(open('$OUT/inputs.json'));v=d$1;print(v if v is not None else '')"; }

# ---- inputs -------------------------------------------------------------
if has inputs || [ ! -f "$OUT/inputs.json" ]; then
  $PY revision/find_inputs.py --output "$OUT" 2>&1 | tee "$OUT/logs/find_inputs.log"
fi
TCMP_ROOT=$(J "['tcmp_root']"); IND_ROOT=$(J "['indian_root']"); CKPT=$(J "['checkpoint']")
echo "TCMP_ROOT=$TCMP_ROOT  IND_ROOT=$IND_ROOT  CKPT=$CKPT" | tee -a "$STATUS"
if [ -z "$CKPT" ]; then
  $PY -m pip show huggingface_hub >/dev/null 2>&1 || $PY -m pip install -q huggingface_hub safetensors
  $PY scripts/download_checkpoint.py --output checkpoints/bioclip-2 ${HF_ENDPOINT:+--endpoint $HF_ENDPOINT} 2>&1 | tee "$OUT/logs/download_checkpoint.log"
  CKPT=$PWD/checkpoints/bioclip-2/open_clip_model.safetensors
fi
for ds in tcmp indian; do
  F=$(J "['${ds}_features']['features'] if d['${ds}_features'] else None"); I=$(J "['${ds}_features']['ids'] if d['${ds}_features'] else None")
  if [ -z "$F" ]; then
    root=$TCMP_ROOT; [ $ds = indian ] && root=$IND_ROOT
    bs=4; [ $ds = indian ] && bs=2
    step extract_bioclip2_$ds $PY -m tcmp_repro.extract --dataset $ds --data-root "$root" --model-dir "$(dirname "$CKPT")" --output "$OUT/features/bioclip2_$ds" --device $DEVICE --batch-size $bs
    F=$OUT/features/bioclip2_$ds/features.npy; I=$OUT/features/bioclip2_$ds/ids.json
  fi
  eval "F_$ds=\"$F\"; I_$ds=\"$I\""
done
echo "features: tcmp=$F_tcmp indian=$F_indian" | tee -a "$STATUS"

# ---- CPU chain (background): W2 OOD-filtered self-training, W4 k x alpha -------
cpu_chain() {
  for ds in indian tcmp; do
    eval "F=\$F_$ds; I=\$I_$ds"
    has w2 && step w2_$ds $PY revision/exp_w2_ood_selftrain.py --dataset $ds --features "$F" --ids "$I" --output "$OUT" --jobs $JOBS
    has w4 && step w4_$ds $PY revision/exp_w4_sensitivity.py --dataset $ds --features "$F" --ids "$I" --output "$OUT" --jobs $JOBS
  done
}
cpu_chain & CPU_PID=$!

# ---- GPU chain: zero-shot, other encoders, development classes, fine-tuning ----
for ds in indian tcmp; do
  eval "F=\$F_$ds; I=\$I_$ds"
  has zeroshot && step zeroshot_$ds $PY revision/exp_w3_zeroshot.py --dataset $ds --features "$F" --ids "$I" --checkpoint "$CKPT" --output "$OUT" --device $DEVICE
done
if has w5; then
  $PY -c "import transformers" 2>/dev/null || $PY -m pip install -q "transformers>=4.40,<5" 2>&1 | tail -2
  for enc in dinov2 clip-quickgelu; do for ds in indian tcmp; do
    root=$TCMP_ROOT; [ $ds = indian ] && root=$IND_ROOT
    step extract_${enc}_$ds $PY revision/extract_features.py --encoder $enc --dataset $ds --data-root "$root" --output "$OUT/features/${enc}_$ds" --device $DEVICE
  done; done
  for enc in dinov2 clip-quickgelu; do for ds in indian tcmp; do
    [ -f "$OUT/features/${enc}_$ds/features.npy" ] && step w5_${enc}_$ds $PY revision/exp_w5_encoders.py --dataset $ds --encoder $enc --features "$OUT/features/${enc}_$ds/features.npy" --ids "$OUT/features/${enc}_$ds/ids.json" --output "$OUT" --jobs $JOBS
  done; done
  step w5_bioclip2_indian $PY revision/exp_w5_encoders.py --dataset indian --encoder bioclip2 --features "$F_indian" --ids "$I_indian" --output "$OUT" --jobs $JOBS
fi
if has dev; then
  step extract_bioclip2_tcmp_dev $PY revision/extract_features.py --encoder bioclip2 --checkpoint "$CKPT" --dataset tcmp_dev --data-root "$TCMP_ROOT" --output "$OUT/features/bioclip2_tcmp_dev" --device $DEVICE
  [ -f "$OUT/features/bioclip2_tcmp_dev/features.npy" ] && step dev_grid $PY revision/exp_dev_grid.py --features-dir "$OUT/features/bioclip2_tcmp_dev" --output "$OUT" --jobs $JOBS
fi
if has finetune; then
  for bb in bioclip2_vitl14 resnet50_imagenet; do for ds in indian tcmp; do
    root=$TCMP_ROOT; [ $ds = indian ] && root=$IND_ROOT
    step finetune_${bb}_$ds $PY revision/exp_w3_finetune.py --dataset $ds --data-root "$root" --checkpoint "$CKPT" --backbone $bb --output "$OUT" --device $DEVICE
  done; done
fi
wait $CPU_PID
$PY revision/collect.py --output "$OUT"
echo "finished $(date)" | tee -a "$STATUS"
