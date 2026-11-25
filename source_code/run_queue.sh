#!/usr/bin/env bash
# 训练 + 评测总队列（数据/输出都在 NFS 项目目录，环境在节点本地盘）
# 逻辑：每个任务开始前检查 GPU 空闲显存，不足则等待；任务串行执行
set -u
ROOT=/private/zdy/deepseek_road
LOCAL=/root/rsc_local
PY=$LOCAL/env/bin/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$LOCAL/cache/huggingface
export TORCH_HOME=$LOCAL/cache/torch
export TMPDIR=$LOCAL/cache/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
LOG=$LOCAL/logs/queue.log
LOCK=$LOCAL/logs/queue.lock
exec >> "$LOG" 2>&1

if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
  echo "[$(date)] 队列已在运行 (pid $(cat $LOCK))，退出"
  exit 0
fi
echo $$ > "$LOCK"
echo "================ QUEUE START $(date) pid=$$ ================"

pick_gpu() {
  local best=-1 bestfree=0 free
  for i in 0 1; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i 2>/dev/null || echo 0)
    free=${free%.*}
    if [ "${free:-0}" -gt "$bestfree" ]; then best=$i; bestfree=$free; fi
  done
  if [ "$bestfree" -lt 9000 ]; then echo ""; else echo "$best"; fi
}
wait_gpu() {
  local g
  while true; do
    g=$(pick_gpu)
    if [ -n "$g" ]; then echo "$g"; return; fi
    echo "[$(date)] GPU 空闲显存不足 9GB，60s 后重试" >> $LOCAL/logs/queue_wait.log
    sleep 60
  done
}

train_job() {
  local name="$1" manifest="$2" labelcol="$3" ncls="$4" trainsp="$5" valsp="$6" testsp="$7" out="$8" epochs="$9" bs="${10}" workers="${11}"
  local gpu
  gpu=$(wait_gpu)
  echo "--- [$(date)] 开始训练: $name (GPU$gpu) ---"
  CUDA_VISIBLE_DEVICES=$gpu TORCH_THREADS=2 $PY $LOCAL/code/train_cls.py \
    --root $ROOT --manifest "$manifest" --label-col "$labelcol" --num-classes "$ncls" \
    --train-split "$trainsp" --val-split "$valsp" --test-split "$testsp" \
    --out "$out" --backbone resnet50 --epochs "$epochs" --bs "$bs" --lr 1e-4 \
    --workers "$workers" --class-weights
  echo "--- [$(date)] 训练结束: $name exit=$? ---" >> $LOCAL/logs/queue_status.log
  echo "$name done $(date)" >> $LOCAL/logs/queue_status.log
}

eval_job() {
  local name="$1" ckpt="$2" manifest="$3" split="$4" domain="$5" ncls="$6" labelcol="$7" out="$8"
  local gpu
  if [ ! -f "$ckpt" ]; then echo "--- [$(date)] 跳过评测 $name：缺少 $ckpt ---"; return; fi
  gpu=$(wait_gpu)
  echo "--- [$(date)] 开始评测: $name (GPU$gpu) ---"
  local extra=""
  if [ -n "$domain" ]; then extra="--eval-domain $domain"; fi
  CUDA_VISIBLE_DEVICES=$gpu TORCH_THREADS=2 $PY $LOCAL/code/train_cls.py --eval-only \
    --root $ROOT --ckpt "$ckpt" --eval-manifest "$manifest" --eval-split "$split" $extra \
    --label-col "$labelcol" --num-classes "$ncls" --workers 4 --out "$out"
  echo "--- [$(date)] 评测结束: $name exit=$? ---"
}

tta_job() {
  local name="$1" ckpt="$2" manifest="$3" split="$4" domain="$5" ncls="$6" labelcol="$7" risk="$8" unknown="$9" out="${10}"
  local gpu
  if [ ! -f "$ckpt" ]; then echo "--- [$(date)] 跳过 TTA $name：缺少 $ckpt ---"; return; fi
  gpu=$(wait_gpu)
  echo "--- [$(date)] 开始 TTA: $name (GPU$gpu) ---"
  local extra=""
  if [ -n "$domain" ]; then extra="--domain $domain"; fi
  CUDA_VISIBLE_DEVICES=$gpu TORCH_THREADS=2 $PY $LOCAL/code/evaluate_tta.py \
    --ckpt "$ckpt" --num-classes "$ncls" --label-col "$labelcol" \
    --manifest "$manifest" --split "$split" $extra --root $ROOT \
    --methods source,bn,tent,sar,cotta --steps 1 --lr 1e-4 \
    --risk "$risk" --unknown-classes "$unknown" --device cuda \
    --out "$out" --bs 64 --workers 4
  echo "--- [$(date)] TTA 结束: $name exit=$? ---"
}

# ============ 任务队列 ============
# 1) RoadSaW12 域内
train_job "roadsaw12" $ROOT/manifests/track1_roadsaw.csv label15 12 train validation test \
          $ROOT/runs/v2_roadsaw12 25 128 6
# 1b) CtD1/2/3 源模型评测（受控域偏移）
for d in ctd1 ctd2 ctd3; do
  eval_job "roadsaw12_${d}" $ROOT/runs/v2_roadsaw12/best.pt $ROOT/manifests/track1_ctd.csv test $d 12 label15 $ROOT/runs/v2_eval_${d}
done

# 2) Track-1 15 类联合
train_job "track1_15cls" $ROOT/manifests/track1_combined.csv label15 15 train validation test \
          $ROOT/runs/v2_track1_15cls 25 128 6

# 3) 清华 6 类域内
train_job "tsinghua6" $ROOT/manifests/track2_tsinghua_train.csv label6 6 train val test \
          $ROOT/runs/v2_tsinghua6 15 256 6
# 3b) 清华 -> 新疆 / 清华 -> 夏季序列 / 新疆 -> 清华
eval_job "ts2xj" $ROOT/runs/v2_tsinghua6/best.pt $ROOT/manifests/track2_xinjiang.csv test "" 6 label6 $ROOT/runs/v2_eval_ts2xj
eval_job "ts2testseq" $ROOT/runs/v2_tsinghua6/best.pt $ROOT/manifests/track2_tsinghua_testseq.csv testseq "" 6 label6 $ROOT/runs/v2_eval_ts2testseq
XJ_CKPT=$ROOT/runs/xj_r50_domain/best.pt
if [ ! -f "$XJ_CKPT" ]; then XJ_CKPT=$ROOT/runs/xj_r50/best.pt; fi
eval_job "xj2ts" $XJ_CKPT $ROOT/manifests/track2_tsinghua_train.csv test "" 6 label6 $ROOT/runs/v2_eval_xj2ts

# 4) TTA 基线（受控域偏移 + 跨域）
RISK12="1,3,5,7,1,3,5,7,1,3,5,7"
for d in ctd1 ctd2 ctd3; do
  tta_job "tta_roadsaw12_${d}" $ROOT/runs/v2_roadsaw12/best.pt $ROOT/manifests/track1_ctd.csv test $d 12 label15 "$RISK12" "" $ROOT/runs/v2_tta_${d}
done
tta_job "tta_ts2xj" $ROOT/runs/v2_tsinghua6/best.pt $ROOT/manifests/track2_xinjiang.csv test "" 6 label6 "1,3,7,9,8,10" "2" $ROOT/runs/v2_tta_ts2xj
tta_job "tta_xj2ts" $XJ_CKPT $ROOT/manifests/track2_tsinghua_train.csv test "" 6 label6 "1,3,7,9,8,10" "2" $ROOT/runs/v2_tta_xj2ts

rm -f "$LOCK"
echo "================ QUEUE DONE $(date) ================"
