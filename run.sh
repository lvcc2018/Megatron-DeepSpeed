#!/bin/bash

# Path
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs
mkdir -p $DIR/tensorboard
LOG_PATH=$DIR/logs
TENSORBOARD_PATH=$DIR/tensorboard

# Data
DATA_PATH=/data/Megatron-LM/data
VOCAB_FILE_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_FILE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
TRAIN_SAMPLES=10_000

# Parallelism
GPUS_PER_NODE=8
TP_SIZE=1
PP_SIZE=2
GLOBAL_BATCH_SIZE=2
MICRO_BATCH_SIZE=1

MASTER_ADDR=localhost
MASTER_PORT=6777

# DeepSpeed
# Bf16 must use z0! it implements its own zero stage 1 equivalent
USE_DEEPSPEED=1
ZERO_STAGE=0
DS_CONFIG_JSON=${BASE_DATA_PATH}/ds_config.json

# Model
NUM_LAYERS=8
NUM_HIDDEN=8
NUM_HEADS=2
SEQ_LEN=512

# Output
SAVE_INTERVAL=50

DTYPE="bf16"

LOG_DIR="$DIR/tensorboard/tp${TP}_pp${PP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_${DTYPE}"
mkdir -p $LOG_DIR

# DeepSpeed config
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $DS_CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": true
  },

  "fp16": {
    "enabled": false,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

# DeepSpeed Args
DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG_JSON} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

MODEL_ARGS=" \
    --pp-partition-method 'type:transformer|embedding' \
    --num-layers $NUM_LAYERS \
    --hidden-size $NUM_HIDDEN \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 192 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --sync-tp-duplicated-parameters \
    --bf16 \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --abort-on-unmet-fused-kernel-constraints \
    --kill-switch-path $KILL_SWITCH_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --pad-vocab-size-to 250880 \
    "

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path $TRAIN_DATA_PATH \
    --valid-weighted-split-paths-path $VALID_DATA_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    --num-workers 2 \
    --valid-num-workers 0 \
    --data-impl mmap \
    "

ALL_ARGS="$MODEL_ARGS $OPTIMIZER_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    $LAUNCHER `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD 2>&1 | tee $LOG_PATH/pretrain_gpt2.log

set +x

echo "END TIME: $(date)"