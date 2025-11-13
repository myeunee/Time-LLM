#!/bin/bash

set -e

# ETT 데이터로 baseline vs MLP vs LSTM 비교
# 데이터는 ETTh1을 사용

MODEL=TimeLLM
COMMENT_NONE="compare-ett-none"
COMMENT_MLPLSTM="compare-ett-mlplstm"

# 데이터셋 종류 선택: ETTh1(시간) 또는 ETTm1(분)
DATASET=${1:-ETTh1}        # 기본: ETTh1
DATAPATH=${2:-${DATASET}.csv}

COMMON_ARGS_BASE="--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ${DATAPATH} \
  --model_id ${DATASET}_256_32 \
  --model $MODEL \
  --data ${DATASET} \
  --features M \
  --target OT \
  --seq_len 256 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 16 \
  --num_workers 0 \
  --learning_rate 3e-4 \
  --llm_model GPT2 \
  --llm_layers 2 \
  --train_epochs 1"

echo "[${DATASET}] Baseline (none)"
python3 run_main.py $COMMON_ARGS_BASE --model_comment $COMMENT_NONE --extra_head none

echo "[${DATASET}] With MLP+LSTM"
python3 run_main.py $COMMON_ARGS_BASE --model_comment $COMMENT_MLPLSTM --extra_head mlp_lstm


