#!/bin/bash

# 1. 먼저 failure 데이터 준비
echo "Preparing failure data..."
python prepare_failure_data.py

# 2. TimeLLM 모델 학습 및 평가
echo "Training TimeLLM for failure prediction..."
/Users/yoonheoz/Time-LLM/venv/bin/accelerate launch --num_processes 1 /Users/yoonheoz/Time-LLM/run_failure_prediction.py \
  --task_name long_term_forecast --is_training 1 --model TimeLLM \
  --root_path /Users/yoonheoz/Time-LLM/dataset --data_path google_failure_data.csv --data ECL \
  --features MS --target failure --freq 5min \
  --seq_len 96 --label_len 12 --pred_len 3 \
  --enc_in 6 --dec_in 6 --c_out 1 \
  --llm_model GPT2 --llm_dim 768 --llm_layers 4 \
  --train_epochs 5 --batch_size 8 --eval_batch_size 4 \
  --learning_rate 0.00001 --loss BCE \
  --num_workers 0 \
  --model_id google_failure --model_comment stable \
  --save_preds_csv --save_plot_png --output_dir /Users/yoonheoz/Time-LLM/outputs-failure

# 3. 결과 시각화
echo "Visualizing results..."
python visualize_failure_prediction.py

echo "All done!"

