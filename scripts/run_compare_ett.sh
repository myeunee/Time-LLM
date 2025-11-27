#!/bin/bash

#SBATCH --job-name=timellm_compare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition=batch_ce_ugrad

# 작업 디렉토리
cd /data/myeunee/graduation_proj/Time-LLM

# Conda 환경 활성화
source ~/.bashrc
conda activate /data/myeunee/timellm

# 로그 디렉토리 생성
mkdir -p logs

# 비교 실험 실행
bash scripts/compare_ett.sh | tee logs/compare_ett_batch_patience_release.log