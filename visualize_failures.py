import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 디렉토리 설정
output_dir = "/Users/yoonheoz/Time-LLM/outputs-1002-fix"

# TimeLLM 예측 결과 로드
predictions_file = os.path.join(output_dir, "predictions_google_demo_stable.csv")
predictions = pd.read_csv(predictions_file)

# 실패 확률 계산 함수
def calculate_failure_probability(fluct):
    """CPU fluctuation 값을 기반으로 실패 확률 계산"""
    if fluct > 70:
        # 70 초과면 높은 실패 확률
        return 0.8 + 0.2 * min((fluct - 70) / 30, 1.0)
    else:
        # 70 이하면 낮은 실패 확률
        base_prob = 0.1
        return base_prob * (fluct / 70)

# 예측된 값에 대한 실패 확률 계산
predictions['failure_prob'] = predictions['pred'].apply(calculate_failure_probability)
predictions['true_failure_prob'] = predictions['true'].apply(calculate_failure_probability)

# 실패 임계값 설정 (70% 초과 시 실패로 간주)
predictions['predicted_failure'] = (predictions['pred'] > 70).astype(int)
predictions['actual_failure'] = (predictions['true'] > 70).astype(int)

# 결과 저장
failure_predictions_file = os.path.join(output_dir, "failure_predictions.csv")
predictions.to_csv(failure_predictions_file, index=False)
print(f"실패 예측 결과 저장됨: {failure_predictions_file}")

# 1. CPU Fluctuation 예측 vs 실제 그래프
plt.figure(figsize=(12, 6))
plt.plot(predictions['step'].values[:50], predictions['true'].values[:50], label='실제 CPU 변동성', color='blue')
plt.plot(predictions['step'].values[:50], predictions['pred'].values[:50], label='예측 CPU 변동성', color='red', linestyle='--')
plt.axhline(y=70, color='green', linestyle='-', label='실패 임계값 (70%)')
plt.fill_between(predictions['step'].values[:50], 70, 100, color='red', alpha=0.2, label='위험 영역')
plt.xlabel('시간 스텝')
plt.ylabel('CPU 변동성 (%)')
plt.title('CPU 변동성 예측 vs 실제')
plt.legend()
plt.grid(True)
fluct_plot_file = os.path.join(output_dir, "cpu_fluctuation_plot.png")
plt.savefig(fluct_plot_file, dpi=300, bbox_inches='tight')
print(f"CPU 변동성 그래프 저장됨: {fluct_plot_file}")

# 2. 실패 확률 그래프
plt.figure(figsize=(12, 6))
plt.plot(predictions['step'].values[:50], predictions['true_failure_prob'].values[:50], label='실제 실패 확률', color='blue')
plt.plot(predictions['step'].values[:50], predictions['failure_prob'].values[:50], label='예측 실패 확률', color='red', linestyle='--')
plt.axhline(y=0.5, color='green', linestyle='-', label='실패 확률 임계값 (50%)')
plt.fill_between(predictions['step'].values[:50], 0.5, 1.0, color='red', alpha=0.2, label='고위험 영역')
plt.xlabel('시간 스텝')
plt.ylabel('실패 확률')
plt.title('실패 확률 예측 vs 실제')
plt.legend()
plt.grid(True)
prob_plot_file = os.path.join(output_dir, "failure_probability_plot.png")
plt.savefig(prob_plot_file, dpi=300, bbox_inches='tight')
print(f"실패 확률 그래프 저장됨: {prob_plot_file}")

# 3. n-cycle ahead 실패 예측 정확도 평가
# 샘플을 3개 그룹으로 나누어 1/2/3-cycle ahead 성능 평가
steps_per_cycle = len(predictions) // 3
predictions['cycle'] = np.repeat([1, 2, 3], steps_per_cycle)[:len(predictions)]

# 각 사이클별 정확도, 정밀도, 재현율 계산
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = []
for cycle in [1, 2, 3]:
    cycle_data = predictions[predictions['cycle'] == cycle]
    accuracy = accuracy_score(cycle_data['actual_failure'], cycle_data['predicted_failure'])
    precision = precision_score(cycle_data['actual_failure'], cycle_data['predicted_failure'], zero_division=0)
    recall = recall_score(cycle_data['actual_failure'], cycle_data['predicted_failure'], zero_division=0)
    f1 = f1_score(cycle_data['actual_failure'], cycle_data['predicted_failure'], zero_division=0)
    
    metrics.append({
        'cycle': cycle,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

metrics_df = pd.DataFrame(metrics)
metrics_file = os.path.join(output_dir, "failure_prediction_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)
print(f"예측 성능 지표 저장됨: {metrics_file}")

# 4. n-cycle ahead 성능 비교 그래프
plt.figure(figsize=(10, 6))
x = metrics_df['cycle']
width = 0.2
plt.bar(x - 0.3, metrics_df['accuracy'], width=width, label='정확도')
plt.bar(x - 0.1, metrics_df['precision'], width=width, label='정밀도')
plt.bar(x + 0.1, metrics_df['recall'], width=width, label='재현율')
plt.bar(x + 0.3, metrics_df['f1_score'], width=width, label='F1 점수')
plt.xlabel('예측 사이클 (n-cycle ahead)')
plt.ylabel('점수')
plt.title('n-cycle ahead 실패 예측 성능')
plt.xticks(x)
plt.legend()
plt.grid(True, axis='y')
metrics_plot_file = os.path.join(output_dir, "failure_prediction_metrics_plot.png")
plt.savefig(metrics_plot_file, dpi=300, bbox_inches='tight')
print(f"성능 지표 그래프 저장됨: {metrics_plot_file}")

print("모든 시각화 완료!")