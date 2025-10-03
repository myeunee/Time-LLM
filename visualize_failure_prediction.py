import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 디렉토리 설정
output_dir = "/Users/yoonheoz/Time-LLM/outputs-failure"

# TimeLLM 예측 결과 로드
predictions_file = os.path.join(output_dir, "predictions_google_failure_stable.csv")
predictions = pd.read_csv(predictions_file)

# 1. ROC 곡선 그리기
plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(predictions['true_binary'].values, predictions['pred_prob'].values)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
roc_plot_file = os.path.join(output_dir, "roc_curve.png")
plt.savefig(roc_plot_file, dpi=300, bbox_inches='tight')
print(f"ROC 곡선 저장됨: {roc_plot_file}")

# 2. Precision-Recall 곡선 그리기
plt.figure(figsize=(10, 8))
precision, recall, thresholds = precision_recall_curve(predictions['true_binary'].values, predictions['pred_prob'].values)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
pr_plot_file = os.path.join(output_dir, "pr_curve.png")
plt.savefig(pr_plot_file, dpi=300, bbox_inches='tight')
print(f"PR 곡선 저장됨: {pr_plot_file}")

# 3. 예측 확률 분포 그리기
plt.figure(figsize=(12, 6))
sns.histplot(data=predictions, x='pred_prob', hue='true_binary', bins=50, kde=True)
plt.xlabel('예측 확률')
plt.ylabel('빈도')
plt.title('실패 예측 확률 분포')
plt.grid(True)
dist_plot_file = os.path.join(output_dir, "probability_distribution.png")
plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
print(f"확률 분포 그래프 저장됨: {dist_plot_file}")

# 4. 시간에 따른 예측 확률 변화 (첫 50개 샘플)
plt.figure(figsize=(12, 6))
for sample in range(min(10, predictions['sample'].max() + 1)):  # 처음 10개 샘플만 표시
    sample_data = predictions[predictions['sample'] == sample]
    plt.plot(sample_data['step'], sample_data['pred_prob'], marker='o', label=f'Sample {sample}')

plt.axhline(y=0.5, color='red', linestyle='--', label='임계값 (0.5)')
plt.xlabel('예측 시점')
plt.ylabel('실패 확률')
plt.title('시간에 따른 실패 확률 변화')
plt.legend()
plt.grid(True)
time_plot_file = os.path.join(output_dir, "time_series_probability.png")
plt.savefig(time_plot_file, dpi=300, bbox_inches='tight')
print(f"시계열 확률 그래프 저장됨: {time_plot_file}")

# 5. 혼동 행렬 시각화
confusion_file = os.path.join(output_dir, "confusion_google_failure_stable.csv")
conf_matrix = pd.read_csv(confusion_file).values

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['예측: 정상', '예측: 실패'],
            yticklabels=['실제: 정상', '실제: 실패'])
plt.title('혼동 행렬')
plt.tight_layout()
cm_plot_file = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
print(f"혼동 행렬 그래프 저장됨: {cm_plot_file}")

print("모든 시각화 완료!")

