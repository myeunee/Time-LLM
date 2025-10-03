# /Users/yoonheoz/Time-LLM/prepare_google_csv.py
import pandas as pd
import numpy as np
import os
import gzip
import glob

# 디렉토리 설정
data_dir = "/Users/yoonheoz/Time-LLM/google-cluster-data"
output_dir = "/Users/yoonheoz/Time-LLM/dataset"

# 압축 파일 읽기
def read_csv_gz(filepath):
    with gzip.open(filepath, 'rt') as f:
        # 스키마 확인 (실제 데이터에 맞게 조정 필요)
        if 'task_usage' in filepath:
            cols = ['start_time', 'end_time', 'job_id', 'task_index', 'machine_id', 
                   'cpu_rate', 'canonical_mem_usage', 'assigned_mem_usage', 
                   'unmapped_page_cache', 'total_page_cache', 'max_mem_usage',
                   'disk_io_time', 'local_disk_space', 'max_cpu_rate', 'max_disk_io_time']
            # 필요한 열만 로드
            df = pd.read_csv(f, header=None, names=cols, usecols=range(15))
        elif 'machine_events' in filepath:
            cols = ['timestamp', 'machine_id', 'event_type', 'platform_id', 'cpu_capacity', 'mem_capacity']
            df = pd.read_csv(f, header=None, names=cols)
        elif 'task_events' in filepath:
            cols = ['timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id', 
                   'event_type', 'user', 'scheduling_class', 'priority', 
                   'cpu_request', 'mem_request', 'disk_request', 'different_machine_constraint']
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f, header=None)
        return df

# 1. 태스크 사용량 데이터 로드 - 여러 파일 통합
usage_files = glob.glob(os.path.join(data_dir, "task_usage/part-*.csv.gz"))
if not usage_files:
    usage_files = [os.path.join(data_dir, "task_usage/part-00000-of-00500.csv.gz")]

print(f"로드할 파일 수: {len(usage_files)}")
usage_dfs = []
for file in usage_files:  # 모든 다운로드된 파일 사용
    print(f"로드 중: {file}")
    df = read_csv_gz(file)
    usage_dfs.append(df)

usage_df = pd.concat(usage_dfs, ignore_index=True)
print(f"로드된 총 행 수: {len(usage_df)}")

# 2. 머신 이벤트 데이터 로드
machine_file = os.path.join(data_dir, "machine_events/part-00000-of-00001.csv.gz")
if os.path.exists(machine_file):
    machine_df = read_csv_gz(machine_file)
    print(f"머신 이벤트 데이터: {len(machine_df)} 행")

# 3. 상위 10개 머신 선택 (가장 많은 데이터가 있는 머신)
top_machines = usage_df['machine_id'].value_counts().nlargest(10).index.tolist()
usage_df = usage_df[usage_df['machine_id'].isin(top_machines)].copy()
print(f"선택된 머신 수: {len(top_machines)}")
print(f"필터링 후 행 수: {len(usage_df)}")

# 4. 시간 변환 및 5분 리샘플
usage_df['date'] = pd.to_datetime(usage_df['start_time'], unit='us')
usage_df = usage_df.set_index('date').sort_index()

# 5. 각 머신별로 처리 후 결합
all_resampled = []
for machine in top_machines:
    machine_data = usage_df[usage_df['machine_id'] == machine].copy()
    
    # 필요한 지표만 선택하고 5분 평균
    cols_to_use = ['cpu_rate', 'canonical_mem_usage', 'disk_io_time', 'local_disk_space', 'max_cpu_rate']
    resampled = machine_data[cols_to_use].resample('5min').mean().interpolate(limit_direction='both')
    
    # CPU 변동성 계산 (30분 창의 롤링 표준편차)
    resampled['cpu_fluct'] = resampled['cpu_rate'].rolling(6, min_periods=1).std() * 100
    
    # 머신 ID 추가
    resampled['machine_id'] = machine
    
    all_resampled.append(resampled)

# 모든 머신 데이터 결합
combined = pd.concat(all_resampled)
combined = combined.sort_index()

# 6. 결과 저장
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "google_demo.csv")
final_df = combined.reset_index()
final_df.to_csv(output_file, index=False)

print(f"처리된 행 수: {len(final_df)}")
print(f"저장 완료: {output_file}")
print(f"컬럼: {final_df.columns.tolist()}")

# 7. 데이터 통계 출력
print("\n데이터 통계:")
print(f"시작 시간: {final_df['date'].min()}")
print(f"종료 시간: {final_df['date'].max()}")
print(f"총 시간 범위: {final_df['date'].max() - final_df['date'].min()}")
print(f"머신별 데이터 포인트 수:")
print(final_df['machine_id'].value_counts())