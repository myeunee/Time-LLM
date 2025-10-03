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
        # 스키마 확인
        if 'task_events' in filepath:
            cols = ['timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id', 
                   'event_type', 'user', 'scheduling_class', 'priority', 
                   'cpu_request', 'mem_request', 'disk_request', 'different_machine_constraint']
            df = pd.read_csv(f, header=None, names=cols)
        elif 'task_usage' in filepath:
            cols = ['start_time', 'end_time', 'job_id', 'task_index', 'machine_id', 
                   'cpu_rate', 'canonical_mem_usage', 'assigned_mem_usage', 
                   'unmapped_page_cache', 'total_page_cache', 'max_mem_usage',
                   'disk_io_time', 'local_disk_space', 'max_cpu_rate', 'max_disk_io_time']
            # 필요한 열만 로드
            df = pd.read_csv(f, header=None, names=cols, usecols=range(15))
        else:
            df = pd.read_csv(f, header=None)
        return df

# 1. task_events 데이터 로드
events_file = os.path.join(data_dir, "task_events/part-00000-of-00500.csv.gz")
if os.path.exists(events_file):
    events_df = read_csv_gz(events_file)
    print(f"이벤트 데이터: {len(events_df)} 행")
else:
    print(f"파일을 찾을 수 없음: {events_file}")
    events_df = pd.DataFrame()

# 2. task_usage 데이터 로드 - 여러 파일 통합
usage_files = glob.glob(os.path.join(data_dir, "task_usage/part-0000[0-9]-of-00500.csv.gz"))
if not usage_files:
    usage_files = [os.path.join(data_dir, "task_usage/part-00000-of-00500.csv.gz")]

print(f"로드할 파일 수: {len(usage_files)}")
usage_dfs = []
for file in usage_files[:10]:  # 처음 10개 파일만 사용 (데이터 크기 제한)
    print(f"로드 중: {file}")
    df = read_csv_gz(file)
    usage_dfs.append(df)

usage_df = pd.concat(usage_dfs, ignore_index=True)
print(f"로드된 총 행 수: {len(usage_df)}")

# 3. 이벤트 유형 분석
if not events_df.empty:
    # 이벤트 유형 개수 확인
    event_counts = events_df['event_type'].value_counts()
    print("\n이벤트 유형 분포:")
    print(event_counts)
    
    # 실패 이벤트 필터링 (FAIL=3, EVICT=2, LOST=6)
    failure_events = events_df[events_df['event_type'].isin([2, 3, 6])]
    print(f"\n실패 이벤트 수: {len(failure_events)}")
    
    # task_id 생성
    events_df['task_id'] = events_df['job_id'].astype(str) + '_' + events_df['task_index'].astype(str)
    failure_events['task_id'] = failure_events['job_id'].astype(str) + '_' + failure_events['task_index'].astype(str)

# 4. 사용량 데이터와 이벤트 데이터 결합
if not events_df.empty and not usage_df.empty:
    # usage_df에 task_id 생성
    usage_df['task_id'] = usage_df['job_id'].astype(str) + '_' + usage_df['task_index'].astype(str)
    
    # 각 task_id에 대한 실패 여부 확인
    failure_tasks = set(failure_events['task_id']) if len(failure_events) > 0 else set()
    usage_df['has_failure'] = usage_df['task_id'].isin(failure_tasks).astype(int)
    
    # 각 task_id에 대한 CPU 변동성 계산 (30분 창의 롤링 표준편차)
    task_groups = usage_df.groupby('task_id')
    
    # CPU 변동성 계산 및 실패 여부 결합
    result_dfs = []
    for task_id, group in task_groups:
        group = group.sort_values('start_time')
        group['cpu_fluct'] = group['cpu_rate'].rolling(6, min_periods=1).std() * 100
        
        # 실패 여부 추가
        has_failure = 1 if task_id in failure_tasks else 0
        group['failure'] = has_failure
        
        result_dfs.append(group)
    
    # 결과 결합
    result_df = pd.concat(result_dfs)
    
    # 시간순 정렬
    result_df = result_df.sort_values('start_time')
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "google_failure_data.csv")
    result_df.to_csv(output_file, index=False)
    
    print(f"\n처리된 행 수: {len(result_df)}")
    print(f"저장 완료: {output_file}")
    
    # 실패율 확인
    failure_rate = result_df['failure'].mean() * 100
    print(f"실패율: {failure_rate:.2f}%")
    
    # 통계 출력
    print("\n데이터 통계:")
    print(f"시작 시간: {pd.to_datetime(result_df['start_time'], unit='us').min()}")
    print(f"종료 시간: {pd.to_datetime(result_df['start_time'], unit='us').max()}")
    print(f"총 시간 범위: {pd.to_datetime(result_df['start_time'], unit='us').max() - pd.to_datetime(result_df['start_time'], unit='us').min()}")
    print(f"CPU 변동성 최솟값: {result_df['cpu_fluct'].min()}")
    print(f"CPU 변동성 최댓값: {result_df['cpu_fluct'].max()}")
    print(f"CPU 변동성 평균: {result_df['cpu_fluct'].mean()}")
else:
    print("이벤트 데이터 또는 사용량 데이터가 비어 있습니다.")
