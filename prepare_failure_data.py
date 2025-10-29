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

# 2. task_usage 데이터 로드 - 극단적 변동성 찾기
usage_files = glob.glob(os.path.join(data_dir, "task_usage/part-0000[0-9]-of-00500.csv.gz"))
if not usage_files:
    usage_files = [os.path.join(data_dir, "task_usage/part-00000-of-00500.csv.gz")]

print(f"로드할 파일 수: {len(usage_files)}")
print("극단적 CPU 변동성을 가진 task 찾는 중...")

# 모든 파일에서 고변동성 task 찾기
high_fluct_tasks = []
usage_dfs = []

# CPU 변동성이 높은 데이터만 선택
for file in usage_files[:5]:  # 5개 파일 스캔
    print(f"스캔 중: {file}")
    df = read_csv_gz(file)
    
    # max_cpu_rate 기준으로 상위 데이터만 선택 (변동성 높을 가능성)
    # max_cpu_rate가 높거나 cpu_rate 변동이 큰 데이터 선택
    threshold = df['max_cpu_rate'].quantile(0.90)  # 상위 10%
    high_cpu = df[df['max_cpu_rate'] > threshold]
    
    usage_dfs.append(high_cpu)
    print(f"  선택된 행: {len(high_cpu)} / {len(df)} (상위 10%)")
    
    # 충분한 데이터 수집하면 중단
    if len(usage_dfs) > 0 and sum(len(d) for d in usage_dfs) > 100000:
        print(f"충분한 데이터 수집 완료")
        break

usage_df = pd.concat(usage_dfs, ignore_index=True)
print(f"로드된 총 행 수: {len(usage_df)}")

# 3. 이벤트 유형 분석 및 실패 task 식별
failure_tasks = set()
if not events_df.empty:
    # 이벤트 유형 개수 확인
    event_counts = events_df['event_type'].value_counts()
    print("\n이벤트 유형 분포:")
    print(event_counts)
    
    # 실패 이벤트 필터링 (FAIL=3, EVICT=2, LOST=6)
    failure_events = events_df[events_df['event_type'].isin([2, 3, 6])].copy()
    print(f"\n실패 이벤트 수: {len(failure_events)}")
    
    # task_id 생성
    events_df['task_id'] = events_df['job_id'].astype(str) + '_' + events_df['task_index'].astype(str)
    failure_events['task_id'] = failure_events['job_id'].astype(str) + '_' + failure_events['task_index'].astype(str)
    failure_tasks = set(failure_events['task_id'])

# 4. 사용량 데이터와 이벤트 데이터 결합
if not usage_df.empty:
    # usage_df에 task_id가 없으면 생성
    if 'task_id' not in usage_df.columns:
        usage_df['task_id'] = usage_df['job_id'].astype(str) + '_' + usage_df['task_index'].astype(str)
    
    # 각 task_id에 대한 실패 여부 확인
    usage_df['has_failure'] = usage_df['task_id'].isin(failure_tasks).astype(int)
    
    # 각 task_id에 대한 CPU 변동성 계산 (30분 창의 롤링 표준편차)
    task_groups = usage_df.groupby('task_id')
    
    # 먼저 모든 task의 CPU 변동성 계산
    print("CPU 변동성 계산 중...")
    task_stats = []
    for task_id, group in task_groups:
        group = group.sort_values('start_time')
        cpu_fluct = group['cpu_rate'].rolling(6, min_periods=1).std() * 100
        max_fluct = cpu_fluct.max()
        has_failure = 1 if task_id in failure_tasks else 0
        task_stats.append({
            'task_id': task_id,
            'max_fluct': max_fluct,
            'has_failure': has_failure,
            'group': group
        })
    
    # CPU 변동성 기준으로 정렬 (높은 순)
    task_stats.sort(key=lambda x: x['max_fluct'], reverse=True)
    
    print(f"전체 task 수: {len(task_stats)}, 최대 CPU 변동성: {task_stats[0]['max_fluct']:.2f}")
    
    # 선택 기준:
    # 1. 실패 task는 모두 포함
    # 2. CPU 변동성이 높은 task 우선 포함
    # 3. 목표: 총 1,500개 task
    result_dfs = []
    failure_count = 0
    high_fluct_count = 0
    normal_count = 0
    max_tasks = 1500
    
    for stat in task_stats:
        should_include = False
        
        if stat['has_failure']:
            should_include = True
            failure_count += 1
        elif stat['max_fluct'] > 50 and high_fluct_count < 500:  # 변동성 50% 이상
            should_include = True
            high_fluct_count += 1
        elif len(result_dfs) < max_tasks:
            should_include = True
            normal_count += 1
        
        if should_include:
            group = stat['group'].copy()
            group['cpu_fluct'] = group['cpu_rate'].rolling(6, min_periods=1).std() * 100
            group['failure'] = stat['has_failure']
            result_dfs.append(group)
    
    print(f"선택된 task - 실패: {failure_count}, 고변동성: {high_fluct_count}, 일반: {normal_count}")
    
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
