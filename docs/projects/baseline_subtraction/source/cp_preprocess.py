"""
PCR 신호 데이터 전처리 모듈

- 이 모듈은 Real-Time PCR 신호의 baseline fitting 알고리즘 최적화를 위한 
  데이터 전처리 기능을 제공
- 다양한 출처의 PCR 데이터를 로드하고, 병합하며, 이상치 탐지 및 전처리를 수행
- Data Engineer들이 C++로 porting 해야하기 때문에 pytorch, scikit-learn, 
  scipy 등의 라이브러리 사용 금지

주요 기능:
- PCR 데이터 파일 로드 및 병합
- 신호 이상치 탐지 및 처리
- 신호 특성(선형 기울기 등) 계산
- 시스템 리소스 모니터링

작성자: Kwangmin Kim
날짜: 2024-07
"""

# Analysis Preparation: 데이터 분석을 위한 기본 라이브러리 
import polars as pl
import pandas as pd
import numpy as np
import os
import pathlib
from typing import Dict, List, Tuple, Optional

# Signal Processing: 신호 처리 및 노이즈 검출을 위한 커스텀 함수들
from source.signal_filter import detect_noise_naively


# -------------------------------------------------------------------
# 데이터 로드 및 경로 관련 함수
# -------------------------------------------------------------------

def setup_data_paths(mudt: bool = True) -> Dict[str, str]:
    """
    MuDT 처리 여부에 따른 데이터 파일 경로 설정
    
    입력 파라미터에 따라 MuDT 전처리가 적용된 또는 적용되지 않은 
    데이터셋에 대한 경로를 생성합
    
    매개변수:
        mudt (bool): MuDT 전처리 적용 여부 (기본값: True)
        
    반환값:
        dict: 각 데이터셋 유형에 대한 파일 경로가 포함된 사전
            - 'raw': 원시 데이터 파일 경로
            - 'auto': Auto baseline 데이터 파일 경로
            - 'cfx': CFX 데이터 파일 경로
            - 'strep_plus1': Strep+1 데이터 파일 경로
            - 'strep_plus2': Strep+2 데이터 파일 경로
            - 'ml': ML 데이터 파일 경로
    """
    rootpath = os.getcwd()
    # 공통 경로 부분과 사용자별 경로 부분 분리
        
    if 'Administrator' in rootpath:       
        user_path = 'C:/Users/Administrator'
    else:
        user_path = 'C:/Users/kmkim'
    
    common_path = 'Desktop/projects/website/docs/data/baseline_optimization/GI-B-I'

    base_path = os.path.join(user_path, common_path)
    base_path = base_path.replace('\\', '/')

    paths = {}
    base_suffix = 'mudt_' if mudt else 'no_mudt_'
    
    paths['raw'] = os.path.join(base_path, 'raw_data', f'{base_suffix}raw_data.parquet')
    paths['auto'] = os.path.join(base_path, 'auto_baseline_data', 
                                f'{base_suffix}auto_baseline_data.parquet')
    paths['cfx'] = os.path.join(base_path, 'cfx_data', f'{base_suffix}cfx_data.parquet')
    paths['strep_plus1'] = os.path.join(base_path, 'strep_plus1_data', 
                                        f'{base_suffix}strep_plus1_data.parquet')
    paths['strep_plus2'] = os.path.join(base_path, 'strep_plus2_data', 
                                        f'{base_suffix}strep_plus2_data.parquet')
    paths['ml'] = os.path.join(base_path, 'ml_data', f'{base_suffix}ml_data.parquet')
    
    return paths

def load_all_datasets(data_paths: Dict[str, str]) -> Dict[str, pl.DataFrame]:
    """
    여러 소스에서 데이터셋 로드
    
    각 데이터 파일 경로에서 parquet 파일을 읽어들여 Polars DataFrame으로 변환
    
    매개변수:
        data_paths (dict): setup_data_paths() 함수에서 생성된 데이터 경로 사전
        
    반환값:
        dict: 각 데이터셋 유형에 대한 Polars DataFrame이 포함된 사전
    """
    datasets = {}
    
    for dataset_type in ["raw", "cfx", "auto", "strep_plus1", "strep_plus2", "ml"]:
        try:
            datasets[dataset_type] = pl.scan_parquet(data_paths[dataset_type]).collect()
        except Exception as e:
            datasets[dataset_type] = pl.DataFrame()
    
    return datasets


def load_and_prepare_parquet(
    i_file_path: str, 
    i_selected_columns: Optional[List[str]] = None, 
    i_combo_key_columns: Optional[List[str]] = None, 
    i_rename_columns: Optional[Dict[str, str]] = None
) -> pl.DataFrame:
    """
    parquet 파일을 로드하고 데이터프레임 준비
    
    파일 경로에서 parquet 파일을 로드하고 선택적으로 열 선택, 
    조합 키 생성, 열 이름 변경 등의 작업을 수행
    
    매개변수:
        i_file_path: parquet 파일이 있는 파일 경로
        i_selected_columns: 데이터 분석에 사용할 열 이름 목록
        i_combo_key_columns: 조합 키 생성에 사용할 열 이름 목록
        i_rename_columns: 열 이름 변경을 위한 매핑 딕셔너리
    
    반환값:
        DataFrame: 처리된 데이터프레임
    """
    o_df = pl.scan_parquet(i_file_path).collect()
    
    if i_selected_columns:
        o_df = o_df.select(i_selected_columns)
    
    if i_combo_key_columns:
        o_df = o_df.with_columns(
            pl.concat_str(
                [pl.col(col).cast(pl.Utf8) for col in i_combo_key_columns],
                separator=" "
            ).alias("combo_key")
        )
    
    if i_rename_columns:
        for old_name, new_name in i_rename_columns.items():
            o_df = o_df.rename({old_name: new_name})
    
    return o_df


# -------------------------------------------------------------------
# 데이터 처리 및 계산 관련 함수
# -------------------------------------------------------------------


def get_column_percentiles(i_data: pl.DataFrame, i_column_name: str) -> pl.Series:
    """
    데이터프레임의 특정 열에 대한 백분위 순위 계산
    
    데이터프레임의 지정된 열에 대해 각 값의 백분위 순위를 계산
    
    매개변수:
        i_data: 지표 점수가 포함된 데이터프레임
        i_column_name: 백분위 순위를 계산할 지표 열의 이름
    
    반환값:
        Series: 지정된 열의 점수에 대한 백분위 순위를 포함하는 시리즈
    """
    def get_percentile(i_sorted_data: np.ndarray, i_value: float) -> float:
        """
        정렬된 점수들에 대한 특정 지표 점수의 백분위 순위를 계산
        
        매개변수:
            i_sorted_data: 정렬된 점수들의 numpy 배열
            i_value: 백분위 순위를 계산할 점수
        
        반환값:
            float: 해당 점수의 백분위 순위
        """
        if not i_sorted_data.size:  # Check if sorted data is empty
            return np.nan  # Return NaN for empty data
        count = np.sum(i_sorted_data <= i_value)
        o_percent = 100.0 * count / len(i_sorted_data)
        return np.round(o_percent, 1)
    
    # Convert to pandas for easy manipulation
    pandas_df = i_data.to_pandas()
    non_empty_values = pandas_df[i_column_name].apply(
        lambda x: x if x else np.nan
    ).dropna()
    i_sorted_data = np.sort(non_empty_values.values)
    o_percentiles = non_empty_values.apply(
        lambda x: get_percentile(i_sorted_data, x)
    )
    
    # Convert back to polars
    return pl.Series(o_percentiles.values)


def process_column_for_outliers(
    i_data: pl.DataFrame, 
    i_column: str, 
    i_groupby_columns: List[str], 
    i_function: callable
) -> pl.DataFrame:
    """
    이상치 탐지 및 관련 지표 계산을 위한 열 처리
    
    지정된 함수를 사용하여 이상치를 탐지하고, 이상치 지표를 계산하여
    그룹별 백분위수와 임계값을 계산합니다.
    
    매개변수:
        i_data: 처리할 데이터프레임
        i_column: 이상치 처리를 위한 열 이름
        i_groupby_columns: 백분위 계산을 위한 그룹화 열 이름 목록
        i_function: 이상치 탐지에 사용할 함수
    
    반환값:
        DataFrame: 이상치 관련 열이 추가된 업데이트된 데이터프레임
    """
    # Convert to pandas for processing since some operations are easier there
    pandas_df = i_data.to_pandas()
    
    # Detect outliers and calculate outlierness metric
    pandas_df[f'outlier_naive_residuals_{i_column}'] = pandas_df[i_column].apply(
        lambda cell: i_function(cell)[1]
    )
    pandas_df[f'outlier_naive_metric_{i_column}'] = pandas_df[i_column].apply(
        lambda cell: i_function(cell)[0]
    )
    
    # Calculate 95th percentile cutoff for outlierness metric
    metric_col = f'outlier_naive_metric_{i_column}'
    pandas_df[f'outlier_naive_cutoff_{i_column}'] = pandas_df.groupby(
        i_groupby_columns
    )[metric_col].transform(lambda x: np.percentile(x, 95))
    
    # Calculate metric percentiles within each group
    for name, group in pandas_df.groupby(i_groupby_columns):
        percentiles = get_column_percentiles(
            pl.DataFrame(group), f'outlier_naive_metric_{i_column}'
        )
        percentile_col = f'outlier_naive_metric_percentile_{i_column}'
        pandas_df.loc[group.index, percentile_col] = percentiles.to_numpy()
    
    # Convert back to polars
    return pl.DataFrame(pandas_df)


def compute_lm_slope(i_signal: np.ndarray) -> List[float]:
    """
    신호 데이터의 선형 기울기와 절편 계산
    
    PCR 신호 데이터의 선형 추세를 파악하기 위해 
    공분산 방법으로 기울기와 절편을 계산
    
    매개변수:
        i_signal: 기울기를 계산할 신호 데이터 배열
    
    반환값:
        list: [slope, intercept] 형태의 기울기와 절편 값
    """
    cycle_length = np.size(i_signal)
    cycle = np.arange(1, cycle_length + 1)
    i_signal_mean = np.mean(i_signal)
    cycle_mean = np.mean(cycle)
    
    # Compute covariance matrix
    cov_matrix = np.cov(i_signal, cycle)
    slope = cov_matrix[0, 1] / cov_matrix[0, 0] if cov_matrix[0, 0] != 0 else 0
    intercept = i_signal_mean - slope * cycle_mean
    
    return [slope, intercept]


# -------------------------------------------------------------------
# 시스템 리소스 모니터링 관련 함수
# -------------------------------------------------------------------


def check_memory_status() -> None:
    """
    현재 시스템의 메모리 사용 상태 확인 및 출력
    
    실행 중인 프로세스의 메모리 사용량과 전체 시스템 메모리 상태를 MB 단위로 확인
    
    반환값:
        None: 메모리 정보를 콘솔에 출력만 하고 반환값은 없음
    """
    import psutil  # memory checking
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 2)  # 메가바이트 단위로 변환
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage = mem_info.rss / (1024 ** 2)  # rss는 실제 메모리 사용량을 나타냄
    available_memory = memory.available / (1024 ** 2)  # 메가바이트 단위로 변환
    
    print(f"Memory used: {memory_usage:.2f} MB")
    print(f"Total memory: {total_memory:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")


def get_disk_usage(path: str = "/") -> None:
    """
    디스크 저장 공간 사용 현황 확인 및 출력
    
    지정된 경로의, 또는 기본 경로("/")의 디스크 용량과 
    사용량, 여유 공간을 GB 단위로 출력
    
    매개변수:
        path: 디스크 사용량을 확인할 경로 (기본값: "/")
    
    반환값:
        None: 디스크 사용 정보를 콘솔에 출력만 하고 반환값은 없음
    """
    import shutil
    usage = shutil.disk_usage(path)
    print(f"Total Disk Capacity: {usage.total / (1024**3):.2f} GB")
    print(f"Used Disk Space: {usage.used / (1024**3):.2f} GB")
    print(f"Free Disk Space: {usage.free / (1024**3):.2f} GB")


def get_package_details() -> Dict[str, str]:
    """
    설치된 Python 패키지 목록과 버전 정보 수집
    
    현재 Python 환경에 설치된 모든 패키지의 이름과 
    버전 정보를 사전 형태로 반환
    
    반환값:
        dict: {패키지명: 버전} 형태의 사전
    """
    import subprocess
    import sys
    
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list'], 
        stdout=subprocess.PIPE, 
        text=True
    )
    lines = result.stdout.split('\n')
    
    package_dict = {}
    for line in lines[2:]:  # 첫 두 줄은 헤더 정보이므로 제외
        parts = line.split()
        if len(parts) >= 2:
            package_name = parts[0]
            version = parts[1]
            package_dict[package_name] = version
    return package_dict


# -------------------------------------------------------------------
# 데이터 병합 및 전처리 함수
# -------------------------------------------------------------------


def merge_datasets(datasets: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    여러 데이터셋을 병합하여 하나의 통합 데이터프레임 생성
    
    각 알고리즘의 결과 데이터셋을 'combo_key' 기준으로 병합하여
    베이스라인 fitting 알고리즘 간 비교 분석이 가능한 형태로 만듦
    
    매개변수:
        datasets: 각 알고리즘별 데이터프레임이 포함된 딕셔너리
            - 'raw': 원시 신호 데이터
            - 'cfx': CFX 알고리즘 처리 결과
            - 'auto': Auto 알고리즘 처리 결과
            - 'strep_plus1': Strep+1 알고리즘 처리 결과
            - 'strep_plus2': Strep+2 알고리즘 처리 결과
            - 'ml': ML 알고리즘 처리 결과
    
    반환값:
        DataFrame: 모든 알고리즘의 결과가 병합된 통합 데이터프레임
    """
    # Extract relevant columns from each dataset
    cfx_df = datasets['cfx'].select(['original_rfu_cfx', 'combo_key'])
    
    auto_df = datasets['auto']
    auto_cols = auto_df.columns[6:]
    auto_df = auto_df.select(auto_cols)
    
    strep_plus1_df = datasets['strep_plus1']
    strep_plus1_cols = strep_plus1_df.columns[6:]
    strep_plus1_df = strep_plus1_df.select(strep_plus1_cols)
    
    strep_plus2_df = datasets['strep_plus2']
    strep_plus2_cols = strep_plus2_df.columns[6:]
    strep_plus2_df = strep_plus2_df.select(strep_plus2_cols)
    
    ml_df = datasets['ml'].select([
        'ml_baseline_fit', 'ml_analysis_absd', 'combo_key'
    ])
    
    # Merge all datasets
    merged_data = (ml_df
        .join(datasets['raw'], on='combo_key')
        .join(auto_df, on='combo_key')
        .join(strep_plus1_df, on='combo_key')
        .join(strep_plus2_df, on='combo_key')
        .join(cfx_df, on='combo_key'))
    
    return merged_data


def preprocess_merged_data(
    merged_data: pl.DataFrame, 
    outlier_naive_metric: float, 
    mudt: bool = True
) -> pl.DataFrame:
    """
    병합된 데이터에 전처리 작업 적용
    
    병합된 데이터에 이상치 탐지 및 선형 기울기 계산 등의 전처리 작업을 수행
    
    매개변수:
        merged_data: 병합된 통합 데이터프레임
        outlier_naive_metric: 이상치 탐지를 위한 임계값
        mudt: MuDT 전처리 적용 여부 (기본값: True)
            True일 경우 'preproc_rfu', False일 경우 'original_rfu' 열 사용
    
    반환값:
        DataFrame: 전처리가 적용된 데이터프레임
    """
    # 이상치 탐지
    columns_to_process = ['original_rfu']
    groupby_columns = ['channel', 'temperature']
    
    for column in columns_to_process:
        merged_data = process_column_for_outliers(
            merged_data, column, groupby_columns, detect_noise_naively
        )
    
    # 선형 기울기 계산
    source_column = 'preproc_rfu' if mudt else 'original_rfu'
    
    # Convert to pandas for this operation
    pandas_df = merged_data.to_pandas()
    pandas_df['linear_slope'] = pandas_df[source_column].apply(
        lambda x: compute_lm_slope(x)[0]
    )
    merged_data = pl.DataFrame(pandas_df)
    
    return merged_data


def filter_data_for_analysis(
    merged_data: pl.DataFrame, 
    outlier_naive_metric: float
) -> pl.DataFrame:
    """
    분석용 데이터 필터링
    
    특정 플레이트, 채널, 온도 조건 및 음성 결과('final_ct < 0')와
    이상치 기준('outlier_naive_metric > threshold')을 만족하는 데이터만 필터링
    
    매개변수:
        merged_data: 병합 및 전처리된 데이터프레임
        outlier_naive_metric: 이상치 필터링 임계값
    
    반환값:
        DataFrame: 분석 조건에 맞게 필터링된 데이터프레임
    """
    # 필터링 기준 설정
    filtered_data = merged_data.filter(
        (pl.col('final_ct') < 0) &
        (pl.col(f'outlier_naive_metric_original_rfu') > outlier_naive_metric)
    )
    
    return filtered_data


# -------------------------------------------------------------------
# 통합 함수 및 메인 함수
# -------------------------------------------------------------------


def prepare_baseline_data(
    outlier_naive_metric: float = 1.65, 
    mudt: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    베이스라인 최적화 분석을 위한 데이터 준비
    
    여러 알고리즘 결과 데이터를 로드, 병합, 전처리하여 
    baseline fitting 알고리즘 비교 분석을 위한 데이터셋을 구성
    
    매개변수:
        outlier_naive_metric: 이상치 탐지를 위한 임계값 (기본값: 1.65)
        mudt: MuDT 전처리 적용 여부 (기본값: True)
    
    반환값:
        tuple: (merged_data, filtered_data)
            - merged_data: 모든 알고리즘 결과가 병합된 전체 데이터셋
            - filtered_data: 분석 조건에 맞게 필터링된 데이터셋
    """
    # 데이터 경로 설정
    data_paths = setup_data_paths(mudt)
    
    # 모든 데이터셋 로드
    datasets = load_all_datasets(data_paths)
    
    # 데이터셋 병합
    merged_data = merge_datasets(datasets)
    
    # 데이터 전처리
    merged_data = preprocess_merged_data(merged_data, outlier_naive_metric, mudt)
    
    # 분석용 데이터 필터링
    filtered_data = filter_data_for_analysis(merged_data, outlier_naive_metric)
    
    return merged_data, filtered_data

def main():
    """
    스크립트 실행시의 메인 함수
    
    주요 기능:
    - 데이터 준비 및 전처리
    - 필요한 분석 실행
    - 결과 출력 또는 저장
    """
    print("베이스라인 피팅 최적화를 위한 데이터 전처리 시작")
    
    # 사용자 정의 매개변수 설정
    outlier_naive_metric = 1.65
    mudt = True
    
    # 데이터 로드 및 전처리
    merged_data, filtered_data = prepare_baseline_data(outlier_naive_metric, mudt)
    
    # 추가 작업 수행
    print(f"처리된 데이터 형태: {merged_data.shape}")
    print(f"필터링된 데이터 형태: {filtered_data.shape}")
    
    # 시스템 자원 상태 확인
    check_memory_status()
    get_disk_usage()
    
    print("베이스라인 피팅 최적화를 위한 데이터 전처리 완료!")


if __name__ == "__main__":
    main()