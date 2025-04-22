"""
PCR 신호 데이터 전처리 모듈

- 이 모듈은 Real-Time PCR 신호의 baseline fitting 알고리즘 최적화를 위한 데이터 전처리 기능을 제공
- 다양한 출처의 PCR 데이터를 로드하고, 병합하며, 이상치 탐지 및 전처리를 수행
- Data Engineer들이 C++로 porting 해야하기 때문에 pytorch, scikit-learn, scipy 등의 라이브러리 사용 금지

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
from scipy.stats import chi2 # https://en.cppreference.com/w/cpp/numeric/random/chi_squared_distribution

# PreProcessing: 데이터 전처리를 위한 유틸리티 라이브러리
import pprint
import pyarrow as pa
import os
import subprocess
from pathlib import Path

# Signal Processing: 신호 처리 및 노이즈 검출을 위한 커스텀 함수들
from source.signal_filter import (detect_noise_naively,
                                    detect_noise_naively_ver2,
                                    detect_noise_naively_ywj1,
                                    detect_noise_naively_pbg,
                                    detect_noise_naively_kkm,
                                    compute_autocorrelation,
                                    test_white_noise)

# Visualization: 데이터 시각화를 위한 라이브러리와 커스텀 함수들
import plotly.express as px
import matplotlib.pyplot as plt
from source.visualization import (find_sub_extremes,
                                    find_global_extremes,
                                    get_comparison_metrics,
                                    compute_bpn,
                                    plot_baseline_subtractions,
                                    plot_single_well,
                                    plot_signal_patterns)


# -------------------------------------------------------------------
# 데이터 로드 및 준비
# -------------------------------------------------------------------

def main_function(outlier_naive_metric=1.65,mudt=True):
    """
    베이스라인 최적화 분석을 위한 데이터를 준비하는 메인 함수
    
    - 여러 알고리즘 결과 데이터셋을 로드하고 병합하여 분석 준비를 수행
    - MuDT 전처리 유무에 따라 다른 데이터셋을 불러옴
    
    매개변수:
    - outlier_naive_metric (float): 이상치 탐지를 위한 임계값 (기본값: 1.65)
    - mudt (bool): MuDT 전처리 적용 여부 (기본값: True)
        
    반환값:
    - tuple: (merged_data, filtered_data) - 병합된 전체 데이터셋과 필터링된 데이터셋
    """

    # variables used for merging dataframes
    
    ## Algorithm 1 결과 데이터의 변수 목록: 타사의 blackbox 알고리즘
    cfx_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 'original_rfu']
    
    ## Algorithm 2 결과 데이터의 변수 목록: legacy
    auto_baseline_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                             'basesub_resultwell', 'basesub_dataprocnum', 'basesub_rd_diff', 
                             'basesub_ivd_cdd_output', 'basesub_cff', 'basesub_scd_fit', 'basesub_r_p2', 
                             'basesub_efc', 'basesub_absd_orig', 'basesub_absd']
    
    ## Algorithm 3 결과 데이터의 변수 목록: 동료의 1차 개선된 신규 알고리즘
    strep_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                     'analysis_absd','analysis_rd_diff','analysis_scd_fit','analysis_efc']
    
    ## Algorithm 4 결과 데이터의 변수 목록: 동료의 2차 개선된 신규 알고리즘
    strep_plus1_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                           'analysis_rd_diff','new_jump_corrected_rfu','new_efc','new_baseline','new_baseline_model','new_absd']
    
    ## Algorithm 5 결과 데이터의 변수 목록: 동료의 3차 개선된 신규 알고리즘
    strep_plus2_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                          'preproc_rfu','analysis_absd','analysis_rd_diff','analysis_scd_fit',
                          'analysis_efc','final_ct','analysis_resultwell','analysis_dataprocnum']
    
    ## Control DSP 결과 데이터의 변수 목록: 현재 운영에 사용되는 서비스 알고리즘 (대조군)
    control_dsp_columns =['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                          'preproc_rfu','analysis_absd','analysis_rd_diff','analysis_scd_fit',
                          'analysis_efc','final_ct','analysis_resultwell','analysis_dataprocnum']
    
    ## 모든 데이터셋에 공통적으로 존재하는 변수 목록
    combo_key_columns = ['name', 'consumable', 'channel', 'temperature', 'well']
    
    # 데이터 경로 설정
    rootpath = os.getcwd()

    if 'Administrator' in rootpath:       
        datapath = 'C:/Users/Administrator/Desktop/projects/website/docs/data/baseline_optimization/GI-B-I'
    else:
        datapath = 'C:/Users/kmkim/Desktop/projects/website/docs/data/baseline_optimization/GI-B-I'
    if mudt:    
        ### with MuDT (신호 분리 전처리 기술) 
        raw_datapath = datapath + '/raw_data/mudt_raw_data.parquet'
        auto_datapath = datapath + '/auto_baseline_data/mudt_auto_baseline_data.parquet'
        cfx_datapath = datapath + '/cfx_data/mudt_cfx_data.parquet'
        strep_plus1_datapath = datapath + '/strep_plus1_data/mudt_strep_plus1_data.parquet'
        strep_plus2_datapath = datapath + '/strep_plus2_data/mudt_strep_plus2_data.parquet'
        ml_datapath = datapath + '/ml_data/mudt_ml_data.parquet' # 머신러닝 기반 신호 분리 알고리즘
    else:
        ### without MuDT (신호 분리 전처리 기술 미적용) 
        raw_datapath = datapath + '/raw_data/no_mudt_raw_data.parquet'
        auto_datapath = datapath + '/auto_baseline_data/no_mudt_auto_baseline_data.parquet'
        cfx_datapath = datapath + '/cfx_data/no_mudt_cfx_data.parquet'
        strep_plus1_datapath = datapath + '/strep_plus1_data/no_mudt_strep_plus1_data.parquet'
        strep_plus2_datapath = datapath + '/strep_plus2_data/no_mudt_strep_plus2_data.parquet'
        ml_datapath = datapath + '/ml_data/no_mudt_ml_data.parquet' # 머신러닝 기반 신호 분리 알고리즘


    # Read parquets 

    raw_data = pl.scan_parquet(raw_datapath).collect().to_pandas()
    cfx_data = pl.scan_parquet(cfx_datapath).collect().to_pandas()
    auto_baseline_data = pl.scan_parquet(auto_datapath).collect().to_pandas()
    strep_plus1_data = pl.scan_parquet(strep_plus1_datapath).collect().to_pandas()
    strep_plus2_data = pl.scan_parquet(strep_plus2_datapath).collect().to_pandas()
    ml_data = pl.scan_parquet(ml_datapath).collect().to_pandas()


    ## variable selection used for merging the dataframes
    cfx_df = cfx_data[['original_rfu_cfx', 'combo_key']]
    auto_baseline_df = auto_baseline_data[auto_baseline_data.columns[6:]]  # Adjust index as necessary
    strep_plus1_df = strep_plus1_data[strep_plus1_data.columns[6:]]  
    strep_plus2_df = strep_plus2_data[strep_plus2_data.columns[6:]]  
    ml_df = ml_data[['ml_baseline_fit','ml_analysis_absd','combo_key']]


    ## Merge dataframes

    merged_data = (ml_df
                    .merge(raw_data, on='combo_key')
                    .merge(auto_baseline_df, on='combo_key')
                    .merge(strep_plus1_df, on ='combo_key')
                    .merge(strep_plus2_df, on = 'combo_key')
                    .merge(cfx_df, on = 'combo_key')
                  )
    negative_data = merged_data[merged_data['final_ct'] < 0]

    
    
    ## Preprocess 
    ### Detect outliers naively
    columns_to_process = ['original_rfu'] #['analysis_absd_orig', 'original_rfu_cfx', 'basesub_absd_orig']
    groupby_columns = ['channel', 'temperature']
    
    for column in columns_to_process:
        #merged_data = process_column_for_outliers(merged_data, column, groupby_columns,detect_noise_naively_ywj1)
        #merged_data = process_column_for_outliers(merged_data, column, groupby_columns,detect_noise_naively_pbg)
        #merged_data = process_column_for_outliers(merged_data, column, groupby_columns,detect_noise_naively_kkm)
        merged_data = process_column_for_outliers(merged_data, column, groupby_columns,detect_noise_naively)
        
    ### Detect White Noise Signals
    #merged_data['autocorrelation'] = merged_data['original_rfu_cfx'].apply(lambda cell: autocorrelation(cell,3))
    #merged_data['noise_pvalue_metric'] = merged_data['original_rfu_cfx'].apply(lambda cell: test_white_noise(cell,3)[1])
    #merged_data['noise_pvalue_cutoff'] = merged_data.groupby(['channel','temperature'])['noise_pvalue_metric'].transform(lambda x: np.percentile(x, 95))          
    #merged_data['noise_pvalue_percentile'] = merged_data.groupby(['channel', 'temperature']).apply(get_column_percentiles, i_column_name='noise_pvalue_metric',include_groups=False).reset_index(level=[0,1], drop=True)
    
    ### linear slope
    if mudt:
        merged_data['linear_slope'] = merged_data['preproc_rfu'].apply(lambda x: compute_lm_slope(x)[0])
    else:
        merged_data['linear_slope'] = merged_data['original_rfu'].apply(lambda x: compute_lm_slope(x)[0])
    
    channels=merged_data['channel'].unique()
    temperatures=merged_data['temperature'].unique()
    plate_names=merged_data['name'].unique()
    well_names=merged_data['well'].unique()
    colors = {'Low':'blue','High':'red'}
    pcrd_name = merged_data['name'].unique()[0]
    channel_name = merged_data['channel'].unique()[0]
    temperature_name = merged_data['temperature'].unique()[0]
    
    filtered_data = merged_data.query("`name` == @pcrd_name & `channel` == @channel_name & `temperature` == @temperature_name & `final_ct` < 0 & `outlier_naive_metric_original_rfu` > @outlier_naive_metric")
    return (merged_data,filtered_data)
    
### Merge Dataframes

def load_and_prepare_parquet(i_file_path, i_selected_columns=None, i_combo_key_columns=None, i_rename_columns=None):
    '''
    load parquets and prepare dataframe for analyses
    
    Args:
    - i_file_path: a file_path where parquet files exist.
    - i_selected_columns: a list of the column names used for a data analysis
    - i_combo_key_columns: a list of the column names used for creating a combination key and merging several data frames
    - i_rename_columns: a list of the column names for renaming
    
    Returns:
    - o_df: a dataframe
    '''

    o_df = pl.scan_parquet(i_file_path).collect().to_pandas()
    if i_selected_columns:
        o_df = o_df[i_selected_columns].copy()
    if i_combo_key_columns:
        o_df['combo_key'] = o_df.apply(lambda x: ' '.join(str(x[col]) for col in i_combo_key_columns), axis=1)
    if i_rename_columns:
        o_df.rename(columns = i_rename_columns, inplace=True)
    
    return o_df


def get_column_percentiles(i_data, i_column_name):
    '''
    Calculate the percentile rank of each score in the specified column of a DataFrame.
    
    Args:
    - i_data: DataFrame containing the metric scores.
    - i_column_name: The name of the metric column whose scores' percentile ranks are to be calculated.
    
    Returns:
    - o_percentiles: A Series containing the percentile ranks of the scores in the specified column.
    '''
    def get_percentile(i_sorted_data, i_value):
        '''
        Calculate the percentile rank of a metric score relative to sorted scores.
        
        Args:
        - i_sorted_data: Sorted numpy array of scores.
        - i_value: The score whose percentile rank is to be calculated.
        
        Returns:
        - o_percent: The percentile rank of the score.
        '''
        if not i_sorted_data.size:  # Check if sorted data is empty
            return np.nan  # Return NaN for empty data
        count = np.sum(i_sorted_data <= i_value)
        o_percent = 100.0 * count / len(i_sorted_data)
        return np.round(o_percent, 1)
    
    non_empty_values = i_data[i_column_name].apply(lambda x: x if x else np.nan).dropna()
    i_sorted_data = np.sort(non_empty_values.values)
    o_percentiles = non_empty_values.apply(lambda x: get_percentile(i_sorted_data, x))
    
    return o_percentiles



def process_column_for_outliers(i_data, i_column, i_groupby_columns,i_function):
    """
    Process a given column for outlier detection and related metrics using detect_noise_naively().
    
    Args:
    - i_data: DataFrame to process.
    - i_column: The name of the column to process for outliers.
    - i_groupby_columns: Columns names to group by for percentile calculations.
    
    Returns:
    - o_result: Updated DataFrame with new outlier-related columns.
    """
    # Detect outliers and calculate outlierness metric
    i_data[f'outlier_naive_residuals_{i_column}'] = i_data[i_column].apply(lambda cell: i_function(cell)[1])
    i_data[f'outlier_naive_metric_{i_column}'] = i_data[i_column].apply(lambda cell: i_function(cell)[0])
    
    # Calculate 95th percentile cutoff for outlierness metric
    i_data[f'outlier_naive_cutoff_{i_column}'] = i_data.groupby(i_groupby_columns)[f'outlier_naive_metric_{i_column}'].transform(lambda x: np.percentile(x, 95))
    
    # Calculate metric percentiles within each group
    i_data[f'outlier_naive_metric_percentile_{i_column}'] = (i_data
                                                             .groupby(i_groupby_columns)
                                                             .apply(lambda grp: get_column_percentiles(grp, f'outlier_naive_metric_{i_column}'))#,include_groups=False)
                                                             .reset_index(level=i_groupby_columns, drop=True))
    
    o_result = i_data
    return o_result

def compute_lm_slope(i_signal):
    cycle_length = np.size(i_signal)
    cycle = range(1,cycle_length+1)
    i_signal_mean = np.mean(i_signal)
    cycle_mean = np.mean(cycle)
    slope = np.cov(i_signal, cycle)[0,1] / np.cov(i_signal, cycle)[0,0]
    intercept = i_signal_mean / -slope*cycle_mean
    return [slope, intercept]  # Return a list containing slope and intercept

def check_memory_status():
    import psutil # memory checking
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 ** 2)  # 메가바이트 단위로 변환
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage = mem_info.rss / (1024 ** 2)  # rss는 실제 메모리 사용량을 나타냄
    available_memory = memory.available / (1024 ** 2)  # 메가바이트 단위로 변환
    
    print(f"Memory used: {memory_usage:.2f} MB")
    print(f"Total memory: {total_memory:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")

def get_disk_usage(path="/"):
    import shutil
    usage = shutil.disk_usage(path)
    print(f"Total Disk Capacity: {usage.total / (1024**3):.2f} GB")
    print(f"Used Disk Space: {usage.used / (1024**3):.2f} GB")
    print(f"Free Disk Space: {usage.free / (1024**3):.2f} GB")

def get_package_details():
    import subprocess
    import sys
    
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.split('\n')
    
    package_dict = {}
    for line in lines[2:]:  # 첫 두 줄은 헤더 정보이므로 제외
        parts = line.split()
        if len(parts) >= 2:
            package_name = parts[0]
            version = parts[1]
            package_dict[package_name] = version
    return package_dict