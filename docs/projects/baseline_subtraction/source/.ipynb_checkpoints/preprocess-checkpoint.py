# Config
import pydsptools.config.pda as pda
import pydsptools.biorad as biorad

# Analysis Preparation
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import chi2 # https://en.cppreference.com/w/cpp/numeric/random/chi_squared_distribution

# DSP Execution
import pydsp.run.worker
import pydsptools.biorad.parse as bioradparse


# PreProcessing
import pprint
import pyarrow as pa
import os
os.chdir('/home/jupyter-kmkim/dsp-research-strep-a/kkm')
import subprocess
from pathlib import Path

from source.signal_filter import (detect_noise_naively,
                                    detect_noise_naively_ver2,
                                    detect_noise_naively_ywj1,
                                    detect_noise_naively_pbg,
                                    detect_noise_naively_kkm,
                                    compute_autocorrelation,
                                    test_white_noise)

# Visualization
import pydsptools.plot as dspplt
import plotly.express as px
import matplotlib.pyplot as plt
from source.visualization import (find_sub_extremes,
                                    find_global_extremes,
                                    get_comparison_metrics,
                                    compute_bpn,
                                    plot_baseline_subtractions,
                                    plot_single_well,
                                    plot_signal_patterns)

# pandas 출력 옵션 설정
#pd.set_option('display.max_rows', None)  # 모든 행 출력
#pd.set_option('display.max_columns', None)  # 모든 열 출력
#pd.set_option('display.width', 1000)  # 셀 너비 설정
#pd.set_option('display.max_colwidth', None)  # 열 내용 전체 출력
# the Running Script

def main_function(outlier_naive_metric=1.65,mudt=True):
    ## variables used for merging dataframes
    cfx_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 'original_rfu']
    auto_baseline_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                             'basesub_resultwell', 'basesub_dataprocnum', 'basesub_rd_diff', 
                             'basesub_ivd_cdd_output', 'basesub_cff', 'basesub_scd_fit', 'basesub_r_p2', 
                             'basesub_efc', 'basesub_absd_orig', 'basesub_absd']
    strep_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                     'analysis_absd','analysis_rd_diff','analysis_scd_fit','analysis_efc']
    strep_plus1_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                           'analysis_rd_diff','new_jump_corrected_rfu','new_efc','new_baseline','new_baseline_model','new_absd']
    strep_plus2_columns = ['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                          'preproc_rfu','analysis_absd','analysis_rd_diff','analysis_scd_fit',
                          'analysis_efc','final_ct','analysis_resultwell','analysis_dataprocnum']
    control_dsp_columns =['name', 'steps', 'consumable', 'well', 'channel', 'temperature', 
                          'preproc_rfu','analysis_absd','analysis_rd_diff','analysis_scd_fit',
                          'analysis_efc','final_ct','analysis_resultwell','analysis_dataprocnum']
    combo_key_columns = ['name', 'consumable', 'channel', 'temperature', 'well']
    
    ## datapaths
    if mudt:    
        ### with MuDT (To Be Organized)
        raw_datapath = './data/GI-B-I/raw_data/computed/dsp2_generic_config_MuDT/dsp/*.parquet'
        auto_datapath = './data/GI-B-I/strep_plus2/computed/dsp2_strep_plus2_config_MuDT/basesub/*.parquet'
        cfx_datapath = './data/cfx-baseline-subtracted/computed/example1/config__dsp2_orig/dsp/*.parquet'
        strep_plus1_datapath = './data/GI-B-I/strep_plus1/dsp2_strep-plus1_config_MuDT.parquet'
        strep_plus2_datapath = './data/GI-B-I/strep_plus2/computed/dsp2_strep_plus2_config_MuDT/dsp/*.parquet'
    else:
        ### without MuDT (To Be Organized)
        raw_datapath = './data/GI-B-I/raw_data/computed/dsp2_generic_config_no-MuDT/dsp/*.parquet'
        auto_datapath = './data/GI-B-I/strep_plus2/computed/dsp2_strep_plus2_config_no-MuDT/basesub/*.parquet'
        cfx_datapath = './data/cfx-baseline-subtracted/computed/example1/config__dsp2_orig/dsp/*.parquet'
        strep_plus1_datapath = './data/GI-B-I/strep_plus1/dsp2_strep-plus1_config_no-MuDT.parquet'
        strep_plus2_datapath = './data/GI-B-I/strep_plus2/computed/dsp2_strep_plus2_config_no-MuDT/dsp/*.parquet'
    
    
    ## Read parquets
    raw_data = load_and_prepare_parquet(raw_datapath, i_combo_key_columns=combo_key_columns)
    cfx_data = load_and_prepare_parquet(cfx_datapath, cfx_columns, combo_key_columns, {'original_rfu': 'original_rfu_cfx'})
    auto_baseline_data = load_and_prepare_parquet(auto_datapath, auto_baseline_columns, combo_key_columns)
    strep_plus1_data = load_and_prepare_parquet(strep_plus1_datapath, strep_plus1_columns, combo_key_columns,
                                        {'new_jump_corrected_rfu': 'strep_plus1_corrected_rfu','new_efc': 'strep_plus1_efc','new_baseline':'strep_plus1_baseline_fit',
                                         'new_baseline_model':'strep_plus1_baseline_model','new_absd':'strep_plus1_analysis_absd','analysis_rd_diff':'strep_plus1_analysis_rd_diff'})
    strep_plus2_data = load_and_prepare_parquet(strep_plus2_datapath, control_dsp_columns, combo_key_columns,
                                          {'preproc_rfu': 'strep_plus2_preproc_rfu', 'analysis_absd': 'strep_plus2_analysis_absd',
                                          'analysis_rd_diff': 'strep_plus2_analysis_rd_diff','analysis_scd_fit':'strep_plus2_analysis_scd_fit', 
                                          'analysis_efc':'strep_plus2_analysis_efc', 'final_ct':'strep_plus2_final_ct', 
                                          'analysis_resultwell':'strep_plus2_analysis_resultwell', 'analysis_dataprocnum': 'strep_plus2_analysis_dataprocnum'})
    
    ## variable selection used for merging the dataframes
    cfx_df = cfx_data[['original_rfu_cfx', 'combo_key']]
    auto_baseline_df = auto_baseline_data[auto_baseline_columns[6:] + ['combo_key']]  # Adjust index as necessary
    #strep_df = strep_data[strep_data.columns[6:]]  
    strep_plus1_df = strep_plus1_data[strep_plus1_data.columns[6:]]  
    strep_plus2_df = strep_plus2_data[strep_plus2_data.columns[6:]]  
    
    ## Merge dataframes
    
    merged_data = (cfx_df
                   .merge(raw_data, on='combo_key')
                   .merge(auto_baseline_df, on='combo_key')
                   .merge(strep_plus1_df, on ='combo_key')
                   .merge(strep_plus2_df, on = 'combo_key')
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
                                                             .apply(lambda grp: get_column_percentiles(grp, f'outlier_naive_metric_{i_column}'),include_groups=False)
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