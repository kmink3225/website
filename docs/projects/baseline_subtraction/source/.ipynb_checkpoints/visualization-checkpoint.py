# Config
import pydsptools.config.pda as pda
import pydsptools.biorad as biorad

# Analysis Preparation
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import chi2 # https://en.cppreference.com/w/cpp/numeric/random/chi_squared_distribution

# DSP Processing
import pydsp.run.worker
import pydsptools.biorad.parse as bioradparse

# PreProcessing
import pprint
import pyarrow as pa
import os
os.chdir('/home/jupyter-kmkim/dsp-research-strep-a/kkm')
import subprocess
from pathlib import Path
from source import signal_filter

# Visualization
import pydsptools.plot as dspplt
import plotly.express as px
import matplotlib.pyplot as plt

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

def plot_temp(i_data, i_pcrd, i_channel, i_well, i_color=None):
    filtered_data = i_data.query("`name` == @i_pcrd & `channel` == @i_channel & well == @i_well")
    '''
    this function plot a single well.

    args:
    - i_data 
    - i_color
    - i_pcrd
    - i_channel
    - i_i_temperature
    - i_well

    return:
    - result: a plt.plot for a single well
    '''
    # Green: fam_data 
    # Orange: hex_data 
    # Red: cRed_data
    # Blue: q670_data

    for index, row in filtered_data.iterrows():
        preproc_rfu = row['preproc_rfu'] 
        cycle = list(range(len(preproc_rfu))) 
        #color = i_color[row['temperature']]
        plt.plot(cycle, preproc_rfu, label=f'Line {index}', color = i_color, alpha= 0.2, ms=10)


def find_sub_extremes(i_signal):
    '''
    Find the maximum and minimum values within each cell in a DataFrame column,
    then find the overall max and min for the column. The output is used for setting the axis limits.
    Args:
        - i_signal: a list of the signal values of a column of a dataframe
    Returns:
        - o_result: a list of min and max of the signal values
    '''
    def process_cell(i_signal):
        '''
        Check if i_signal is list-like and not empty
        Args:
            - i_signal: a list of the signal values of a column of a dataframe
        Returns:
            - o_result: a list of min and max of the signal values
        '''
        
        if isinstance(i_signal, (list, np.ndarray, pd.Series)) and len(i_signal):
            o_result = pd.Series([np.max(i_signal), np.min(i_signal)], index=['max', 'min'])
        else:
            o_result = pd.Series([pd.NA, pd.NA], index=['max', 'min'])
        return o_result
    
    extremes_df = i_signal.apply(process_cell)
    extremes_df = extremes_df.dropna()
    
    if extremes_df.empty:
        return [None, None]
    
    overall_max = extremes_df['max'].max()
    overall_min = extremes_df['min'].min()
    o_result = [overall_max, overall_min]
    return o_result

def find_global_extremes(i_data):
    '''
    Find the global extremes for the input dataframe, 'i_data'.
    Args:
        - i_data: a dataframe
    Returns:
        - o_result: a list of min and max of the signal values
    '''
    # if i_data = pd.dataFrame(),
    if isinstance(i_data, pd.DataFrame):
        extremes_results = {column: find_sub_extremes(i_data[column]) for column in i_data.columns}
        overall_max = float('-inf')
        overall_min = float('inf') 
    
        for key in extremes_results.keys():
            if key in extremes_results:
                column_max, column_min = extremes_results[key]
                overall_max = max(overall_max, column_max)
                overall_min = min(overall_min, column_min)
    # if i_data = pd.Series(),
    elif isinstance(i_data, pd.Series):
        overall_max, overall_min = find_sub_extremes(i_data)
    else:
        return [None, None]
        
    o_result = [overall_max,overall_min]
    return o_result

def get_comparison_metrics(i_grouped_data,i_columns):
    '''
    Compute absolute_means_sum and squared_means_sum for the columns ['analysis_absd_orig', 'original_rfu_cfx', 'basesub_absd_orig', 'analysis_absd_orig_strep']
    
    Args:
        i_grouped_data: a dataframe that is grouped of the 'filtered_data' using groupby() in pandas
        i_columns: a list of columns to calculate metrics
    Returns:
        o_result: a series of metrics
    '''
    metrics = {}
    
    for column in i_columns:
        # Filtering out empty lists and concatenating non-empty lists into a numpy array
        non_empty_lists = i_grouped_data[column].apply(lambda x: len(x) > 0)
        length = non_empty_lists.sum()
        absolute_means_sum = i_grouped_data[column][non_empty_lists].apply(lambda x: np.abs(x).mean()).sum()
        squared_means_sum = i_grouped_data[column][non_empty_lists].apply(lambda x: (x**2).mean()).sum()
        metric_array = np.concatenate(i_grouped_data[column][non_empty_lists].to_numpy())
        
        if metric_array.size > 0:
            column_metrics = {
                'length': length,
                'absolute_sum': round(absolute_means_sum, 2),  
                'squared_sum': round(squared_means_sum, 2),
                'mse': round(squared_means_sum/length, 2),
                'mae': round(absolute_means_sum/length, 2)
            }
        else:
            column_metrics = {
                'length': 0,
                'absolute_sum': 0, 
                'squared_sum': 0,
                'mse': 0,
                'mae': 0
            }
        metrics[column] = column_metrics
        o_result = pd.Series(metrics)
    return o_result

def compute_bpn(i_signal, i_reference_value):
    '''
    compute BPN (Background Point Normalization) to transform a signal points to the reference value.
    
    Args:
        i_data: a list of rfu signal points
        i_reference_value: a reference value used for transforming the signal to the reference value
        colors: customizing colors
    
    Returns:
        Output: 6 Pannels of plt.subplots()
    '''
    if i_signal.size == 0:  # Check if the signal is empty
        return [False, 0, []]  # Return default values for empty signals    
    mean = np.mean(i_signal)
    std = np.std(i_signal)
    after_bpn = [(point / mean) * i_reference_value for point in i_signal] # a normalized signal with the reference value = 100
    o_result = after_bpn
    return(o_result)

def plot_baseline_subtractions(i_data, i_pcrd, i_channel, i_temperature, colors=None, mudt = False):
    '''
    Plot the results, the baseline-subtracted data by the baseline-fitting algorithms to intuitively compare their overall performance.
    
    Args:
        i_data: a dataframe, a result of merging the DSP execution result of raw RFU and the RFU baseline-subtracted by CFX manager, by the auto-baseline module, by the strep assay, and by the newly-developed algorithms.
        i_pcrd: pcrd_name
        i_channel: channel_name
        i_temperature: temperature_name
        colors: customizing colors
    
    Returns:
        Output: 6 Pannels of plt.subplots()
    '''
    i_data = i_data.query("`name`==@i_pcrd & `channel` == @i_channel & `temperature`==@i_temperature").copy()
    
    if mudt:
        titles = ['[After BPN] Raw RFU', '[After BPN] Preproc RFU', '[DSP] Original ABSD', 
                  '[Auto] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Strep+2] Baseline-Subtracted RFU']
        data_keys = ['original_rfu_after_bpn','preproc_rfu_after_bpn', 'analysis_absd_orig', 
                     'basesub_absd_orig', 'strep_plus1_analysis_absd','strep_plus2_analysis_absd']
    else:
        titles = ['[After BPN] Raw RFU', '[CFX] Baseline-Subtracted RFU', '[DSP] Original ABSD', 
                  '[Auto] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Strep+2] Baseline-Subtracted RFU']
        data_keys = ['original_rfu_after_bpn', 'original_rfu_cfx', 'analysis_absd_orig', 
                     'basesub_absd_orig','strep_plus1_analysis_absd','strep_plus2_analysis_absd']
    
    original_rfu_min_mean=i_data['original_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'original_rfu_after_bpn'] = i_data['original_rfu'].apply(lambda x: compute_bpn(x, original_rfu_min_mean))
    
    preproc_rfu_min_mean=i_data['preproc_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'preproc_rfu_after_bpn'] = i_data['preproc_rfu'].apply(lambda x: compute_bpn(x, preproc_rfu_min_mean))
        
    array_columns = [col for col in i_data.columns if i_data[col].apply(lambda x: isinstance(x, np.ndarray)).any()]
    rfu = i_data['original_rfu'].values[0]
    cycle_length = len(rfu)
       
    for column in array_columns:
        i_data[column] = i_data[column].apply(lambda x: np.zeros(cycle_length) if x is None or x.size == 0 else x)

    
    baseline_data = i_data[['analysis_absd_orig','basesub_absd_orig', 'original_rfu_cfx','strep_plus1_analysis_absd','strep_plus2_analysis_absd']]
    
    original_rfu_after_bpn_limits = find_global_extremes(i_data['original_rfu_after_bpn'])
    preproc_rfu_after_bpn_limits = find_global_extremes(i_data['preproc_rfu_after_bpn'])
    limits = find_global_extremes(baseline_data)
    
    grouping_columns = ['analysis_absd_orig', 'original_rfu_cfx', 'basesub_absd_orig', 'strep_plus1_analysis_absd', 'strep_plus2_analysis_absd']
    error_metrics_df = (i_data
                        .groupby(['name', 'channel', 'temperature'])
                        .apply(lambda x: get_comparison_metrics(x, grouping_columns),include_groups=False)
                        .reset_index(drop=True))
    error_metrics_dict = error_metrics_df.iloc[0]
            
    metric_min=round(i_data['outlier_naive_metric_original_rfu'].min(),2)
    metric_max=round(i_data['outlier_naive_metric_original_rfu'].max(),2)
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) #(15, 10)
    
    for (i, j), ax, title, key in zip(np.ndindex(axs.shape), axs.flat, titles, data_keys):
        if key:  
            rfu_values = []
            for index, row in i_data.iterrows():
                rfu = row[key]
                rfu_values.extend(rfu)
                cycle = list(range(len(rfu)))
               
                if (i,j) == (0,0):
                    ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(original_rfu_min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,1):
                    if mudt:
                        ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(preproc_rfu_min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                    else:
                        ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['original_rfu_cfx']['length'],2)}\nMAE: {round(error_metrics_dict['original_rfu_cfx']['mae'],2)}\nMSE: {round(error_metrics_dict['original_rfu_cfx']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i,j) == (0,2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['analysis_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['analysis_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['analysis_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i,j) == (1,0):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['basesub_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['basesub_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['basesub_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i,j) == (1,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus1_analysis_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus1_analysis_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus1_analysis_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i, j) == (1, 2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus2_analysis_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus2_analysis_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus2_analysis_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                ax.plot(cycle, rfu, alpha=0.5)
                ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
                ax.set_title(title)
            if (i,j) == (0,0):
                ax.axhline(y=original_rfu_min_mean,color='black',linestyle='dotted',linewidth=2)
                if rfu_values:
                    ax.set_ylim([min(original_rfu_after_bpn_limits)*0.98,max(original_rfu_after_bpn_limits)*1.02])
            elif (i,j) == (0,1):
                if mudt:
                    ax.axhline(y=preproc_rfu_min_mean,color='black',linestyle='dotted',linewidth=2)
                    ax.set_ylim([min(preproc_rfu_after_bpn_limits)*1.02,max(preproc_rfu_after_bpn_limits)*0.98])
                else:
                    ax.set_ylim(min(limits)*1.2,max(limits)*1.5)    
            else:
                ax.set_ylim(min(limits)*1.2,max(limits)*1.5)        
        else:  
            ax.set_title(title)
    pydsp_version = get_package_details().get('pydsp')
    fig.suptitle(f'PCRD: {i_pcrd}\npydsp Version: {pydsp_version}, Channel: {i_channel}, Temperature: {i_temperature}, The Number of Signals: {i_data.shape[0]}', 
                 ha = 'left', x=0.02, fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_single_well(i_data, i_pcrd, i_channel, i_temperature, i_well, mudt = True, colors=None):
    '''
    Plot the results, the baseline-subtracted data by the baseline-fitting algorithms to intuitively compare their overall performance.
    
    Args:
        i_data: a dataframe filtering merged_data by some conditions:
            ex) `merged_data.query("`name` == @pcrd_name & `channel` == @channel_name & `temperature` == @temperature_name & `final_ct` < 0 & `outlier_naive_metric_original_rfu` > 2")`
        i_pcrd: pcrd_name
        i_channel: channel_name
        i_temperature: temperature_name
        colors: customizing colors
    
    Returns:
        Output: 6 Pannels of plt.subplots()
    '''
    i_data = i_data.query("`name`==@i_pcrd & `channel` == @i_channel & `temperature`==@i_temperature & `well` == @i_well").copy()
    
    if mudt:
        raw_rfu = 'preproc_rfu'
    else:
        raw_rfu = 'original_rfu'
    
    columns = [ # (new_column, from, to) 
    ('dsp_correction_fit', raw_rfu, 'analysis_absd_orig'), # create DSP Correction + DSP Fit data 
    ('basesub_correction_fit', raw_rfu, 'basesub_absd_orig'), # create Auto Correction + Auto Fit data 
    ('cfx_correction_fit', raw_rfu, 'original_rfu_cfx'), # create CFX Correction + CFX Fit data
    ('strep_plus1_correction_fit', raw_rfu , 'strep_plus1_analysis_absd'), # create strep+1 Correction + strep+1 Fit data
    ('strep_plus2_correction_fit', raw_rfu, 'strep_plus2_analysis_absd'), # create strep+2 Correction + strep+2 Fit data
    
    ('dsp_corrected_rfu', raw_rfu, 'analysis_rd_diff'), # create DSP Correction data
    ('basesub_corrected_rfu', raw_rfu, 'basesub_rd_diff'), # create Auto Correction data  
    # cannot create CFX Correction: it is 'black box'
    ('strep_plus1_corrected_rfu', raw_rfu, 'strep_plus1_analysis_rd_diff'), # create strep_plus1 Correction data  
    ('strep_plus2_corrected_rfu',raw_rfu, 'strep_plus2_analysis_rd_diff'), # create strep+2 Correction data    
    ]
    
    
    raw_scale_columns = [raw_rfu,'dsp_correction_fit', 'basesub_correction_fit','cfx_correction_fit',
                        'strep_plus1_correction_fit','strep_plus2_correction_fit']
    correction_scale_columns = [
        raw_rfu,'dsp_corrected_rfu', 'basesub_corrected_rfu', 'strep_plus1_corrected_rfu', 'strep_plus2_corrected_rfu'#,
        #'analysis_scd_fit', 'basesub_scd_fit', 'strep_plus1_baseline_fit', 'strep_plus2_analysis_scd_fit'
    ]
                            
    subtract_scale_columns = ['basesub_absd_orig','original_rfu_cfx', 'analysis_absd_orig', 
                              'strep_plus1_analysis_absd','strep_plus2_analysis_absd']
    
    titles = ['[DSP] Algorithm Performance', '[Auto] Algorithm Performance', '[CFX] Algorithm Performance', '[Strep+1] Algorithm Performance', '[Strep+2] Algorithm Performance',
          '[DSP] Fitting Performance', '[Auto] Fitting Performance', '[CFX] Fitting Performance', '[Strep+1] Fitting Performance', '[Strep+2] Fitting Performance',
          '[DSP] Baseline-Subtracted RFU', '[Auto] Baseline-Subtracted RFU', '[CFX] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Strep+2] Baseline-Subtracted RFU']
    
    preprocessing_columns = [
    ('dsp_correction_fit', raw_rfu),
    ('basesub_correction_fit', raw_rfu),
    ('cfx_correction_fit', raw_rfu),
    ('strep_plus1_correction_fit', raw_rfu),
    ('strep_plus2_correction_fit', raw_rfu)]
    
    correction_columns = [
    ('dsp_corrected_rfu', raw_rfu, 'analysis_scd_fit'), 
    ('basesub_corrected_rfu', raw_rfu, 'basesub_scd_fit'),
    (raw_rfu, raw_rfu, 'cfx_correction_fit'),
    ('strep_plus1_corrected_rfu', raw_rfu, 'strep_plus1_baseline_fit'),
    ('strep_plus2_corrected_rfu', raw_rfu, 'strep_plus2_analysis_scd_fit')]
    
    subtracted_columns = [    
    ('analysis_absd_orig'),
    ('basesub_absd_orig'),
    ('original_rfu_cfx'),
    ('strep_plus1_analysis_absd'),
    ('strep_plus2_analysis_absd')]
    
    if mudt:
        columns = [col for col in columns if 'cfx' not in col[0].lower()]
        raw_scale_columns= [col for col in raw_scale_columns if 'cfx' not in col.lower()]
        correction_scale_columns = [col for col in correction_scale_columns if 'cfx' not in col[0].lower()]
        subtract_scale_columns = [col for col in subtract_scale_columns if 'cfx' not in col[0].lower()]
        titles= [col for col in titles if 'cfx' not in col.lower()]
        preprocessing_columns= [col for col in preprocessing_columns if not any('cfx' in item.lower() for item in col)]
        correction_columns=  [col for col in correction_columns if not any('cfx' in item.lower() for item in col)]
        subtracted_columns = [col for col in subtracted_columns if 'cfx' not in col.lower()]
        fig, axs = plt.subplots(3, 4, figsize=(18, 12)) #(15, 10)
    else:
        fig, axs = plt.subplots(3, 5, figsize=(18, 12)) #(15, 10)
    
    
    rfu = i_data[raw_rfu].values[0]
    cycle_length = len(rfu)
    cycle = list(range(cycle_length))       
    
    array_columns = [col for col in i_data.columns if i_data[col].apply(lambda x: isinstance(x, np.ndarray)).any()]
    for column in array_columns:
        i_data[column] = i_data[column].apply(lambda x: np.zeros(cycle_length) if x is None or x.size == 0 else x)
    
    for new_column, column_a, column_b in columns:
        i_data[new_column] = list(map(lambda x, y: [a - b for a, b in zip(x, y)], i_data[column_a], i_data[column_b]))
    
    raw_scale_data = i_data[raw_scale_columns]
    correction_scale_data = i_data[correction_scale_columns]
    subtract_scale_data = i_data[subtract_scale_columns]
    
    raw_limits = find_global_extremes(raw_scale_data)
    correction_limits = find_global_extremes(correction_scale_data)
    subtraction_limits = find_global_extremes(subtract_scale_data)
    
    rfu_values = [] 
    correct_alpha = 0.5
    fit_alpha = 0.5
    
    def set_ylim(ax, limits, index):
        if max(limits) < 0:
            max_limit_ratio = 0.95
        else:
            max_limit_ratio = 1.05
        if min(limits) < 0: 
            min_limit_ratio = 1.05
        else:
            min_limit_ratio = 0.95
        
        min_limit = np.sign(min(limits)) * np.abs(min(limits)) * min_limit_ratio
        max_limit = np.sign(max(limits)) * np.abs(max(limits)) * max_limit_ratio
        ax.set_ylim([min_limit, max_limit])
    
    for ((i, j), ax, title) in zip(np.ndindex(axs.shape), axs.flat, titles):
        if i == 0:
            ax.plot(cycle, i_data[preprocessing_columns[j][0]].values[0], 'red', alpha=0.5, label = "Correction & Fit")
            ax.plot(cycle, i_data[preprocessing_columns[j][1]].values[0], 'k-', alpha=0.5, label = "Raw Data")
    
        elif i == 1:
            ax.plot(cycle, i_data[correction_columns[j][0]].values[0], 'b--', alpha=1, label = "Correction")
            ax.plot(cycle, i_data[correction_columns[j][1]].values[0], 'k-', alpha=0.5, label = "Raw Data")
            ax.plot(cycle, i_data[correction_columns[j][2]].values[0], 'red', alpha=0.5, label = "Fitted Data" )
        else:
            ax.plot(cycle, i_data[subtracted_columns[j]].values[0], 'k-', alpha=0.5, label = "Subtracted Data")
            ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
    
        if i == 0:
            set_ylim(ax, raw_limits, 0)
        elif i == 1:
            set_ylim(ax, correction_limits, 1)
        else:
            set_ylim(ax, subtraction_limits, 2)
    
        ax.set_title(title)
        ax.legend()
    pydsp_version = get_package_details().get('pydsp')
    fig.suptitle(f'PCRD: {i_pcrd}\npydsp Version: {pydsp_version}, Channel: {i_channel}, Temperature: {i_temperature}, Well: {i_well}', 
                     ha = 'left', x=0.02, fontsize=15)
    
    plt.tight_layout()
    plt.show()


def plot_signal_patterns(i_data,i_channel,i_temperature, mudt):

    
    '''
    Plot the results, the baseline-subtracted data by the baseline-fitting algorithms to intuitively compare their overall performance.
    
    Args:
        i_data: a dataframe, a result of merging the DSP execution result of raw RFU and the RFU baseline-subtracted by CFX manager, by the auto-baseline module, by the strep assay, and by the newly-developed algorithms.
        i_channel: channel_name
        i_temperature: temperature_name
        colors: customizing colors
    
    Returns:
        Output: 6 Pannels of plt.subplots()
    '''
    
    i_data = i_data.query("`channel` == @i_channel & `temperature`==@i_temperature").copy()
    
    if mudt:
        titles = ['[After BPN] Raw RFU', '[After BPN] Preproc RFU', '[DSP] Original ABSD', 
                  '[Auto] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Strep+2] Baseline-Subtracted RFU']
        data_keys = ['original_rfu_after_bpn','preproc_rfu_after_bpn', 'analysis_absd_orig', 
                     'basesub_absd_orig', 'strep_plus1_analysis_absd','strep_plus2_analysis_absd']
        baseline_columns = ['analysis_absd_orig','basesub_absd_orig','strep_plus1_analysis_absd','strep_plus2_analysis_absd']
    else:
        titles = ['[After BPN] Raw RFU', '[CFX] Baseline-Subtracted RFU', '[DSP] Original ABSD', 
                  '[Auto] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Strep+2] Baseline-Subtracted RFU']
        data_keys = ['original_rfu_after_bpn', 'original_rfu_cfx', 'analysis_absd_orig', 
                     'basesub_absd_orig', 'strep_plus1_analysis_absd','strep_plus2_analysis_absd']
        baseline_columns = ['analysis_absd_orig','basesub_absd_orig', 'original_rfu_cfx','strep_plus1_analysis_absd','strep_plus2_analysis_absd']
    
    original_rfu_min_mean=i_data['original_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'original_rfu_after_bpn'] = i_data['original_rfu'].apply(lambda x: compute_bpn(x, original_rfu_min_mean))
    
    preproc_rfu_min_mean=i_data['preproc_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'preproc_rfu_after_bpn'] = i_data['preproc_rfu'].apply(lambda x: compute_bpn(x, preproc_rfu_min_mean))
        
    array_columns = [col for col in i_data.columns if i_data[col].apply(lambda x: isinstance(x, np.ndarray)).any()]
    rfu = i_data['original_rfu'].values[0]
    cycle_length = len(rfu)
       
    for column in array_columns:
        i_data[column] = i_data[column].apply(lambda x: np.zeros(cycle_length) if x is None or x.size == 0 else x)
    
    
    baseline_data = i_data[baseline_columns]
    
    original_rfu_after_bpn_limits = find_global_extremes(i_data['original_rfu_after_bpn'])
    preproc_rfu_after_bpn_limits = find_global_extremes(i_data['preproc_rfu_after_bpn'])
    limits = find_global_extremes(baseline_data)
    
    grouping_columns = ['analysis_absd_orig', 'original_rfu_cfx', 'basesub_absd_orig', 'strep_plus1_analysis_absd', 'strep_plus2_analysis_absd']
    error_metrics_df = (i_data
                        .groupby(['name', 'channel', 'temperature'])
                        .apply(lambda x: get_comparison_metrics(x, grouping_columns),include_groups=False)
                        .reset_index(drop=True))
    error_metrics_dict = error_metrics_df.iloc[0]
            
    metric_min=round(i_data['outlier_naive_metric_original_rfu'].min(),2)
    metric_max=round(i_data['outlier_naive_metric_original_rfu'].max(),2)
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) #(15, 10)
    
    for (i, j), ax, title, key in zip(np.ndindex(axs.shape), axs.flat, titles, data_keys):
        if key:  
            rfu_values = []
            for index, row in i_data.iterrows():
                rfu = row[key]
                rfu_values.extend(rfu)
                cycle = list(range(len(rfu)))
               
                if (i,j) == (0,0):
                    ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(original_rfu_min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,1):
                    if mudt:
                        ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(preproc_rfu_min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                    else:
                        ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['original_rfu_cfx']['length'],2)}\nMAE: {round(error_metrics_dict['original_rfu_cfx']['mae'],2)}\nMSE: {round(error_metrics_dict['original_rfu_cfx']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i,j) == (0,2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['analysis_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['analysis_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['analysis_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i,j) == (1,0):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['basesub_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['basesub_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['basesub_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i,j) == (1,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus1_analysis_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus1_analysis_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus1_analysis_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                elif (i, j) == (1, 2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus2_analysis_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus2_analysis_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus2_analysis_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
                ax.plot(cycle, rfu, alpha=0.5)
                ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
                ax.set_title(title)
            if (i,j) == (0,0):
                ax.axhline(y=original_rfu_min_mean,color='black',linestyle='dotted',linewidth=2)
                if rfu_values:
                    ax.set_ylim([min(original_rfu_after_bpn_limits)*0.98,max(original_rfu_after_bpn_limits)*1.02])
            elif (i,j) == (0,1):
                if mudt:
                    ax.axhline(y=preproc_rfu_min_mean,color='black',linestyle='dotted',linewidth=2)
                    ax.set_ylim([min(preproc_rfu_after_bpn_limits)*1.02,max(preproc_rfu_after_bpn_limits)*0.98])
                else:
                    ax.set_ylim(min(limits)*1.2,max(limits)*1.5)    
            else:
                ax.set_ylim(min(limits)*1.2,max(limits)*1.5)        
        else:  
            ax.set_title(title)
    pydsp_version = get_package_details().get('pydsp')
    fig.suptitle(f'pydsp Version: {pydsp_version}, Channel: {i_channel}, Temperature: {i_temperature}, The Number of Signals: {i_data.shape[0]}', 
                 ha = 'left', x=0.02, fontsize=15)
    plt.tight_layout()
    plt.show()