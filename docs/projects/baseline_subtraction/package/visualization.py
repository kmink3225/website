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
from package import signal_filter

# Visualization
import pydsptools.plot as dspplt
import plotly.express as px
import matplotlib.pyplot as plt


def plot_single_well(i_data, i_pcrd, i_channel, i_well, i_color=None):
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

def plot_baseline_subtractions(i_data, i_pcrd, i_channel, i_temperature, colors=None):
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
    
    min_mean=i_data['original_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'after_bpn_rfu'] = i_data['original_rfu'].apply(lambda x: compute_bpn(x, min_mean))
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) #(15, 10)
    
    baseline_data = i_data[['analysis_absd_orig','basesub_absd_orig', 'original_rfu_cfx','strep_analysis_absd_orig','strep_plus1_absd']]
    after_bpn_limits = find_global_extremes(i_data['after_bpn_rfu'])
    limits = find_global_extremes(baseline_data)
        
    titles = ['[After BPN] RFU', '[DSP] Original ABSD', '[Auto] Baseline-Subtracted RFU', '[CFX] Baseline-Subtracted RFU', 
            '[Strep] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Empty] Empty Pannel']
    data_keys = ['after_bpn_rfu', 'analysis_absd_orig', 'basesub_absd_orig', 'original_rfu_cfx', 
                     'strep_analysis_absd_orig', 'strep_plus1_absd', None, 'outlier_naive_metric_original_rfu']
    grouping_columns = ['analysis_absd_orig', 'original_rfu_cfx', 'basesub_absd_orig', 'strep_analysis_absd_orig', 'strep_plus1_absd']
    error_metrics_df = (i_data
                        .groupby(['name', 'channel', 'temperature'])
                        .apply(lambda x: get_comparison_metrics(x, grouping_columns),include_groups=False)
                        .reset_index(drop=True))
    error_metrics_dict = error_metrics_df.iloc[0]
            
    metric_min=round(i_data['outlier_naive_metric_original_rfu'].min(),2)
    metric_max=round(i_data['outlier_naive_metric_original_rfu'].max(),2)
    
    for (i, j), ax, title, key in zip(np.ndindex(axs.shape), axs.flat, titles, data_keys):
        if key:  
            rfu_values = []
            for index, row in i_data.iterrows():
                rfu = row[key]
                rfu_values.extend(rfu)
                cycle = list(range(len(rfu)))
               #color = colors[row['temperature']]
                if (i,j) == (0,0):
                    ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['analysis_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['analysis_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['analysis_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['basesub_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['basesub_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['basesub_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (1,0):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['original_rfu_cfx']['length'],2)}\nMAE: {round(error_metrics_dict['original_rfu_cfx']['mae'],2)}\nMSE: {round(error_metrics_dict['original_rfu_cfx']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (1,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_analysis_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['strep_analysis_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_analysis_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i, j) == (1, 2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus1_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus1_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus1_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                ax.plot(cycle, rfu, alpha=0.5)
                ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
                ax.set_title(title)
            if (i,j) == (0,0):
                ax.axhline(y=min_mean,color='black',linestyle='dotted',linewidth=2)
                if rfu_values:
                    ax.set_ylim([min(after_bpn_limits)*0.98,max(after_bpn_limits)*1.02])
            else:
                    ax.set_ylim(min(limits)*1.2,max(limits)*1.5)        
        else:  
            ax.set_title(title)
    
    fig.suptitle(f'PCRD: {i_pcrd}\nChannel: {i_channel}, Temperature: {i_temperature}, The Number of Signals: {i_data.shape[0]}', 
                 ha = 'left', x=0.02, fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_signal_patterns(i_data, i_pcrd, i_channel, i_temperature, colors=None):
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
    
    min_mean=i_data['original_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'after_bpn_rfu'] = i_data['original_rfu'].apply(lambda x: compute_bpn(x, min_mean))
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) #(15, 10)
    
    baseline_data = i_data[['analysis_absd_orig','basesub_absd_orig', 'original_rfu_cfx','strep_analysis_absd_orig','strep_plus1_absd']]
    after_bpn_limits = find_global_extremes(i_data['after_bpn_rfu'])
    limits = find_global_extremes(baseline_data)
        
    titles = ['[After BPN] RFU', '[DSP] Original ABSD', '[Auto] Baseline-Subtracted RFU', '[CFX] Baseline-Subtracted RFU', 
            '[Strep] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU', '[Empty] Empty Pannel']
    data_keys = ['after_bpn_rfu', 'analysis_absd_orig', 'basesub_absd_orig', 'original_rfu_cfx', 
                     'strep_analysis_absd_orig', 'strep_plus1_absd', None, 'outlier_naive_metric_original_rfu']
    grouping_columns = ['analysis_absd_orig', 'original_rfu_cfx', 'basesub_absd_orig', 'strep_analysis_absd_orig', 'strep_plus1_absd']
    error_metrics_df = (i_data
                        .groupby(['name', 'channel', 'temperature'])
                        .apply(lambda x: get_comparison_metrics(x, grouping_columns),include_groups=False)
                        .reset_index(drop=True))
    error_metrics_dict = error_metrics_df.iloc[0]
            
    metric_min=round(i_data['outlier_naive_metric_original_rfu'].min(),2)
    metric_max=round(i_data['outlier_naive_metric_original_rfu'].max(),2)
    
    for (i, j), ax, title, key in zip(np.ndindex(axs.shape), axs.flat, titles, data_keys):
        if key:  
            rfu_values = []
            for index, row in i_data.iterrows():
                rfu = row[key]
                rfu_values.extend(rfu)
                cycle = list(range(len(rfu)))
               #color = colors[row['temperature']]
                if (i,j) == (0,0):
                    ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['analysis_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['analysis_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['analysis_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['basesub_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['basesub_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['basesub_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (1,0):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['original_rfu_cfx']['length'],2)}\nMAE: {round(error_metrics_dict['original_rfu_cfx']['mae'],2)}\nMSE: {round(error_metrics_dict['original_rfu_cfx']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (1,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_analysis_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['strep_analysis_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_analysis_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i, j) == (1, 2):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus1_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus1_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus1_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                ax.plot(cycle, rfu, alpha=0.5)
                ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
                ax.set_title(title)
            if (i,j) == (0,0):
                ax.axhline(y=min_mean,color='black',linestyle='dotted',linewidth=2)
                if rfu_values:
                    ax.set_ylim([min(after_bpn_limits)*0.98,max(after_bpn_limits)*1.02])
            else:
                    ax.set_ylim(min(limits)*1.2,max(limits)*1.5)        
        else:  
            ax.set_title(title)
    
    fig.suptitle(f'PCRD: {i_pcrd}\nChannel: {i_channel}, Temperature: {i_temperature}, The Number of Signals: {i_data.shape[0]}', 
                 ha = 'left', x=0.02, fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_single_well(i_data, i_pcrd, i_channel, i_temperature, i_well):
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
    
    i_data = i_data.query("`well` == @i_well").copy()
    columns = [
        ('corrected_rfu', 'original_rfu', 'analysis_rd_diff'), # (new_column,from,to) 
        ('dsp_fit', 'original_rfu', 'analysis_absd_orig'),
        ('basesub_corrected_rfu', 'original_rfu', 'basesub_rd_diff'),
        ('basesub_fit', 'original_rfu', 'basesub_absd_orig'),
        ('original_rfu_cfx_fit', 'original_rfu', 'original_rfu_cfx'),
        ('strep_corrected_rfu', 'original_rfu', 'strep_analysis_rd_diff'),
        ('strep_fit', 'original_rfu', 'strep_analysis_absd_orig'),
        ('strep_plus1_fit', 'original_rfu', 'strep_plus1_absd')
    ]
    
    for new_column, column_a, column_b in columns:
        i_data[new_column] = list(map(lambda x, y: [a - b for a, b in zip(x, y)], i_data[column_a], i_data[column_b]))
    
    baseline_scale_data = i_data[['corrected_rfu','dsp_fit','analysis_scd_fit',
                            'basesub_corrected_rfu','basesub_fit','basesub_scd_fit',
                            'original_rfu_cfx_fit',
                            'strep_corrected_rfu','strep_fit','strep_analysis_scd_fit',
                            'strep_plus1_corrected_rfu', 'strep_plus1_baseline_fit','strep_plus1_fit']]
    subtract_scale_data = i_data[['analysis_absd_orig', 'basesub_absd_orig','original_rfu_cfx',
                                  'strep_analysis_absd_orig', 'strep_plus1_absd']]
    
    limits = find_global_extremes(baseline_scale_data)
    limits_subtraction = find_global_extremes(subtract_scale_data)
    
    titles = ['[DSP] Algorithm Performance', '[Auto] Algorithm Performance', '[CFX] Algorithm Performance', '[Strep] Algorithm Performance', '[Strep+1] Algorithm Performance',
              '[DSP] Fitting Performance', '[Auto] Fitting Performance', '[CFX] Fitting Performance', '[Strep] Fitting Performance', '[Strep+1] Fitting Performance',
              '[DSP] Baseline-Subtracted RFU', '[Auto] Baseline-Subtracted RFU', '[CFX] Baseline-Subtracted RFU', '[Strep] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU']
    
    preprocessing_columns = [
        ('dsp_fit', 'original_rfu'),
        ('basesub_fit', 'original_rfu'),
        ('original_rfu_cfx_fit', 'original_rfu'),
        ('strep_fit', 'original_rfu'),
        ('strep_plus1_fit', 'original_rfu')]
    baseline_columns = [
        ('corrected_rfu', 'original_rfu', 'analysis_scd_fit'), 
        ('basesub_corrected_rfu', 'original_rfu', 'basesub_scd_fit'),
        ('original_rfu', 'original_rfu', 'original_rfu_cfx_fit'),
        ('strep_corrected_rfu', 'original_rfu', 'strep_analysis_scd_fit'),
        ('strep_plus1_corrected_rfu', 'original_rfu', 'strep_plus1_baseline_fit')]
    
    subtracted_columns = [    
        ('analysis_absd_orig'),
        ('basesub_absd_orig'),
        ('original_rfu_cfx'),
        ('strep_analysis_absd_orig'),
        ('strep_plus1_absd')]
    
    fig, axs = plt.subplots(3, 5, figsize=(18, 12)) #(15, 10)
    
    rfu_values = []
    rfu = i_data['original_rfu'].values[0]
    cycle = list(range(len(rfu)))        
    correct_alpha = 0.5
    fit_alpha = 0.5
    
    for ((i, j), ax, title) in zip(np.ndindex(axs.shape), axs.flat, titles):
        if i == 0:
            ax.plot(cycle, i_data[preprocessing_columns[j][0]].values[0], 'red', alpha=0.5, label = "Correction & Fit")
            ax.plot(cycle, i_data[preprocessing_columns[j][1]].values[0], 'k-', alpha=0.5, label = "Raw Data")
            
        elif i == 1:
            ax.plot(cycle, i_data[baseline_columns[j][0]].values[0], 'b--', alpha=1, label = "Correction")
            ax.plot(cycle, i_data[baseline_columns[j][1]].values[0], 'k-', alpha=0.5, label = "Raw Data")
            ax.plot(cycle, i_data[baseline_columns[j][2]].values[0], 'red', alpha=0.5, label = "Fitted Data" )
        else:
            ax.plot(cycle, i_data[subtracted_columns[j]].values[0], 'k-', alpha=0.5, label = "Subtracted Data")
            ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
    
        # Set limits based on row
        if i == 0:
            ax.set_ylim([min(limits) * 0.98, max(limits) * 1.02])
        elif i == 1:
            ax.set_ylim([min(limits) * 0.98, max(limits) * 1.02])
        else:
            ax.set_ylim([min(limits_subtraction) * 1.05, max(limits_subtraction) * 1.05])
        ax.set_title(title)
        ax.legend()
    fig.suptitle(f'PCRD: {i_pcrd}\nChannel: {i_channel}, Temperature: {i_temperature}, Well: {i_well}', 
                     ha = 'left', x=0.02, fontsize=15)
    
    plt.tight_layout()
    plt.show()

def plot_signal_patterns(i_data):

    min_mean=i_data['original_rfu'].apply(lambda x: np.mean(x)).min()
    i_data.loc[:, 'after_bpn_rfu'] = i_data['original_rfu'].apply(lambda x: compute_bpn(x, min_mean))
    
    #baseline_data = i_data[['analysis_absd_orig','basesub_absd_orig', 'original_rfu_cfx','strep_analysis_absd_orig','strep_plus1_absd']]
    baseline_data = i_data[['basesub_absd_orig', 'original_rfu_cfx','strep_analysis_absd_orig','strep_plus1_absd']]
    after_bpn_limits = find_global_extremes(i_data['after_bpn_rfu'])
    limits = find_global_extremes(baseline_data)
    titles = ['[After BPN] RFU', '[Auto] Baseline-Subtracted RFU', 
              '[CFX] Baseline-Subtracted RFU', '[Strep+1] Baseline-Subtracted RFU']
    data_keys = ['after_bpn_rfu','basesub_absd_orig', 'original_rfu_cfx', 'strep_plus1_absd']
    grouping_columns = ['original_rfu_cfx', 'basesub_absd_orig', 'strep_plus1_absd']
    
    
    error_metrics_df = (i_data
                        .groupby(['name', 'channel', 'temperature'])
                        .apply(lambda x: get_comparison_metrics(x, grouping_columns),include_groups=False)
                        .reset_index(drop=True))
    
    error_metrics_dict = error_metrics_df.iloc[0]
            
    metric_min=round(i_data['outlier_naive_metric_original_rfu'].min(),2)
    metric_max=round(i_data['outlier_naive_metric_original_rfu'].max(),2)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) #(15, 10)
    
    
    for (i, j), ax, title, key in zip(np.ndindex(axs.shape), axs.flat, titles, data_keys):
        if key:  
            rfu_values = []
            for index, row in i_data.iterrows():
                rfu = row[key]
                rfu_values.extend(rfu)
                cycle = list(range(len(rfu)))
               #color = colors[row['temperature']]
                if (i,j) == (0,0):
                    ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\nOutlier Naive Metric: [{metric_min},{metric_max}]\nBPN Reference Value: {round(min_mean)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (0,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['basesub_absd_orig']['length'],2)}\nMAE: {round(error_metrics_dict['basesub_absd_orig']['mae'],2)}\nMSE: {round(error_metrics_dict['basesub_absd_orig']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (1,0):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['original_rfu_cfx']['length'],2)}\nMAE: {round(error_metrics_dict['original_rfu_cfx']['mae'],2)}\nMSE: {round(error_metrics_dict['original_rfu_cfx']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                elif (i,j) == (1,1):
                    ax.text(0.05, 0.98, f"n: {round(error_metrics_dict['strep_plus1_absd']['length'],2)}\nMAE: {round(error_metrics_dict['strep_plus1_absd']['mae'],2)}\nMSE: {round(error_metrics_dict['strep_plus1_absd']['mse'],2)}", 
                            verticalalignment='top', horizontalalignment='left', 
                            transform=ax.transAxes)
                                
                ax.plot(cycle, rfu, alpha=0.5)
                ax.axhline(y=0,color='black',linestyle='dotted',linewidth=2)
                ax.set_title(title)
            if (i,j) == (0,0):
                ax.axhline(y=min_mean,color='black',linestyle='dotted',linewidth=2)
                if rfu_values:
                    ax.set_ylim([min(after_bpn_limits)*0.98,max(after_bpn_limits)*1.02])
            else:
                    ax.set_ylim(min(limits)*1.2,max(limits)*1.5)        
        else:  
            ax.set_title(title)
    fig.suptitle(f"DataProcNumber: {i_data['analysis_dataprocnum'].unique()[0]}\nThe Number of Signals: {i_data.shape[0]}",ha = 'left', x=0.02, fontsize=15)
    
    plt.tight_layout()
    plt.show()

