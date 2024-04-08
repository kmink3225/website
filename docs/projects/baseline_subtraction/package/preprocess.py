# Config
#import pydsptools.config.pda as pda
#import pydsptools.biorad as biorad

# Analysis Preparation
#import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import chi2 # https://en.cppreference.com/w/cpp/numeric/random/chi_squared_distribution

# DSP Processing
#import pydsp.run.worker
#import pydsptools.biorad.parse as bioradparse

# PreProcessing
import pprint
import pyarrow as pa
import os
#os.chdir('/home/jupyter-kmkim/dsp-research-strep-a/kkm')
import subprocess
from pathlib import Path
from package import signal_filter
from package import (signal_filter,visualization)

# Visualization
#import pydsptools.plot as dspplt
import plotly.express as px
import matplotlib.pyplot as plt


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

