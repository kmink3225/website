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
import subprocess
from pathlib import Path

# Visualization
import pydsptools.plot as dspplt
import plotly.express as px
import matplotlib.pyplot as plt

# Noise Naive Detection

# Noise Naive Detection

def detect_noise_naively(i_signal, i_threshold=1.28):
    '''
    detect noisy signals.     
    Args:
    - i_signal: a rfu signal with a data type as a list
    - i_threshold: the z score
    Returns:
    - o_result
    '''
    
    if i_signal.size == 0:  # Check if the signal is empty
        return [False, 0, []]  # Return default values for empty signals
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)

    if std == 0:  # Prevent division by zero
        return [False, 0, []]

    z_scores = [(point - mean) / std for point in i_signal]
    outliers = [score * std + mean for score in z_scores if abs(score) > i_threshold]
    outliers_boolean = [abs(score) > i_threshold for score in z_scores]
    outlier_existence = 1 if any(outliers_boolean) else 0
    outlierness_metric_sum = sum(abs(score) for score in z_scores if abs(score) > i_threshold)
    outlierness_metric = outlierness_metric_sum / len(outliers) if len(outliers) != 0 else 0
    o_result = [outlierness_metric, outliers,z_scores,outlier_existence]
    return o_result

def detect_noise_naively_ver2(i_signal, i_threshold=1.28):
    '''
    detect noisy signals. 10 cycle 버리고 다시 카운트할 것
    
    Args:
    - i_signal: a rfu signal with a data type as a list
    - i_threshold: the z score = 1.65 

    Returns:
    - o_result: [any(outliers),outlierness_metric, outliers] with the data types of [boolean, a real number, a list of outliers ] 
    '''
    i_signal = i_signal[10:]
    if i_signal.size == 0:  # Check if the signal is empty
        return [False, 0, []]  # Return default values for empty signals
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)

    if std == 0:  # Prevent division by zero
        return [False, 0, []]

    z_scores = [(point - mean) / std for point in i_signal]
    outliers = [score for score in z_scores if abs(score) > i_threshold]
    outliers_boolean = [abs(score) > i_threshold for score in z_scores]
    outlier_existence = 1 if any(outliers_boolean) else 0
    outlierness_metric_sum = sum(abs(score) for score in z_scores if abs(score) > i_threshold)
    outlierness_metric = outlierness_metric_sum / len(outliers) if len(outliers) != 0 else 0
    o_result = [outlier_existence, outlierness_metric, outliers]
    return o_result

def detect_noise_naively_ywj1(i_signal):
    '''
    sum(abs(RFU_i)) after BPN
        
    Args:
    - i_signal: a rfu signal with a data type as a list

    Returns:
    - o_result: [noise_metric, residuals, after_bpn] with the data types of [a real number, a list, a list] 
    '''
    
    if i_signal.size == 0:  # Check if the signal is empty
        return [False, 0, []]  # Return default values for empty signals
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)

    after_bpn = [(point / mean) * 100 for point in i_signal] # a normalized signal with the reference value = 100
    mean_adjusted = np.mean(after_bpn)
    residuals = [(rfu - mean_adjusted) for rfu in after_bpn]
    noise_metric = sum(abs(rfu) for rfu in residuals)
    
    o_result = [noise_metric, residuals, after_bpn]
    return o_result
    
def detect_noise_naively_pbg(i_signal):
    '''
    Detect noisy signals by discarding the first 10 cycles and then evaluating the rest based on a dynamic threshold
    that captures the top 20% of deviation in terms of scores calculated as a percentage of the mean.
    '''
    
    if len(i_signal) == 0:  # Check if the signal is empty after discarding
        return [False, 0, []]  # Return default values for empty signals
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)
    
    if std == 0:  # Prevent division by zero
        return [False, 0, []]
    
    scores = [(point - mean) for point in i_signal]  # Calculate scores as percentage of mean
    threshold = np.percentile(scores, 80)  # Dynamic threshold for top 20% scores
    
    outliers = [score + mean for score in scores if abs(score) > threshold]  # Identify outliers
    outlier_existence = bool(outliers)  # True if there are any outliers
    outlierness_metric = sum(abs(score) - threshold for score in outliers) / len(outliers) if outliers else 0  # Calculate outlierness metric
    
    return [outlierness_metric, outliers, scores, outlier_existence]

def detect_noise_naively_kkm(i_signal):
    '''
    Detect noisy signals by subtracting a linear regression from the rfu values of the original signal 
    and by summing the absolute values of the residuals.
    '''

    if len(i_signal) == 0: 
        return [False, 0, []]
        
    mean = np.mean(i_signal)
    after_bpn = [(point / mean) * 100 for point in i_signal]  # Calculate scores as percentage of mean
    x = list(range(1, len(after_bpn) + 1))
    slope = np.cov(x, after_bpn, bias=True)[0, 1] / np.var(x, ddof=0)
    intercept = np.mean(after_bpn) - slope * np.mean(x)
    x_values = np.array(x)
    linear_fit = slope * x_values + intercept
    residuals = after_bpn-linear_fit
    metric = sum(abs(residuals))
    
    return [metric, residuals,slope,intercept]
    

# detect white noise signals
def compute_autocorrelation(i_signal, i_lag):
    '''
    this function computes autocorrelation value of a signal

    Args:
    - i_signal: a PCR signal
    - i_lag: a lag window

    Returns:
    - o_result: an autocorreation value
    '''
    n = len(i_signal)
    mean_x = np.mean(i_signal)
    denominator = np.sum((i_signal - mean_x) ** 2)
    numerator = np.sum((i_signal[:-i_lag] - mean_x) * (i_signal[i_lag:] - mean_x))
    o_result = numerator / denominator 
    return o_result

def test_white_noise(i_signal, i_max_lag):
    '''
    this function tests if a signal is white noise.

    args:
    - i_signal: a PCR signal
    - i_max_lag: a max lag

    return:
    - o_result: a test statistic
    '''
    n = len(i_signal)
    statistic = 0
    for lag in range(1, i_max_lag + 1):
        rho = autocorrelation(i_signal, lag)
        statistic += (n * (n + 2) * rho ** 2) / (n - lag)
    p_value = 1 - chi2.cdf(statistic, i_max_lag)
    o_result = [statistic, p_value]
    return o_result
