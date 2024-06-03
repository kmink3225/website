import pydsp.run.worker
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
import shutil
import subprocess
import pathlib as Path
import psutil # memory checking
import shutil
from source.preprocess import (check_memory_status,
                                get_disk_usage)
# Visualization
import pydsptools.plot as dspplt
import plotly.express as px
import matplotlib.pyplot as plt

# Son's Module
import pydsptools.utils
import pydsptools.biorad.parse as pdabioradparse
import pydsptools.plot as dspplt

import sys
import yaml
import fsspec

from dataclasses import dataclass
from numpy import diff
import seaborn as sns
import matplotlib.patches as patches
import scipy.io as sio
import math
import plotly.express as px
import plotly.graph_objects as go

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

parent_dir = os.path.abspath('../hsson1')
sys.path.append(parent_dir)

## Import the entire module
import source.general_derivative_analysis
import source.derivative_peak_analysis_ver1_9
import source.baseline_estimation_ver1_3
import source.calculate_ct_ver1_1
import source.plot_analysis

import importlib

## Reload the module to reflect any changes
importlib.reload(source.general_derivative_analysis)
importlib.reload(source.derivative_peak_analysis_ver1_9)
importlib.reload(source.baseline_estimation_ver1_3)
importlib.reload(source.calculate_ct_ver1_1)
importlib.reload(source.plot_analysis)

## Now, import the specific functions from the reloaded module
from source.general_derivative_analysis import (moving_average,
                                                compute_derivative_LSR,
                                                compute_max_deriv_data,
                                                plot_max_deriv_data,
                                                compute_max_deriv_smoothed_data,
                                                plot_max_deriv_smoothed_data)
from source.derivative_peak_analysis_ver1_9 import (get_peak_groups,
                                             calculate_crossing_points_for_data,
                                             get_peak_properties_for_row,
                                             plot_peak_properties,
                                             plot_derivative_baseline_modeling)
from source.baseline_estimation_ver1_3 import (coder_scd_fitting,
                                        coder_section_rp2,
                                        adjusted_r2,
                                        operation_baseline_fitting,
                                        calculate_base_rfu,
                                        plot_baseline,
                                        plot_comparison)
from source.calculate_ct_ver1_1 import (compute_crossing_point,
                                        label_threshold_result,
                                        plot_data_threshold_crossing)
from source.plot_analysis import (plot_Signal,
                                  plot_2d_scatter,
                                  PGR_manager_plot)



INPUT_PARQUET_DIR = "./data/GI-B-I/GI-B-I-100/computed/pcr_results/"
CONFIG_YML_PATH = "./config/yaml/PRJDS001/GI-B-I/dsp2_generic_config_no-MuDT.yml"
TO_DSP_RESULT_DIR = "./data/GI-B-I/raw_data/computed/dsp2_generic_config_no_MuDT"

pydsp.run.worker.multiple_tasks(
    INPUT_PARQUET_DIR, # Input directory
    CONFIG_YML_PATH,   # Configuration
    TO_DSP_RESULT_DIR,     # Output directory
    2,                 # Number of processes
    is_verbose=False    # Verboase mode
)


CONFIG_YML_PATH = "./config/yaml/PRJDS001/GI-B-I/dsp2_generic_config_MuDT.yml"
TO_DSP_RESULT_DIR = "./data/GI-B-I/raw_data/computed/dsp2_generic_config_MuDT"

pydsp.run.worker.multiple_tasks(
    INPUT_PARQUET_DIR, # Input directory
    CONFIG_YML_PATH,   # Configuration
    TO_DSP_RESULT_DIR,     # Output directory
    4,                 # Number of processes
    is_verbose=False    # Verboase mode
)

INPUT_PARQUET_DIR = "./data/GI-B-I/GI-B-I-100/computed/pcr_results/"
CONFIG_YML_PATH = "./config/yaml/PRJDS001/GI-B-I/dsp2_strep_plus1_config_MuDT.yml"
TO_DSP_RESULT_DIR = "./data/GI-B-I/strep_plus1/computed/dsp2_strep_plus1_config_MuDT"

pydsp.run.worker.multiple_tasks(
    INPUT_PARQUET_DIR, # Input directory
    CONFIG_YML_PATH,   # Configuration
    TO_DSP_RESULT_DIR,     # Output directory
    4,                 # Number of processes
    is_verbose=False    # Verboase mode
)


INPUT_PARQUET_DIR = "./data/GI-B-I/GI-B-I-100/computed/pcr_results/"
CONFIG_YML_PATH = "./config/yaml/PRJDS001/GI-B-I/dsp2_strep_plus1_config_no-MuDT.yml"
TO_DSP_RESULT_DIR = "./data/GI-B-I/strep_plus1/computed/dsp2_strep_plus1_config_no-MuDT"

pydsp.run.worker.multiple_tasks(
    INPUT_PARQUET_DIR, # Input directory
    CONFIG_YML_PATH,   # Configuration
    TO_DSP_RESULT_DIR,     # Output directory
    4,                 # Number of processes
    is_verbose=False    # Verboase mode
)



INPUT_PARQUET_DIR = "./data/GI-B-I/GI-B-I-100/computed/pcr_results/"
CONFIG_YML_PATH = "./config/yaml/PRJDS001/GI-B-I/dsp2_strep_plus2_config_MuDT.yml"
TO_DSP_RESULT_DIR = "./data/GI-B-I/strep_plus2/computed/dsp2_strep_plus2_config_MuDT"

pydsp.run.worker.multiple_tasks(
    INPUT_PARQUET_DIR, # Input directory
    CONFIG_YML_PATH,   # Configuration
    TO_DSP_RESULT_DIR,     # Output directory
    4,                 # Number of processes
    is_verbose=False    # Verboase mode
)


