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

INPUT_PARQUET_DIR = "./data/GI-B-I/GI-B-I-100/computed/pcr_results/"
CONFIG_YML_PATH = "./config/yaml/PRJDS001/GI-B-I/dsp2_strep_plus2_config_no-MuDT.yml"
TO_DSP_RESULT_DIR = "./data/GI-B-I/strep_plus2/computed/dsp2_strep_plus2_config_no-MuDT"

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
