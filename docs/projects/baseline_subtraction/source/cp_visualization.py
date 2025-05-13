"""
PCR 신호 데이터 시각화 모듈

이 모듈은 PCR(Polymerase Chain Reaction) 신호 데이터를 다양한 방식으로 시각화한다.

주요 기능:
- 베이스라인 차감 알고리즘 결과 비교 시각화
- 단일 웰(well)에 대한 신호 데이터 시각화 
- BPN(Background Point Normalization) 기반 신호 변환 및 표현
- 여러 알고리즘(DSP, Auto, CFX, Strep+2, ML)의 베이스라인 보정 성능 비교

Examples
--------
>>> import pandas as pd
>>> from cp_visualization import plot_baseline_subtractions
>>> data = pd.read_parquet('sample_pcr_data.parquet')
>>> plot_baseline_subtractions(data, 'Sample1', 'FAM', 60.0)

See Also
--------
cp_signal_estimation : PCR 신호 데이터 기반 베이스라인 최적화 예측 모듈

Notes
-----
이 모듈은 matplotlib을 사용해 다양한 시각화를 구현한다.

작성자: Kwangmin Kim
날짜: 2024-07
"""

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_package_details():
    """
    현재 Python 환경에 설치된 모든 패키지와 해당 버전 정보를 가져옴
    
    이 함수는 'pip list' 명령어를 실행하여 설치된 모든 패키지 목록을 가져오고,
    패키지 이름과 버전을 딕셔너리 형태로 반환
    
    Returns:
        dict: 패키지 이름을 키로, 버전을 값으로 하는 딕셔너리
    
    Example:
        >>> get_package_details()
        {'numpy': '1.21.0', 'pandas': '1.3.0', ...}
    """
    import subprocess # 외부 명령어 실행
    import sys  
    
    # pip list 명령어 실행
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list'], 
        stdout=subprocess.PIPE, 
        text=True
    )
    lines = result.stdout.split('\n')
    
    # 패키지 정보 파싱
    package_dict = {}
    for line in lines[2:]:  # 첫 두 줄은 헤더 정보이므로 제외
        parts = line.split()
        if len(parts) >= 2:
            package_name = parts[0]
            version = parts[1]
            package_dict[package_name] = version
            
    return package_dict


def find_sub_extremes(i_signal):
    '''
    DataFrame 열의 각 셀 내에서 최대값과 최소값을 찾고, 
    전체 열에 대한 최대/최소값을 계산하며, 이 결과는 축 범위 설정에 사용됨
    
    Args:
        i_signal: DataFrame 열의 신호값 리스트
        
    Returns:
        o_result: 신호값의 최대값과 최소값을 포함하는 리스트 [max, min]
    '''
    def process_cell(i_signal):
        '''
        입력값이 리스트 형태이고 비어있지 않은지 확인
        
        Args:
            i_signal: DataFrame 열의 신호값 리스트
            
        Returns:
            o_result: 신호값의 최대값과 최소값을 포함하는 Series
        '''
        # 신호가 유효한 데이터 타입이고 비어있지 않은 경우
        if isinstance(i_signal, (list, np.ndarray, pd.Series)) and len(i_signal):            
            return pd.Series(
                [np.max(i_signal), np.min(i_signal)], 
                index=['max', 'min']
            )
        return pd.Series([pd.NA, pd.NA], index=['max', 'min'])
    
    # 각 셀에 대해 최대/최소값 찾고 결측치 제거
    extremes_df = i_signal.apply(process_cell).dropna()
    
    if extremes_df.empty:
        return [None, None]
    
    # 전체 최대/최소값 계산
    return [extremes_df['max'].max(), extremes_df['min'].min()]


def find_global_extremes(i_data):
    """입력 데이터프레임 또는 시리즈의 전역 최대/최소값을 찾는다.
    
    이 함수는 단일 시리즈 또는 데이터프레임의 모든 열에 대해 
    전역 최대값과 최소값을 계산한다.
    
    Parameters
    ----------
    i_data : pandas.DataFrame or pandas.Series
        최대/최소값을 찾을 데이터프레임 또는 시리즈
        
    Returns
    -------
    list
        신호값의 최대값과 최소값을 포함하는 리스트 [max, min]
        유효한 데이터가 없는 경우 [None, None] 반환
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [0, 5, -1]})
    >>> find_global_extremes(df)
    [5, -1]
    
    >>> series = pd.Series([10, 20, 5, 15])
    >>> find_global_extremes(series)
    [20, 5]
    """
    # 데이터프레임
    if isinstance(i_data, pd.DataFrame):
        # 각 열에 대한 최대/최소값 계산
        extremes_results = {column: find_sub_extremes(i_data[column]) 
                           for column in i_data.columns}
        
        overall_max = float('-inf')
        overall_min = float('inf')
        
        # 모든 열의 최대/최소값 중에서 전역 최대/최소값 찾기
        for column_max, column_min in extremes_results.values():
            if column_max is not None and column_min is not None:
                overall_max = max(overall_max, column_max)
                overall_min = min(overall_min, column_min)
    # 시리즈
    elif isinstance(i_data, pd.Series):
        overall_max, overall_min = find_sub_extremes(i_data)
    else:
        return [None, None]
    
    return [overall_max, overall_min]


def get_comparison_metrics(i_grouped_data, i_columns):
    '''
    여러 알고리즘의 성능 비교를 위한 메트릭을 계산하는 함수
    
    Args:
        i_grouped_data: groupby()로 그룹화된 데이터프레임
        i_columns: 메트릭을 계산할 열 이름 목록
    Returns:
        o_result: 각 알고리즘별 메트릭 정보가 담긴 Series
    '''
    metrics = {}
    
    for column in i_columns:
        # 빈 리스트 필터링
        filtered_data = i_grouped_data[column][i_grouped_data[column].apply(len) > 0]
        
        if not filtered_data.empty:
            length = len(filtered_data)
            # 벡터화된 연산으로 효율성 향상
            absolute_means = filtered_data.apply(lambda x: np.abs(x).mean())
            squared_means = filtered_data.apply(lambda x: np.square(x).mean())
            
            absolute_means_sum = absolute_means.sum()
            squared_means_sum = squared_means.sum()
            
            metrics[column] = {
                'length': length,
                'absolute_sum': round(absolute_means_sum, 2),
                'squared_sum': round(squared_means_sum, 2),
                'mse': round(squared_means_sum/length, 2),
                'mae': round(absolute_means_sum/length, 2)
            }
        else:
            metrics[column] = {
                'length': 0,
                'absolute_sum': 0,
                'squared_sum': 0,
                'mse': 0,
                'mae': 0
            }
    
    return pd.Series(metrics)


def compute_bpn(i_signal, i_reference_value):
    """신호 포인트를 참조 값으로 스케일로 정규화하는 BPN(Background Point Normalization)을 계산한다.
    
    이 함수는 PCR 데이터의 배경 신호를 정규화하여 다양한 샘플 간 비교가 가능하게 한다.
    입력 신호의 처음 3-12 사이클의 평균값을 기준으로 전체 신호를 정규화한다.
    
    Parameters
    ----------
    i_signal : numpy.ndarray
        정규화할 RFU 신호 포인트 배열
    i_reference_value : float
        신호 변환에 사용되는 참조 값
    
    Returns
    -------
    list or numpy.ndarray
        정규화된 신호 배열. 빈 입력의 경우 [False, 0, []] 반환
    
    Examples
    --------
    >>> signal = np.array([10, 12, 15, 20, 30, 50, 100])
    >>> reference = 100
    >>> normalized = compute_bpn(signal, reference)
    >>> print(normalized)
    [66.67, 80.0, 100.0, 133.33, 200.0, 333.33, 666.67]
    
    Notes
    -----
    신호의 처음 3-12 사이클의 평균값이 기준으로 사용된다.
    """
    if i_signal.size == 0:  # 빈 신호에 대한 기본값 반환
        return [False, 0, []]  
        
    # 신호의 3-12번째 사이클 평균으로 정규화
    mean = np.mean(i_signal[3:12])
    # 0으로 나누기 방지
    mean = mean if mean != 0 else 1.0
    
    # 참조값 기준으로 신호 정규화
    after_bpn = [(point / mean) * i_reference_value for point in i_signal]
    
    return after_bpn


###########################################
# 베이스라인 차감 시각화
###########################################

def prepare_baseline_data(i_data, i_pcrd, i_channel, i_temperature, mudt=False):
    """
    베이스라인 차감 시각화를 위한 데이터 준비
    
    Args:
        i_data: 원본 데이터프레임
        i_pcrd: PCRD 이름
        i_channel: 채널 이름
        i_temperature: 온도 값
        mudt: MuDT 데이터 여부
        
    Returns:
        tuple: (필터링된 데이터, 타이틀 리스트, 데이터 키 리스트, 
        원본 RFU 평균 최소값, 전처리 RFU 평균 최소값)
    """
    # 필터링된 데이터 준비
    filtered_data = i_data.query(
        "`name`==@i_pcrd & `channel` == @i_channel & `temperature`==@i_temperature"
    ).copy()
    
    # MuDT 데이터 여부에 따른 타이틀과 데이터 키 설정
    if mudt:
        titles = [
            '[After BPN] Raw RFU', 
            '[After BPN] Preproc RFU', 
            '[DSP] Original ABSD', 
            '[Auto] Baseline-Subtracted RFU', 
            '[Strep+2] Baseline-Subtracted RFU', 
            '[ML] Baseline-Subtracted RFU'
        ]
        data_keys = [
            'original_rfu_after_bpn', 
            'preproc_rfu_after_bpn', 
            'analysis_absd_orig', 
            'basesub_absd_orig', 
            'strep_plus2_analysis_absd', 
            'ml_analysis_absd'
        ]
    else:
        titles = [
            '[After BPN] Raw RFU', 
            '[CFX] Baseline-Subtracted RFU', 
            '[DSP] Original ABSD', 
            '[Auto] Baseline-Subtracted RFU', 
            '[Strep+2] Baseline-Subtracted RFU', 
            '[ML] Baseline-Subtracted RFU'
        ]
        data_keys = [
            'original_rfu_after_bpn', 
            'original_rfu_cfx', 
            'analysis_absd_orig', 
            'basesub_absd_orig', 
            'strep_plus2_analysis_absd', 
            'ml_analysis_absd'
        ]
    
    # BPN 계산 - 원본 RFU
    original_rfu_min_mean = filtered_data['original_rfu'].apply(
        lambda x: np.mean(x)
    ).min()

    filtered_data.loc[:, 'original_rfu_after_bpn'] = filtered_data['original_rfu'].apply(
        lambda x: compute_bpn(x, original_rfu_min_mean)
    )
    
    # BPN 계산 - 전처리 RFU
    preproc_rfu_min_mean = filtered_data['preproc_rfu'].apply(
        lambda x: np.mean(x)
    ).min()

    filtered_data.loc[:, 'preproc_rfu_after_bpn'] = filtered_data['preproc_rfu'].apply(
        lambda x: compute_bpn(x, preproc_rfu_min_mean)
    )
    
    # 배열 데이터 전처리
    array_columns = [
        col for col in filtered_data.columns 
        if filtered_data[col].apply(lambda x: isinstance(x, np.ndarray)).any()
    ]

    cycle_length = len(filtered_data['original_rfu'].values[0])
    
    for column in array_columns:
        filtered_data[column] = filtered_data[column].apply(
            lambda x: np.zeros(cycle_length) if x is None or x.size == 0 else x
        )
    
    return (filtered_data, titles, data_keys, 
            original_rfu_min_mean, preproc_rfu_min_mean)

def calculate_error_metrics(filtered_data):
    """
    오류 메트릭 계산
    
    Args:
        filtered_data: 필터링된 데이터프레임
        
    Returns:
        tuple: (오류 메트릭 딕셔너리, 메트릭 최소값, 메트릭 최대값)
    """
    # 베이스라인 데이터 추출
    baseline_data = filtered_data[[
        'analysis_absd_orig', 
        'basesub_absd_orig', 
        'original_rfu_cfx', 
        'strep_plus2_analysis_absd', 
        'ml_analysis_absd'
    ]]
    
    # 그룹화에 사용할 컬럼 정의
    grouping_columns = [
        'analysis_absd_orig', 
        'original_rfu_cfx', 
        'basesub_absd_orig', 
        'strep_plus2_analysis_absd', 
        'ml_analysis_absd'
    ]
    
    # 오류 메트릭 계산
    error_metrics_df = (
        filtered_data
        .groupby(['name', 'channel', 'temperature'])
        .apply(lambda x: get_comparison_metrics(x, grouping_columns))
        .reset_index(drop=True)
    )
    
    # 첫 번째 행의 메트릭 추출
    error_metrics_dict = error_metrics_df.iloc[0]
    
    # 메트릭 최소/최대값 계산
    metric_min = round(filtered_data['outlier_naive_metric_original_rfu'].min(), 2)
    metric_max = round(filtered_data['outlier_naive_metric_original_rfu'].max(), 2)
    
    # 제한값 계산
    original_rfu_after_bpn_limits = find_global_extremes(filtered_data['original_rfu_after_bpn'])
    preproc_rfu_after_bpn_limits = find_global_extremes(filtered_data['preproc_rfu_after_bpn'])
    limits = find_global_extremes(baseline_data)
    
    # 결과 반환
    return (
        error_metrics_dict, 
        metric_min, 
        metric_max, 
        original_rfu_after_bpn_limits, 
        preproc_rfu_after_bpn_limits, 
        limits
    )


def plot_single_baseline_panel(ax, i_data, panel_info, error_metrics_dict, 
                              limits_info, mudt=False):
    """
    베이스라인 시각화의 단일 패널 그리기
    
    Args:
        ax: matplotlib 축 객체
        i_data: 데이터프레임
        panel_info: (i, j, title, key) 튜플
        error_metrics_dict: 오류 메트릭 딕셔너리
        limits_info: (metric_min, metric_max, 
                      original_min_mean, preproc_min_mean, 
                      original_after_bpn_limits, preproc_after_bpn_limits, limits) 튜플
        mudt: MuDT 데이터 여부
    """
    
    i, j, title, key = panel_info
    metric_min, metric_max = limits_info[0], limits_info[1]
    original_rfu_min_mean, preproc_rfu_min_mean = limits_info[2], limits_info[3]
    original_rfu_after_bpn_limits = limits_info[4]
    preproc_rfu_after_bpn_limits = limits_info[5]
    limits = limits_info[6]
    
    if not key:
        ax.set_title(title)
        return
        
    rfu_values = []
    for index, row in i_data.iterrows():
        rfu = row[key]
        rfu_values.extend(rfu)
        cycle = list(range(len(rfu)))
        ax.plot(cycle, rfu, alpha=0.5)
    
    # 패널별 텍스트 및 레이아웃 설정
    if (i, j) == (0, 0):
        ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\n"
               f"Outlier Naive Metric: [{metric_min},{metric_max}]\n"
               f"BPN Reference Value: {round(original_rfu_min_mean)}", 
               verticalalignment='top', horizontalalignment='left', 
               transform=ax.transAxes)
        ax.axhline(y=original_rfu_min_mean, color='black', linestyle='dotted', linewidth=2)
        if rfu_values:
            ax.set_ylim([min(original_rfu_after_bpn_limits)*0.98, 
                         max(original_rfu_after_bpn_limits)*1.02])
    elif (i, j) == (0, 1):
        ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\n"
               f"Outlier Naive Metric: [{metric_min},{metric_max}]\n"
               f"BPN Reference Value: {round(preproc_rfu_min_mean)}", 
               verticalalignment='top', horizontalalignment='left', 
               transform=ax.transAxes)
        ax.axhline(y=preproc_rfu_min_mean, color='black', linestyle='dotted', linewidth=2)
        if rfu_values:
            ax.set_ylim([min(preproc_rfu_after_bpn_limits)*0.98, 
                         max(preproc_rfu_after_bpn_limits)*1.02])
    elif (i, j) == (1, 0) or (i, j) == (1, 1) or (i, j) == (1, 2) or (i, j) == (0, 2):
        if key in error_metrics_dict:
            metrics = error_metrics_dict[key]
            ax.text(0.05, 0.98, f"N: {i_data.shape[0]}\n"
                   f"MAE: {metrics['mae']}\n"
                   f"MSE: {metrics['mse']}", 
                   verticalalignment='top', horizontalalignment='left', 
                   transform=ax.transAxes)
        if rfu_values:
            ax.set_ylim([min(limits)*0.98, max(limits)*1.02])
        
    # 공통 레이아웃 설정
    ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2)
    ax.set_title(title)

def plot_baseline_subtractions(i_data, i_pcrd, i_channel, i_temperature, 
                              colors=None, mudt=False):
    """베이스라인 차감 알고리즘별 결과를 시각화하여 직관적으로 성능을 비교한다.
    
    여러 베이스라인 차감 알고리즘의 결과를 2×3 서브플롯으로 시각화하여 비교한다.
    
    Parameters
    ----------
    i_data : pandas.DataFrame
        raw RFU와 여러 알고리즘으로 베이스라인 차감된 RFU 결과가 병합된 데이터프레임.
        필수 열: 'name', 'channel', 'temperature', 'original_rfu', 'preproc_rfu',
        'analysis_absd_orig', 'basesub_absd_orig', 'original_rfu_cfx',
        'strep_plus2_analysis_absd', 'ml_analysis_absd'
    i_pcrd : str
        PCRD 이름
    i_channel : str
        채널 이름 (예: 'FAM', 'HEX', 'ROX')
    i_temperature : float
        온도 값 (°C)
    colors : list, optional
        그래프에 사용할 색상 목록. 기본값은 None
    mudt : bool, optional
        MuDT 데이터 여부. True일 경우 preproc_rfu 사용, False일 경우 original_rfu 사용.
        기본값은 False
    
    Returns
    -------
    None
        6개의 패널로 구성된 서브플롯을 화면에 표시한다:
        - [After BPN] Raw RFU: BPN 정규화 후 원본 RFU 신호
        - [CFX/After BPN] Baseline-Subtracted/Preproc RFU: CFX 또는 전처리된 RFU 신호
        - [DSP] Original ABSD: DSP 알고리즘으로 베이스라인 차감된 신호
        - [Auto] Baseline-Subtracted RFU: 자동 베이스라인 차감 알고리즘 결과
        - [Strep+2] Baseline-Subtracted RFU: Strep+2 알고리즘 결과
        - [ML] Baseline-Subtracted RFU: 머신러닝 기반 알고리즘 결과
    
    Examples
    --------
    >>> import pandas as pd
    >>> from cp_visualization import plot_baseline_subtractions
    >>> # 데이터 로드
    >>> data = pd.read_parquet('pcr_results.parquet')
    >>> # 특정 PCR 실험, 채널, 온도에 대한 시각화
    >>> plot_baseline_subtractions(data, 'Experiment1', 'FAM', 60.0)
    >>> # MuDT 데이터 시각화
    >>> plot_baseline_subtractions(data, 'MuDT_Exp', 'ROX', 63.5, mudt=True)
    
    See Also
    --------
    plot_single_well : 단일 웰에 대한 상세 시각화
    """
    
    # 베이스라인 시각화에 필요한 데이터 준비
    filtered_data, titles, data_keys, original_rfu_min_mean, preproc_rfu_min_mean = (
        prepare_baseline_data(i_data, i_pcrd, i_channel, i_temperature, mudt)
    )
    # 메트릭 계산
    (
        error_metrics_dict, 
        metric_min, 
        metric_max, 
        original_rfu_after_bpn_limits, 
        preproc_rfu_after_bpn_limits, 
        limits
    ) = calculate_error_metrics(filtered_data)
    
    # 시각화를 위한 서브플롯 생성
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    # 패널 그리기에 필요한 정보 모음
    limits_info = (
        metric_min, 
        metric_max, 
        original_rfu_min_mean, 
        preproc_rfu_min_mean,
        original_rfu_after_bpn_limits, 
        preproc_rfu_after_bpn_limits, 
        limits
    )
    
    # 각 패널 그리기
    for (i, j), ax, title, key in zip(np.ndindex(axs.shape), axs.flat, titles, data_keys):
        panel_info = (i, j, title, key)
        plot_single_baseline_panel(
            ax, 
            filtered_data, 
            panel_info, 
            error_metrics_dict, 
            limits_info, 
            mudt
        )
    
    # 전체 레이아웃 설정
    pydsp_version = get_package_details().get('pydsp')
    fig.suptitle(
        f'PCRD: {i_pcrd}\n'
        f'pydsp Version: {pydsp_version}, '
        f'Channel: {i_channel}, Temperature: {i_temperature}, '
        f'The Number of Signals: {filtered_data.shape[0]}', 
        ha='left', 
        x=0.02, 
        fontsize=15
    )
    
    plt.tight_layout()
    plt.show()


###########################################
# 싱글 웰 시각화
########################################### 

def prepare_single_well_data(i_data, i_pcrd, i_channel, i_temperature, i_well, mudt=True):
    """
    단일 웰 시각화를 위한 데이터 준비
    
    Args:
        i_data: 원본 데이터프레임
        i_pcrd: PCRD 이름
        i_channel: 채널 이름
        i_temperature: 온도 값
        i_well: 웰 번호
        mudt: MuDT 데이터 여부
        
    Returns:
        tuple: (필터링된 데이터, raw_rfu, 컬럼 정보, 사이클 정보)
    """
    filtered_data = i_data.query("`name`==@i_pcrd & `channel` == @i_channel & "
                              "`temperature`==@i_temperature & `well` == @i_well").copy()
    
    # raw_rfu 설정
    raw_rfu = 'preproc_rfu' if mudt else 'original_rfu'
    
    # 컬럼 설정
    columns = [
        ('dsp_correction_fit', raw_rfu, 'analysis_absd_orig'),
        ('basesub_correction_fit', raw_rfu, 'basesub_absd_orig'),
        ('cfx_correction_fit', raw_rfu, 'original_rfu_cfx'),
        ('strep_plus2_correction_fit', raw_rfu, 'strep_plus2_analysis_absd'),
        ('ml_correction_fit', raw_rfu, 'ml_analysis_absd'),
        ('dsp_corrected_rfu', raw_rfu, 'analysis_rd_diff'),
        ('basesub_corrected_rfu', raw_rfu, 'basesub_rd_diff'),
        ('strep_plus2_corrected_rfu', raw_rfu, 'strep_plus2_analysis_rd_diff'),
        ('ml_corrected_rfu', raw_rfu, 'strep_plus2_analysis_rd_diff')
    ]
    
    raw_scale_columns = [
        raw_rfu, 'dsp_correction_fit', 'basesub_correction_fit', 
        'cfx_correction_fit', 'strep_plus2_correction_fit', 'ml_correction_fit'
    ]
    
    correction_scale_columns = [
        raw_rfu, 'dsp_corrected_rfu', 'basesub_corrected_rfu', 
        'strep_plus2_corrected_rfu', 'ml_corrected_rfu'
    ]
    
    subtract_scale_columns = [
        'basesub_absd_orig', 'original_rfu_cfx', 'analysis_absd_orig',
        'strep_plus2_analysis_absd', 'ml_analysis_absd'
    ]
    
    titles = [
        '[DSP] Algorithm Performance', '[Auto] Algorithm Performance', 
        '[CFX] Algorithm Performance', '[Strep+2] Algorithm Performance', 
        '[ML] Algorithm Performance', '[DSP] Fitting Performance', 
        '[Auto] Fitting Performance', '[CFX] Fitting Performance', 
        '[Strep+2] Fitting Performance', '[ML] Fitting Performance',
        '[DSP] Baseline-Subtracted RFU', '[Auto] Baseline-Subtracted RFU', 
        '[CFX] Baseline-Subtracted RFU', '[Strep+2] Baseline-Subtracted RFU', 
        '[ML] Baseline-Subtracted RFU'
    ]
    
    preprocessing_columns = [
        ('dsp_correction_fit', raw_rfu),
        ('basesub_correction_fit', raw_rfu),
        ('cfx_correction_fit', raw_rfu),
        ('strep_plus2_correction_fit', raw_rfu),
        ('ml_correction_fit', raw_rfu)
    ]
    
    correction_columns = [
        ('dsp_corrected_rfu', raw_rfu, 'analysis_scd_fit'),
        ('basesub_corrected_rfu', raw_rfu, 'basesub_scd_fit'),
        (raw_rfu, raw_rfu, 'cfx_correction_fit'),
        ('strep_plus2_corrected_rfu', raw_rfu, 'strep_plus2_analysis_scd_fit'),
        ('ml_corrected_rfu', raw_rfu, 'ml_baseline_fit')
    ]
    
    subtracted_columns = [
        ('analysis_absd_orig'),
        ('basesub_absd_orig'),
        ('original_rfu_cfx'),
        ('strep_plus2_analysis_absd'),
        ('ml_analysis_absd')
    ]
    
    # mudt에 따른 컬럼 필터링
    if mudt:
        columns = [col for col in columns if 'cfx' not in col[0].lower()]
        raw_scale_columns = [col for col in raw_scale_columns if 'cfx' not in col.lower()]
        correction_scale_columns = [col for col in correction_scale_columns 
                                  if not hasattr(col, 'lower') or 'cfx' not in col.lower()]
        subtract_scale_columns = [col for col in subtract_scale_columns if 'cfx' not in col.lower()]
        titles = [col for col in titles if 'cfx' not in col.lower()]
        preprocessing_columns = [col for col in preprocessing_columns 
                               if not any('cfx' in item.lower() for item in col)]
        correction_columns = [col for col in correction_columns 
                            if not any('cfx' in item.lower() for item in col)]
        subtracted_columns = [col for col in subtracted_columns if 'cfx' not in col.lower()]
    
    # 사이클 정보 계산
    rfu = filtered_data[raw_rfu].values[0]
    cycle_length = len(rfu)
    cycle = list(range(cycle_length))
    
    # 배열 데이터 처리
    array_columns = [col for col in filtered_data.columns 
                    if filtered_data[col].apply(lambda x: isinstance(x, np.ndarray)).any()]
    for column in array_columns:
        filtered_data[column] = filtered_data[column].apply(
            lambda x: np.zeros(cycle_length) if x is None or x.size == 0 else x)
    
    # 새 컬럼 생성
    for new_column, column_a, column_b in columns:
        filtered_data[new_column] = list(map(
            lambda x, y: [a - b for a, b in zip(x, y)], 
            filtered_data[column_a], filtered_data[column_b]))
    
    column_info = {
        'raw_scale': raw_scale_columns,
        'correction_scale': correction_scale_columns,
        'subtract_scale': subtract_scale_columns,
        'titles': titles,
        'preprocessing': preprocessing_columns,
        'correction': correction_columns,
        'subtracted': subtracted_columns
    }
    
    cycle_info = {
        'cycle': cycle,
        'cycle_length': cycle_length
    }
    
    return filtered_data, raw_rfu, column_info, cycle_info

def calculate_limits(i_data, column_info):
    """
    시각화를 위한 축 제한값 계산
    
    Args:
        i_data: 필터링된 데이터프레임
        column_info: 컬럼 정보 딕셔너리
        
    Returns:
        tuple: (raw_limits, correction_limits, subtraction_limits)
    """
    # 데이터 추출
    raw_scale_data = i_data[column_info['raw_scale']]
    correction_scale_data = i_data[column_info['correction_scale']]
    subtract_scale_data = i_data[column_info['subtract_scale']]
    
    # 한계값 계산
    raw_limits = find_global_extremes(raw_scale_data)
    correction_limits = find_global_extremes(correction_scale_data)
    subtraction_limits = find_global_extremes(subtract_scale_data)
    
    return raw_limits, correction_limits, subtraction_limits

def draw_single_well_panel(ax, i_data, i, j, title, column_info, cycle_info, limits):
    """
    단일 웰 시각화의 패널 그리기
    
    Args:
        ax: matplotlib 축 객체
        i_data: 필터링된 데이터프레임
        i, j: 서브플롯 위치
        title: 패널 제목
        column_info: 컬럼 정보 딕셔너리
        cycle_info: 사이클 정보 딕셔너리
        limits: 축 제한값 튜플
    """
    cycle = cycle_info['cycle']
    raw_limits, correction_limits, subtraction_limits = limits
    
    # 행(row) 기준으로 그래프 그리기
    if i == 0:  # 첫 번째 행: 알고리즘 성능
        ax.plot(cycle, i_data[column_info['preprocessing'][j][0]].values[0], 
                'red', alpha=0.5, label="Correction & Fit")
        ax.plot(cycle, i_data[column_info['preprocessing'][j][1]].values[0], 
                'k-', alpha=0.5, label="Raw Data")
        set_ylim(ax, raw_limits, 0)
    
    elif i == 1:  # 두 번째 행: 피팅 성능
        ax.plot(cycle, i_data[column_info['correction'][j][0]].values[0], 
                'b--', alpha=1, label="Correction")
        ax.plot(cycle, i_data[column_info['correction'][j][1]].values[0], 
                'k-', alpha=0.5, label="Raw Data")
        ax.plot(cycle, i_data[column_info['correction'][j][2]].values[0], 
                'red', alpha=0.5, label="Fitted Data")
        set_ylim(ax, correction_limits, 1)
    
    else:  # 세 번째 행: 베이스라인 차감 RFU
        ax.plot(cycle, i_data[column_info['subtracted'][j]].values[0], 
                'k-', alpha=0.5, label="Subtracted Data")
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2)
        set_ylim(ax, subtraction_limits, 2)
    
    ax.set_title(title)
    ax.legend()

def set_ylim(ax, limits, index):
    """y축 제한값을 설정합니다.
    
    데이터의 최대/최소값에 따라 적절한 y축 범위를 설정합니다.
    음수 값과 양수 값에 대해 다른 비율을 적용합니다.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        제한값을 설정할 matplotlib 축 객체
    limits : list
        [max, min] 형태의 한계값 리스트
    index : int
        행 인덱스 (시각화 타입 식별용)
    
    Returns
    -------
    None
        matplotlib 축 객체의 ylim 속성을 직접 수정합니다.
    
    Notes
    -----
    음수 최대값은 0.95배, 양수 최대값은 1.05배로 확장합니다.
    음수 최소값은 1.05배, 양수 최소값은 0.95배로 확장합니다.
    """
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

def plot_single_well(i_data, i_pcrd, i_channel, i_temperature, i_well, mudt=True, colors=None):
    '''
    특정 웰의 베이스라인 차감 결과를 시각화하여 알고리즘 성능을 비교
    
    Args:
        i_data: 원본 데이터프레임
        i_pcrd: PCRD 이름
        i_channel: 채널 이름
        i_temperature: 온도 값
        i_well: 웰 번호
        mudt: MuDT 데이터 여부, 기본값은 True
        colors: 커스텀 색상, 기본값은 None
    
    Returns:
        None: 서브플롯을 화면에 표시
    '''
    # 데이터 준비
    filtered_data, raw_rfu, column_info, cycle_info = prepare_single_well_data(
        i_data, i_pcrd, i_channel, i_temperature, i_well, mudt)
    
    # 축 제한값 계산
    limits = calculate_limits(filtered_data, column_info)
    
    # 서브플롯 생성
    if mudt:
        fig, axs = plt.subplots(3, 4, figsize=(18, 12))
    else:
        fig, axs = plt.subplots(3, 5, figsize=(18, 12))
    
    # 각 패널 그리기
    for (i, j), ax, title in zip(np.ndindex(axs.shape), axs.flat, column_info['titles']):
        draw_single_well_panel(ax, filtered_data, i, j, title, column_info, 
                              cycle_info, limits)
    
    # 전체 레이아웃 설정
    pydsp_version = get_package_details().get('pydsp')
    fig.suptitle(f'PCRD: {i_pcrd}\npydsp Version: {pydsp_version}, '
                f'Channel: {i_channel}, Temperature: {i_temperature}, Well: {i_well}', 
                ha='left', x=0.02, fontsize=15)
    
    plt.tight_layout()
    plt.show()
