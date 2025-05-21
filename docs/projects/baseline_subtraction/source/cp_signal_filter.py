"""
PCR 신호 필터링 및 노이즈 감지 모듈

이 모듈은 PCR(Polymerase Chain Reaction) 신호 데이터에서 노이즈를 감지하고
분석하기 위한 다양한 알고리즘을 제공합한다. 주요 기능으로는 이상치 감지,
화이트 노이즈 검정, 자기상관 계산 등이 포함된다.

주요 함수:
- detect_noise_naively: 기본적인 Z-점수 기반 노이즈 감지
- detect_noise_naively_ver2: 처음 10 사이클을 제외한 Z-점수 기반 노이즈 감지
- detect_noise_naively_ywj1: BPN(Baseline Percentage Normalization) 기반 노이즈 측정
- detect_noise_naively_pbg: 동적 임계값을 사용한 노이즈 감지
- detect_noise_naively_kkm: 선형 회귀 기반 노이즈 감지
- compute_autocorrelation: 신호의 자기상관 계산
- test_white_noise: 화이트 노이즈 검정
"""

# Config
# Analysis Preparation
import numpy as np
from scipy.stats import chi2  # Chi-squared distribution for white noise test

# Noise Naive Detection
def detect_noise_naively(i_signal, i_threshold=1.28):
    '''
    노이즈 신호를 감지한다.
    
    Args:
    - i_signal: 리스트 형태의 RFU 신호 데이터
    - i_threshold: Z 점수 임계값 (기본값 1.28)
    
    Returns:
    - o_result: [이상치 정도, 이상치 목록, Z score 목록, 이상치 존재 여부]를 
      포함하는 리스트
    '''
    
    if i_signal.size == 0:  
        return [False, 0, []]  # 빈 신호에 대한 기본값 반환
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)

    if std == 0:  # NAN 방지
        return [False, 0, []]

    z_scores = [(point - mean) / std for point in i_signal]
    outliers = [score * std + mean for score in z_scores if abs(score) > i_threshold]
    outliers_boolean = [abs(score) > i_threshold for score in z_scores]
    outlier_existence = 1 if any(outliers_boolean) else 0
    outlierness_metric_sum = sum(abs(score) for score in z_scores 
                               if abs(score) > i_threshold)
    outlierness_metric = outlierness_metric_sum / len(outliers) if len(outliers) != 0 else 0
    o_result = [outlierness_metric, outliers, z_scores, outlier_existence]
    return o_result


def detect_noise_naively_ver2(i_signal, i_threshold=1.28):
    '''
    노이즈 신호를 감지한다. 처음 10 사이클을 제외하고 분석한다.
    
    Args:
    - i_signal: 리스트 형태의 RFU 신호 데이터
    - i_threshold: Z 점수 임계값 (기본값 1.28, 참고값 1.65)

    Returns:
    - o_result: [이상치 존재 여부, 이상치 정도, 이상치 목록]을 포함하는 리스트로
      [불리언, 실수, 이상치 리스트] 형태로 반환한다.
    '''
    i_signal = i_signal[10:]
    if i_signal.size == 0:  # 신호가 비어있는지 확인
        return [False, 0, []]  # 빈 신호에 대한 기본값 반환
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)

    if std == 0:  # 0으로 나누기 방지
        return [False, 0, []]

    z_scores = [(point - mean) / std for point in i_signal]
    outliers = [score for score in z_scores if abs(score) > i_threshold]
    outliers_boolean = [abs(score) > i_threshold for score in z_scores]
    outlier_existence = 1 if any(outliers_boolean) else 0
    outlierness_metric_sum = sum(abs(score) for score in z_scores 
                               if abs(score) > i_threshold)
    outlierness_metric = outlierness_metric_sum / len(outliers) if len(outliers) != 0 else 0
    o_result = [outlier_existence, outlierness_metric, outliers]
    return o_result


def detect_noise_naively_ywj1(i_signal):
    '''
    BPN(Baseline Percentage Normalization) 후 |RFU_i|의 합계를 계산한다.
        
    Args:
    - i_signal: 리스트 형태의 RFU 신호 데이터

    Returns:
    - o_result: [노이즈 측정값, 잔차, BPN 적용 후 신호]를 포함하는 리스트로
      [실수, 리스트, 리스트] 형태로 반환한다.
    '''
    
    if i_signal.size == 0:  # 신호가 비어있는지 확인
        return [False, 0, []]  # 빈 신호에 대한 기본값 반환
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)

    after_bpn = [(point / mean) * 100 for point in i_signal]  # 기준값 100으로 정규화된 신호
    mean_adjusted = np.mean(after_bpn)
    residuals = [(rfu - mean_adjusted) for rfu in after_bpn]
    noise_metric = sum(abs(rfu) for rfu in residuals)
    
    o_result = [noise_metric, residuals, after_bpn]
    return o_result
    

def detect_noise_naively_pbg(i_signal):
    '''
    처음 10 사이클을 제외하고 평균 대비 편차의 상위 20%를 기준으로 한 동적 임계값을 
    사용하여 노이즈 신호를 감지한다.
    
    Args:
    - i_signal: 리스트 형태의 RFU 신호 데이터
    
    Returns:
    - o_result: [이상치 정도, 이상치 목록, 점수, 이상치 존재 여부]를 포함하는 리스트로
      [실수, 리스트, 리스트, 불리언] 형태로 반환한다.
    '''
    
    if len(i_signal) == 0:  # 신호가 비어있는지 확인
        return [False, 0, []]  # 빈 신호에 대한 기본값 반환
    
    mean = np.mean(i_signal)
    std = np.std(i_signal)
    
    if std == 0:  # 0으로 나누기 방지
        return [False, 0, []]
    
    scores = [(point - mean) for point in i_signal]  # 평균 대비 편차 계산
    threshold = np.percentile(scores, 80)  # 상위 20% 점수에 대한 동적 임계값
    
    outliers = [score + mean for score in scores if abs(score) > threshold]  # 이상치 식별
    outlier_existence = bool(outliers)  # 이상치 존재 여부
    outlierness_metric = sum(abs(score) - threshold for score in outliers) / len(outliers) if outliers else 0  # 이상치 정도 계산
    
    return [outlierness_metric, outliers, scores, outlier_existence]


def detect_noise_naively_kkm(i_signal):
    '''
    원본 신호의 RFU 값에서 선형 회귀를 차감하고 잔차의 절대값 합계를 계산하여
    노이즈 신호를 감지한다.
    
    Args:
    - i_signal: 리스트 형태의 RFU 신호 데이터
    
    Returns:
    - o_result: [측정값, 잔차, 기울기, 절편]을 포함하는 리스트로
      [실수, 리스트, 실수, 실수] 형태로 반환한다.
    '''

    if len(i_signal) == 0: 
        return [False, 0, []]
        
    mean = np.mean(i_signal)
    after_bpn = [(point / mean) * 100 for point in i_signal]  # 평균 대비 백분율로 계산
    x = list(range(1, len(after_bpn) + 1))
    slope = np.cov(x, after_bpn, bias=True)[0, 1] / np.var(x, ddof=0)
    intercept = np.mean(after_bpn) - slope * np.mean(x)
    x_values = np.array(x)
    linear_fit = slope * x_values + intercept
    residuals = after_bpn - linear_fit
    metric = sum(abs(residuals))
    
    return [metric, residuals, slope, intercept]
    

# 화이트 노이즈 신호 감지
def compute_autocorrelation(i_signal, i_lag):
    '''
    신호의 자기상관(autocorrelation) 값을 계산한다.

    Args:
    - i_signal: PCR 신호 데이터
    - i_lag: 지연 윈도우(lag window)

    Returns:
    - o_result: 자기상관 값
    '''
    mean_x = np.mean(i_signal)
    denominator = np.sum((i_signal - mean_x) ** 2)
    numerator = np.sum((i_signal[:-i_lag] - mean_x) * (i_signal[i_lag:] - mean_x))
    o_result = numerator / denominator 
    return o_result


def test_white_noise(i_signal, i_max_lag):
    '''
    신호가 화이트 노이즈인지 검정한다.

    Args:
    - i_signal: PCR 신호 데이터
    - i_max_lag: 최대 지연(lag) 값

    Returns:
    - o_result: [검정 통계량, p-값]을 포함하는 리스트
    '''
    n = len(i_signal)
    statistic = 0
    for lag in range(1, i_max_lag + 1):
        rho = compute_autocorrelation(i_signal, lag)
        statistic += (n * (n + 2) * rho ** 2) / (n - lag)
    p_value = 1 - chi2.cdf(statistic, i_max_lag)
    o_result = [statistic, p_value]
    return o_result