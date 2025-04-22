"""
PCR 신호 데이터 기반 베이스라인 최적화 예측 모듈

이 모듈은 다양한 수학적 방법을 사용하여 PCR 신호 데이터의 기저선(baseline)을 예측하고
최적화하는 알고리즘을 제공

주요 기능:
- 신경망 기반 신호 예측 (compute_simple_nn)
- 공액 경사법(Conjugate Gradient Method)을 통한 최적화
- 다양한 기저 함수를 활용한 비선형 변환 및 예측
- 데이터 정규화 및 역정규화 
- 시계열 데이터 예측 및 처리

작성자: Kwangmin Kim
날짜: 2024-07
"""

# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

# simple nueral network
def compute_simple_nn(y):
    """
    신경망 기반 신호 예측 함수
    
    간단한 다층 신경망을 사용하여 1D 입력 신호에 대한 예측을 수행
    입력 신호를 정규화하고, 4개 레이어(8-8-2-1)로 구성된 신경망을 통해 학습 및 예측
    
    매개변수:
        y (array-like): 입력 신호 데이터
        
    반환값:
        list: 예측된 신호 값들의 리스트
    """
    x = np.linspace(0, 1, len(y))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    model = Sequential([
        Dense(8, activation='relu', input_dim=1),
        Dense(8, activation='softplus'), #softplus
        Dense(2, activation='selu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    history = model.fit(x_norm, y_norm, epochs=500, verbose=0)
    y_pred = model.predict(x_norm)
    y_pred_denorm = y_pred * (y.max() - y.min()) + y.min()
    o_result = [item for sublist in y_pred_denorm for item in sublist]

    return o_result

def f_alpha(alpha, fun, x, s, args=()):
    """
    최적화를 위한 1D line search 함수
    
    주어진 방향으로 이동했을 때의 목적 함수 값을 계산
    
    매개변수:
        alpha: 1D 독립 변수(스텝 크기)
        fun: 원래 목적 함수
        x: 시작점
        s: 1D 검색 방향
        args: 목적 함수에 전달되는 추가 인자들의 튜플
        
    반환값:
        float: 현재 alpha 값에서의 목적 함수 값
    """
    x_new = x + alpha * s
    
    return fun(x_new, *args)

def search_golden_section(fun, dfun, x, search_direction, args=(), initial_delta=1.0e-2, tolerance=1e-15):
    """
    황금 분할 검색 알고리즘
    
    1차원 방향에서의 최적 스텝 크기(alpha)를 찾기 위한 황금 분할 검색 방법을 구현합니다.
    https://en.wikipedia.org/wiki/Golden-section_search 참조
    
    매개변수:
        fun: 원래 목적 함수
        dfun: 목적 함수의 기울기(이 구현에서는 사용하지 않음)
        x: 시작점
        search_direction: 1D 검색 방향
        args: 목적 함수에 전달되는 추가 인자들의 튜플
        initial_delta: 초기 불확실성 구간을 결정하는 초기 추정 간격
        tolerance: 정지 기준
        
    반환값:
        tuple: 최적의 alpha 값을 포함하는 1-요소 튜플
    """
    golden_ratio = (np.sqrt(5) + 1) / 2
        
    left_bound = 0.
    left_value = f_alpha(left_bound, fun, x, search_direction, args)
    mid_point = initial_delta
    mid_value = f_alpha(mid_point, fun, x, search_direction, args)
    while left_value < mid_value:
        initial_delta = 0.1 * initial_delta
        mid_point = initial_delta
        mid_value = f_alpha(mid_point, fun, x, search_direction, args)
    
    step = 1
    right_bound = mid_point + initial_delta * (golden_ratio**step)
    right_value = f_alpha(right_bound, fun, x, search_direction, args)
    while mid_value > right_value:
        left_bound = mid_point
        mid_point = right_bound
        left_value = mid_value
        mid_value = right_value
        
        step += 1
        right_bound = mid_point + initial_delta * (golden_ratio**step)
        right_value = f_alpha(right_bound, fun, x, search_direction, args)

    inner_point = left_bound + (right_bound - left_bound) / golden_ratio
    inner_value = f_alpha(inner_point, fun, x, search_direction, args)
    
    while abs(mid_point - inner_point) > tolerance:
        if f_alpha(mid_point, fun, x, search_direction, args) < f_alpha(inner_point, fun, x, search_direction, args):
            right_bound = inner_point
        else:
            left_bound = mid_point

        # we recompute both c and d here to avoid loss of precision 
        # which may lead to incorrect results or infinite loop
        mid_point = right_bound - (right_bound - left_bound) / golden_ratio
        inner_point = left_bound + (right_bound - left_bound) / golden_ratio

    return ((right_bound + left_bound) / 2,)

def compute_conjugate_gradient(
    objective_func, 
    gradient_func, 
    initial_point, 
    args=(), 
    epsilon=1.0e-7, 
    max_iterations=2500, 
    verbose=False, 
    callback=None
):
    """
    공액 경사법(CGM)을 사용한 함수 최적화
    
    주어진 함수와 그 기울기를 사용하여 공액 경사법으로 최적화를 수행
    
    매개변수:
        objective_func : 최적화 할 함수
        gradient_func  : 최적화할 함수의 도함수  
        initial_point  : 초기값
        args          : objective_func(x, *args)로 함수를 호출하기 위한 추가 매개변수
        epsilon       : 정기 기준
        max_iterations: 최대 반복수
        verbose       : 실행 정보를 텍스트로 출력
        callback      : initial_point에 대한 기록을 위한 콜백함수
    """

    if verbose:
        print("#####START OPTIMIZATION#####")
        print("INIT POINT : {}, dtype : {}".format(initial_point, initial_point.dtype))

    current_point = initial_point
    
    for iteration in range(max_iterations):
        gradient = gradient_func(current_point, *args)
        
        if np.linalg.norm(gradient) < epsilon:
            if verbose:
                print("Stop criterion break Iter: {:5d}, x: {}".format(
                    iteration, current_point))
                print("\n")
            break

        if iteration == 0:
            direction = -gradient
        else:
            beta = (np.linalg.norm(gradient) / np.linalg.norm(prev_gradient))**2
            direction = -gradient + beta * direction
        
        step_size = search_golden_section(
            objective_func, gradient_func, current_point, direction, args=args)[0]
        current_point = current_point + step_size * direction
        prev_gradient = gradient.copy()

        if callback:
            callback(current_point)

    else:
        print("Stop max iter:{:5d} x:{}".format(iteration, current_point))

    return current_point


def haar_wavelet(x):
    """
    Haar Wavelet Basis Function
    
    정의역 [0,1)에서 단순한 Haar 웨이블릿 함수를 계산
    입력 값이 [0,0.5) 구간에 있으면 1, [0.5,1) 구간에 있으면 -1,
    그 외의 경우 0을 반환
    
    매개변수:
        x (float): 입력 값
        
    반환값:
        int: 입력 값에 대한 Haar 웨이블릿 함수 값 (1, -1, 또는 0)
    """
    if 0 <= x < 0.5:
        return 1
    elif 0.5 <= x < 1:
        return -1
    else:
        return 0

def piecewise_basis(x):
    """
    Piecewise Basis Function
    
    구간에 따라 다른 함수를 적용하는 조각별 기저 함수
    - x < 0.3: 이차 함수 (x^2)
    - 0.3 <= x < 0.6: 선형 함수 (3x + 1)
    - x >= 0.6: 사인 함수 (sin(2πx))
    
    매개변수:
        x (float): 입력 값
        
    반환값:
        float: 입력 값에 대한, 구간에 따라 다른 함수 값
    """
    if x < 0.3:
        return x**2
    elif 0.3 <= x < 0.6:
        return 3 * x + 1
    else:
        return np.sin(2 * np.pi * x)

def compute_phi(X):
    """
    다양한 기저 함수를 적용하여 특성 변환 행렬 생성
    
    입력 데이터에 대해 사인, 코사인, haar_wavelet, piecewise_basis 등
    다양한 비선형 변환을 적용하여 특성 공간을 확장
    
    매개변수:
        X (array-like): 변환할 입력 데이터
        
    반환값:
        array: 각 기저 함수가 적용된 변환된 데이터 행렬
              각 행은 X의 하나의 데이터 포인트에 대응하며,
              각 열은 다른 기저 함수에 의한 변환 결과
    """
    phi_functions = [
        lambda x: np.sin(x),  # Example sine function for variation
        haar_wavelet,         # Haar Wavelet function
        piecewise_basis,      # Piecewise Basis function
        lambda x: np.cos(x)   # Example cosine function for variation
    ]
    
    PHI = np.array([[phi(x) for phi in phi_functions] for x in X])
    return PHI

# Regularization
    
#reg_intercept = True
#lamda = 0.01

def compute_l1_loss(w, P, x, y):
    """
    L1 정규화를 적용한 손실 함수 계산
    
    목적 함수 J(w)는 오차 제곱합(MSE)과 L1 정규화 항의 합으로 구성
    J(w) = (1/2) * sum_{n=1}^{N} {y(x_n,w) - t_n}^2 + (λ/2) * sum|w_i|
    
    매개변수:
        w (array): 가중치 행렬, 형태는 (M,) 또는 (M,K)
                  여기서 K는 가중치 벡터 개수, M은 기저함수 개수
        P (int): 다항식 차수
        x (array): 입력 데이터, 형태는 (N,)
        y (array): 타겟 값, 형태는 (N,)
        
    반환값:
        float: 계산된 손실 함수 값
    """
    PHI = np.array([ x**i for i in range(P+1) ]).T 
    y_pred = np.dot(w.T, PHI.T) #(N,) or (K,N)
        
    if not reg_intercept :
        reg =  (lamda/2.)*(np.abs(w[1:]).sum(axis=0)) 
    else :
        reg =  (lamda/2.)*(np.abs(w).sum(axis=0))   
    
    # 기존 MSE에 reg를 더해서 리턴
    return 0.5*(( (y - y_pred)**2 ).sum(axis=-1)) + reg

    #################################################

def compute_l1_gradent(w, P, x, y):
    """
    This function computes the analytic gradient of the objective function 
    with L2 regularization.

    w: 가중치 행렬, (M,) or (M,K) 여기서 K는 가중치 벡터 개수, 
       M은 기저함수 개수 또는 데이터 차원
    P: polynomial degree, scalar
    x, y : data for error function eval.
      x: data, (N,M)
      y: target, (N,)
    """
    PHI = np.array([ x**i for i in range(P+1) ]).T
    g = np.dot(w.T, np.dot(PHI.T, PHI) ) - np.dot(y.T, PHI) # (1,M) or (K,M)
    g_reg = (lamda/2.) * np.sign(w) 

    if not reg_intercept :
        g_reg[0] = 0
        
    g = g + g_reg.T
    return g

def compute_l2_loss(w, P, x, y):
    """
    L2 정규화를 적용한 손실 함수 계산
    
    목적 함수 J(w)는 오차 제곱합(MSE)과 L2 정규화 항의 합으로 구성됩니다:
    J(w) = (1/2) * sum_{n=1}^{N} {y(x_n,w) - t_n}^2 + (λ/2) * ||w||^2
    
    매개변수:
        w (array): 가중치 행렬, 형태는 (M,) 또는 (M,K)
                  여기서 K는 가중치 벡터 개수, M은 기저함수 개수
        P (int): 다항식 차수
        x (array): 입력 데이터, 형태는 (N,)
        y (array): 타겟 값, 형태는 (N,)
        
    반환값:
        float: 계산된 손실 함수 값
    """
    PHI = np.array([ x**i for i in range(P+1) ]).T 
    y_pred = np.dot(w.T, PHI.T) 
 
    if not reg_intercept:
      reg = (lamda*0.5)*np.linalg.norm(w[1:],axis=0)**2
    else: 
      reg = (lamda *0.5)*np.linalg.norm(w,axis=0)**2
  
    return 0.5*(( (y - y_pred)**2 ).sum(axis=-1)) + reg
    
def compute_l2_gradent(w, P, x, y):
    """
    This function computes the analytic gradient of the objective function 
    with L2 regularization.

    w: 가중치 행렬, (M,) or (M,K) 여기서 K는 가중치 벡터 개수, 
       M은 기저함수 개수 또는 데이터 차원
    P: polynomial degree, scalar
    x, y : data for error function eval.
      x: data, (N,M)
      y: target, (N,)
    """
    PHI = np.array([ x**i for i in range(P+1) ]).T
    g = np.dot(w.T, np.dot(PHI.T, PHI) ) - np.dot(y.T, PHI)
    g_reg = lamda*w # (M,K)

    if not reg_intercept :
        g_reg[0] = 0

    g = g + g_reg.T
    
    return g

def plot_result(X_train, y_train):
    """
    학습 데이터와 예측 결과를 시각화하는 함수
    
    2x2 서브플롯으로 구성된 그래프를 생성하여 다양한 차수의 다항식 피팅 결과를 표시합니다.
    각 서브플롯은 동일한 x, y 축 범위를 공유하며, 학습 데이터 포인트와 예측 곡선을 함께 표시합니다.
    
    매개변수:
        X_train (array-like): 학습에 사용된 입력 데이터
        y_train (array-like): 학습에 사용된 타겟 데이터
        
    반환값:
        None: 그래프를 화면에 표시만 하고 반환값은 없음
    """
    fig, ax = plt.subplots(figsize=(15, 10), nrows=2, ncols=2, 
                          sharex='all', sharey='all')

    j = 0

    for P in Ps:
        p, q = divmod(j, 2)
        X = np.array([x_raw**i for i in range(P+1)])  

        y = (W[j].reshape(-1, 1) * X).sum(axis=0)

        ax[p, q].xaxis.set_tick_params(labelsize=18)
        ax[p, q].yaxis.set_tick_params(labelsize=18)
        ax[p, q].set_xlabel('$x$', fontsize=20)
        ax[p, q].set_ylabel('$y$', fontsize=20)
        ax[p, q].grid(False)

        ax[p, q].plot(X_train, y_train, 'o', markersize=8, color='k')
        ax[p, q].legend(loc='upper right', fontsize=18)

        j += 1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax_ in ax.flat:
        ax_.label_outer()

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()

# Characteristic Equation

def log_normalize(X):
    """
    로그 스케일 기반 데이터 정규화
    
    입력 데이터를 로그 변환 후 [0, 1] 범위로 정규화합니다.
    음수 값이 있는 경우에도 처리 가능하도록 오프셋을 적용합니다.
    
    매개변수:
        X (array-like): 정규화할 입력 데이터
        
    반환값:
        tuple: (정규화된 데이터, 최소값, 최대값, 원본 최소값)
              역정규화에 필요한 정보를 포함합니다.
    """
    X_min_original = np.min(X)
    # 모든 값을 양수로 만들기 위해 최소값을 더함
    X_positive = X - X_min_original + 1e-10  # 작은 값을 더해 로그의 0 입력을 방지

    # 로그 변환
    X_log = np.log(X_positive)

    # 정규화
    X_min = np.min(X_log)
    X_max = np.max(X_log)
    X_normalized = (X_log - X_min) / (X_max - X_min)

    return X_normalized, X_min, X_max, X_min_original

def log_denormalize(X_normalized, X_min, X_max, X_min_original):
    """
    로그 정규화된 데이터의 역변환
    
    log_normalize로 정규화된 데이터를 원래 스케일로 되돌립니다.
    
    매개변수:
        X_normalized: 정규화된 데이터
        X_min: 로그 변환 후의 최소값
        X_max: 로그 변환 후의 최대값
        X_min_original: 원본 데이터의 최소값
        
    반환값:
        array: 원래 스케일로 복원된 데이터
    """
    # 역정규화
    X_log_restored = X_normalized * (X_max - X_min) + X_min

    # 지수 변환
    X_restored = np.exp(X_log_restored)

    # 원래 스케일로 복원
    X_restored = X_restored + X_min_original - 1e-10

    return X_restored

def apply_basis_functions(X):
    """
    비선형 기저 함수 변환
    
    입력 데이터에 다양한 비선형 기저 함수를 적용하여 특성 공간을 확장합니다.
    다항식, 지수, 로그, 역수, ReLU 변형 등의 함수를 적용합니다.
    
    매개변수:
        X (array-like): 변환할 입력 데이터
        
    반환값:
        array: 각 기저 함수가 적용된 변환된 데이터 행렬
    """
    phi1 = lambda x: x**0 
    phi2 = lambda x: x    
    phi3 = lambda x: x**2 
    phi4 = lambda x: x**3 
    phi5 = lambda x: np.exp(x) - 1  # 지수 증가
    phi6 = lambda x: np.exp(-x)     # 지수 감소
    phi7 = lambda x: np.log(x + 1)  # 로그 함수 (느린 증가)
    phi8 = lambda x: 1 / (x + 1)    # 역수 함수 (빠른 감소)
    phi9 = lambda x: np.maximum(0, x - 0.5)**2  # ReLU의 변형 (비선형성 증가)
    
    return np.array([[phi1(x), phi2(x), phi3(x), phi4(x), phi5(x), phi6(x), phi7(x), phi8(x), phi9(x)] for x in X])

def J(w, P, x, y):
    """
    비선형 기저 함수를 사용한 목적 함수 계산
    
    입력 데이터에 기저 함수를 적용한 후 MSE 손실을 계산
    
    매개변수:
        w (array): 가중치 벡터
        P (int): 다항식 차수(이 함수에서는 실제로 사용되지 않으나 API 호환성을 위해 유지)
        x (array): 입력 데이터
        y (array): 타겟 값
        
    반환값:
        float: 계산된 목적 함수 값
    """
    PHI = apply_basis_functions(x)
    y_pred = np.dot(w.T, PHI.T)
    return 0.5*(((y-y_pred)**2).sum(axis=-1))

def grad(w, P, x, y):
    """
    비선형 기저 함수를 사용한 목적 함수의 기울기 계산
    
    기저 함수 변환을 적용한 데이터에 대해 MSE 손실의 기울기를 계산
    
    매개변수:
        w (array): 가중치 벡터
        P (int): 다항식 차수(이 함수에서는 실제로 사용되지 않으나 API 호환성을 위해 유지)
        x (array): 입력 데이터
        y (array): 타겟 값
        
    반환값:
        array: 계산된 기울기 벡터
    """
    PHI = apply_basis_functions(x)
    g = np.dot(w.T, np.dot(PHI.T, PHI)) - np.dot(y.T, PHI)
    return g.astype(np.float64)

def CGM(f, df, x, args=(), eps=1.0e-7, max_iter=300, verbose=False):
    """
    공액 경사법(Conjugate Gradient Method) 구현
    
    비선형 최적화 문제를 효율적으로 해결하기 위한 공액 경사법 알고리즘입니다.
    특히 대규모 희소 시스템에 효과적이며, 선형 검색과 결합하여 사용됩니다.
    
    매개변수:
        f (callable): 최적화할 목적 함수
        df (callable): 목적 함수의 그래디언트(기울기) 함수
        x (array): 시작점(초기 추정치)
        args (tuple): 목적 함수와 그래디언트 함수에 전달될 추가 인자
        eps (float): 수렴 판단 기준 (그래디언트 노름이 이 값보다 작으면 종료)
        max_iter (int): 최대 반복 횟수
        verbose (bool): 진행 상황 출력 여부
        
    반환값:
        array: 최적화된 매개변수 벡터
    """
    c = df(x, *args)
    if np.linalg.norm(c) < eps:
        return x
    d = -c
    for k in range(max_iter):
        alpha = line_search(f, df, x, d, args)
        x_new = x + alpha * d
        c_new = df(x_new, *args)
        if np.linalg.norm(c_new) < eps:
            return x_new
        beta = np.dot(c_new, c_new) / np.dot(c, c)
        d = -c_new + beta * d
        x = x_new
        c = c_new
    if verbose:
        print(f"CGM reached max iterations: {max_iter}")
    return x

def line_search(f, df, x, d, args):
    """
    Backtracking Line Search 방법을 이용한 스텝 크기 결정
    
    Armijo 조건을 만족시키는 스텝 크기를 찾기 위한 단순 백트래킹 라인 서치 알고리즘입니다.
    초기 스텝 크기에서 시작하여 목적 함수가 충분히 감소할 때까지 스텝 크기를 줄여나갑니다.
    
    매개변수:
        f (callable): 최적화할 목적 함수
        df (callable): 목적 함수의 그래디언트(기울기) 함수
        x (array): 현재 위치
        d (array): 검색 방향 (보통 그래디언트의 음수 방향)
        args (tuple): 목적 함수와 그래디언트 함수에 전달될 추가 인자
        
    반환값:
        float: Armijo 조건을 만족하는 최적의 스텝 크기
    """
    alpha, beta = 0.01, 0.5  # Armijo 조건 파라미터와 백트래킹 비율
    t = 1.0  # 초기 스텝 크기
    while f(x + t * d, *args) > f(x, *args) - alpha * t * np.dot(df(x, *args), d):
        t *= beta  # 스텝 크기 감소
    return t

def predict_time_series(X):
    """
    시계열 데이터 예측
    
    입력 시계열 데이터를 정규화하고, 기저 함수 변환 후 공액 경사법을 사용하여 
    최적 파라미터를 찾아 예측값을 생성
    
    매개변수:
        X (array-like): 예측할 시계열 데이터
        
    반환값:
        array: 예측된 시계열 데이터
    """
    X_norm, X_min, X_max, X_min_original = log_normalize(X)
    PHI = apply_basis_functions(X_norm)
    w = np.random.uniform(-1, 1, 9) 
    optimized_w = CGM(J, grad, w, args=(8, X_norm, X_norm), verbose=False)
    y_pred = np.dot(optimized_w.T, PHI.T)
    return log_denormalize(y_pred, X_min, X_max, X_min_original)

def process_dataframe(df, input_column, output_column):
    """
    데이터프레임 내 시계열 데이터 일괄 처리
    
    데이터프레임의 특정 열에 포함된 시계열 데이터를 전부 처리하여
    결과를 새로운 열에 저장
    
    매개변수:
        df (DataFrame): 처리할 데이터프레임
        input_column (str): 입력 시계열 데이터가 포함된 열 이름
        output_column (str): 처리 결과를 저장할 열 이름
        
    반환값:
        DataFrame: 처리된 시계열 데이터가 포함된 데이터프레임
    """
    def process_row(row):
        X = row[input_column]
        if isinstance(X, np.ndarray) and X.size > 0:
            row[output_column] = predict_time_series(X)
        else:
            row[output_column] = np.array([])
        return row
    
    df[output_column] = None
    return df.apply(process_row, axis=1)

#ml_data = merged_data[['combo_key','preproc_rfu']]

#small_names = merged_data['name'].unique()[:2]
#temp_data = merged_data.query('name in @small_names')
#
#
#ml_data = process_dataframe(temp_data, 'preproc_rfu', 'ml_baseline_fit')
#ml_data['ml_analysis_absd'] = ml_data['preproc_rfu'] - ml_data['ml_baseline_fit']
#ml_data.to_parquet('C:/Users/kmkim/Desktop/projects/website/docs/data/baseline_optimization/GI-B-I/ml_data/mudt_ml_data.parquet')