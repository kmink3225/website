# simple nueral network

def compute_simple_nn(y):
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


def f_alpha(alpha, fun, x, s, args=()) :
    """
    alpha : 1D independent variable
    fun   : Original objective function
    x     : Start point
    s     : 1D search direction
    args  : Tuple extra arguments passed to the objective function
    """
    x_new = x + alpha * s
    
    return fun(x_new, *args)

def search_golden_section(fun, dfun, x, s, args=(), delta=1.0e-2, tol=1e-15):
    """
    https://en.wikipedia.org/wiki/Golden-section_search and [arora]
    
    fun   : Original objective function
    dfun  : Objective function gradient which is not used
    x     : Start point
    s     : 1D search directin
    args  : Tuple extra arguments passed to the objective function
    delta : Init. guess interval determining initial interval of uncertainty
    tol   : stop criterion
    """
    gr = (np.sqrt(5) + 1) / 2
        
    AL = 0.
    FL = f_alpha(AL, fun, x, s, args)
    AA = delta
    FA = f_alpha(AA, fun, x, s, args)
    while  FL < FA :
        delta = 0.1*delta
        AA = delta
        FA = f_alpha(AA, fun, x, s, args)
    
    j = 1
    AU = AA + delta * (gr**j)
    FU = f_alpha(AU, fun, x, s, args)
    while FA > FU :
        AL = AA
        AA = AU
        FL = FA
        FA = FU
        
        j += 1
        AU = AA + delta * (gr**j)
        FU = f_alpha(AU, fun, x, s, args)

    AB = AL + (AU - AL) / gr
    FB = f_alpha(AB, fun, x, s, args)
    
    while abs(AA - AB) > tol:
        if f_alpha(AA, fun, x, s, args) < f_alpha(AB, fun, x, s, args):
            AU = AB
        else:
            AL = AA

        # we recompute both c and d here to avoid loss of precision 
        # which may lead to incorrect results or infinite loop
        AA = AU - (AU - AL) / gr
        AB = AL + (AU - AL) / gr

    return ( (AU + AL) / 2, )


def compute_conjugate_gradient(f, df, x, args=(), eps=1.0e-7, max_iter=2500, verbose=False, callback=None):
    """
    f       : 최적화 할 함수
    df      : 최적화할 함수의 도함수
    x       : 초기값
    args    : f(x, *args)로 f를 호출하기 위한 추가 매개변수
    eps     : 정기 기준
    max_iter: 최대 반복수
    verbose : 실행 정보를 텍스트로 출력 
    callback: x에 대한 기록을 위한 콜백함수
    """

    if verbose:
        print("#####START OPTIMIZATION#####")
        print("INIT POINT : {}, dtype : {}".format(x, x.dtype))

    for k in range(max_iter):
        c = df(x, *args)
        if np.linalg.norm(c) < eps :
            if verbose:
                print("Stop criterion break Iter: {:5d}, x: {}".format(k, x))
                print("\n")
            break

        if k == 0 :
            d = -c
        else:
            beta = (np.linalg.norm(c) / np.linalg.norm(c_old))**2
            d = -c + beta*d
        
        alpha = search_golden_section(f, df, x, d, args=args)[0]
        x = x + alpha * d
        c_old = c.copy()    

        if callback :
            callback(x)    

    else:
        print("Stop max iter:{:5d} x:{}".format(k, x)) 

    return x


def haar_wavelet(x):
    """
    A simple Haar wavelet function.
    """
    if 0 <= x < 0.5:
        return 1
    elif 0.5 <= x < 1:
        return -1
    else:
        return 0

def piecewise_basis(x):
    """
    A simple piecewise basis function.
    """
    if x < 0.3:
        return x**2
    elif 0.3 <= x < 0.6:
        return 3 * x + 1
    else:
        return np.sin(2 * np.pi * x)

# Define the Haar Wavelet function
def haar_wavelet(x):
    if 0 <= x < 0.5:
        return 1
    elif 0.5 <= x < 1:
        return -1
    else:
        return 0

# Define the Piecewise Basis function
def piecewise_basis(x):
    if x < 0.3:
        return x**2
    elif 0.3 <= x < 0.6:
        return 3 * x + 1
    else:
        return np.sin(2 * np.pi * x)

def compute_phi(X):
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
    This function computes the objective function with L1 regularization. 
    J(w)= (1/2) * sum_{n=1}^{N} {y(x_n,w) - t_n}^2
    y(x_n; w) = w_0*x^0 + w_1*x^1 + w_2*x^2 + ... + w_M*x^M
    
    w: 가중치 행렬, (M,) or (M,K) 여기서 K는 가중치 벡터 개수, 
       M은 기저함수 개수 또는 데이터 차원
    P: polynomial degree, scalar
    x, y : data for error function eval.
      x: data, (N,M)
      y: target, (N,)
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
    This function computes the objective function with L2 regularization. 
    J(w)= (1/2) * sum_{n=1}^{N} {y(x_n,w) - t_n}^2
    y(x_n; w) = w_0*x^0 + w_1*x^1 + w_2*x^2 + ... + w_M*x^M
    
    w: 가중치 행렬, (M,) or (M,K) 여기서 K는 가중치 벡터 개수, 
       M은 기저함수 개수 또는 데이터 차원
    P: polynomial degree, scalar
    x, y : data for error function eval.
      x: data, (N,M)
      y: target, (N,)
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

def plot_result(X_train,y_train):
    fig, ax = plt.subplots(figsize=(15,10), nrows=2, ncols=2, 
                       sharex='all', sharey='all')

    j = 0

    for P in Ps :
        p, q = divmod(j, 2)
        X = np.array([ x_raw**i for i in range(P+1) ])  

        y = (W[j].reshape(-1,1) * X).sum(axis=0)

        ax[p,q].xaxis.set_tick_params(labelsize=18)
        ax[p,q].yaxis.set_tick_params(labelsize=18)
        ax[p,q].set_xlabel('$x$', fontsize=20)
        ax[p,q].set_ylabel('$y$', fontsize=20)
        ax[p,q].grid(False)

        ax[p,q].plot(X_train, y_train, 'o',  markersize=8 , color='k')
        ax[p,q].legend(loc='upper right', fontsize=18)

        j+=1

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax_ in ax.flat:
        ax_.label_outer()

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()

# Characteristic Equation

def log_normalize(X):
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
    # 역정규화
    X_log_restored = X_normalized * (X_max - X_min) + X_min

    # 지수 변환
    X_restored = np.exp(X_log_restored)

    # 원래 스케일로 복원
    X_restored = X_restored + X_min_original - 1e-10

    return X_restored

def apply_basis_functions(X):
    
    #hi1 = lambda x: x**0 
    #hi2 = lambda x: x    
    #hi3 = lambda x: x**2 
    #hi4 = lambda x: x**3 
    #hi5 = lambda x: x**4 
    #hi6 = lambda x: x**5 
    #hi7 = lambda x: x**6 
    #hi8 = lambda x: x**7 
    #hi9 = lambda x: x**8 
    ###################################
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
    PHI = apply_basis_functions(x)
    y_pred = np.dot(w.T, PHI.T)
    return 0.5*(((y-y_pred)**2).sum(axis=-1))

def grad(w, P, x, y):
    PHI = apply_basis_functions(x)
    g = np.dot(w.T, np.dot(PHI.T, PHI)) - np.dot(y.T, PHI)
    return g.astype(np.float64)

def CGM(f, df, x, args=(), eps=1.0e-7, max_iter=300, verbose=False):
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
    alpha, beta = 0.01, 0.5
    t = 1.0
    while f(x + t * d, *args) > f(x, *args) - alpha * t * np.dot(df(x, *args), d):
        t *= beta
    return t

def predict_time_series(X):
    X_norm, X_min, X_max, X_min_original = log_normalize(X)
    PHI = apply_basis_functions(X_norm)
    w = np.random.uniform(-1, 1, 9) 
    optimized_w = CGM(J, grad, w, args=(8, X_norm, X_norm), verbose=False)
    y_pred = np.dot(optimized_w.T, PHI.T)
    return log_denormalize(y_pred, X_min, X_max, X_min_original)

def process_dataframe(df, input_column, output_column):
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