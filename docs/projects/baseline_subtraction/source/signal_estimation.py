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

def compute_phi(X):
    phi_functions = [
        lambda x: x[0] + x[1],
        lambda x: x[0] * x[1],
        lambda x: x[0] / x[1],
        lambda x: x[0] ** x[1],
    ]
    PHI = np.array([[phi(x) for phi in phi_functions] for x in X])
    return PHI

############
import numpy as np

def compute_phi(X):
    # Define a center for the RBF, and choose a spread parameter
    rbf_center = np.mean(X)
    rbf_spread = np.std(X)

    phi_functions = [
        lambda x: np.exp(-((x - rbf_center)**2) / (2 * rbf_spread**2)),  # RBF
        lambda x: np.sin(2 * np.pi * x),  # Fourier basis (sine)
        lambda x: np.cos(2 * np.pi * x),  # Fourier basis (cosine)
        lambda x: 1 if x < 0.5 else -1,  # Wavelet Piecewise (simple example)
    ]
    
    # Convert X to a numpy array for easier manipulation
    X = np.array(X)

    # Calculate PHI for each x in X for all phi_functions
    PHI = np.array([[phi(x) for phi in phi_functions] for x in X])
    
    return PHI

# Example data: 50 data points
X = list(np.linspace(0, 1, 50))  # Adjust this based on your actual data

PHI = compute_phi(X)

# To print or analyze PHI, uncomment the following line
# print(PHI)


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

# Example usage with an array of data points
X = np.linspace(0, 1, 50)

# Applying the Haar wavelet and piecewise basis function to each point in X
Y_haar = np.array([haar_wavelet(x) for x in X])
Y_piecewise = np.array([piecewise_basis(x) for x in X])

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(X, Y_haar, label='Haar Wavelet')
plt.title('Haar Wavelet')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X, Y_piecewise, label='Piecewise Basis')
plt.title('Piecewise Basis Function')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np

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

# Generate an example input array
X = np.linspace(0, 1, 50)

# Compute the PHI matrix
PHI = compute_phi(X)

print(PHI)

###########

def compute_loss(w, P, x, y):
    """
    This function computes the objective function. 
    
    w: 가중치 행렬, (M,) or (M,K) 여기서 K는 가중치 벡터 개수, 
       M은 기저함수 개수 또는 데이터 차원
    P: polynomial degree, scalar
    x, y : data for error function eval.
      x: data, (N,M)
      y: target, (N,)
    """
    #PHI = np.array([ x**i for i in range(P+1) ]).T
    P=P
    phi = compute_phi(x)
    PHI = phi.T
    
    y_pred = np.dot(w.T, PHI.T) 
    return 0.5*(((y-y_pred)**2).sum(axis=-1)) 

def get_gradient(w, P, x, y, grad_type = 'float64'):
    """
    This function computes the analytic gradient of the objective function.

    w: 가중치 행렬, (M,) or (M,K) 여기서 K는 가중치 벡터 개수, 
       M은 기저함수 개수 또는 데이터 차원
    P: polynomial degree, scalar
    x, y : data for error function eval.
      x: data, (N,M)
      y: target, (N,)
    """
    #PHI = np.array([ x**i for i in range(P+1) ]).T
    P=P
    phi = compute_phi(x)
    PHI = phi.T
    # broadcasting
    g = np.dot(w.T, np.dot(PHI.T, PHI) ) - np.dot(y.T, PHI) 
    
    return g.astype(grad_type)

def rmse(w, J, P, x, y) :
    return np.sqrt( (2*J(w, P, x, y)) / x.shape[0] )

## Regularization

# Regularization
    
reg_intercept = True
lamda = 0.01

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