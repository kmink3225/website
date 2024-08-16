import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(X):
    X_min = np.min(X)
    X_max = np.max(X)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized, X_min, X_max

def denormalize(X_normalized, X_min, X_max):
    X_restored = X_normalized * (X_max - X_min) + X_min
    return X_restored

def apply_basis_functions(X):
    phi1 = lambda x: x**0  # 상수항
    phi2 = lambda x: x**1  # 1차항
    phi3 = lambda x: x**2  # 2차항
    phi4 = lambda x: x**3  # 3차항
    phi5 = lambda x: x**4  # 4차항
    phi6 = lambda x: x*np.e**(-x)  # 지수항
    return np.array([[phi1(x), phi2(x), phi3(x), phi4(x), phi5(x), phi6(x)] for x in X])

def J(w, P, x, y):
    PHI = np.array([x**i for i in range(P+1)]).T
    y_pred = np.dot(w.T, PHI.T)
    return 0.5*(((y-y_pred)**2).sum(axis=-1))

def grad(w, P, x, y):
    PHI = np.array([x**i for i in range(P+1)]).T
    g = np.dot(w.T, np.dot(PHI.T, PHI)) - np.dot(y.T, PHI)
    return g.astype(np.float64)

def CGM(f, df, x, args=(), eps=1.0e-7, max_iter=600, verbose=False):
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
    X_norm, X_min, X_max = normalize(X)
    PHI = apply_basis_functions(X_norm)
    w = np.random.uniform(-1, 1, 6)
    optimized_w = CGM(J, grad, w, args=(5, X_norm, X_norm), verbose=False)
    y_pred = np.dot(optimized_w.T, PHI.T)
    return denormalize(y_pred, X_min, X_max)

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

def plot_results(df, input_column, output_column, num_samples=9):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= num_samples:
            break

        X = row[input_column]
        y_pred = row[output_column]

        if isinstance(X, np.ndarray) and X.size > 0:
            axes[i].plot(X, label='Original Data')
            axes[i].plot(y_pred, label='Predicted Data')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].legend()

    plt.tight_layout()
    plt.show()