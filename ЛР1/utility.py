import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv

np.random.seed(0)  # чтобы результаты воспроизводились

def bhattacharyya_distance(M0, B0, M1, B1):
    # M0, M1 – (2,), B0, B1 – (2,2)
    B_avg = 0.5 * (B0 + B1)
    term1 = 0.25 * (M1 - M0).T @ inv(B_avg) @ (M1 - M0)
    term2 = 0.5 * np.log(det(B_avg) / np.sqrt(det(B0) * det(B1)))
    return float(term1 + term2)

def mahalanobis_distance(M0, M1, B):
    # B – общая корреляционная матрица
    diff = M1 - M0
    return float(diff.T @ inv(B) @ diff)

def simulate_normal(M, B, N):
    """
    Моделирование N реализаций двумерного нормального вектора X ~ N(M, B)
    """
    B00, B01 = B[0, 0], B[0, 1]
    B11 = B[1, 1]

    a00 = np.sqrt(B00)
    a10 = B01 / a00
    a11 = np.sqrt(max(0, B11 - (B01**2 / B00)))

    A = np.array([[a00, 0],
                  [a10, a11]])
    
    xi = np.random.randn(N, 2)      # ξ ~ N(0, I)
    X = xi @ A.T + M                # X = A ξ + M
    return X

def simulate_binary_vector(rep_2d: np.ndarray, N: int, p: float, rng: np.random.Generator):
    """
    Моделирование выборки N бинарных векторов.
    p – массив вероятностей P(X_i = 1) длины n.
    """
    rep = np.asarray(rep_2d, dtype=np.int8).reshape(-1, 1)  # (n,1)
    n = rep.shape[0]
    flips = np.floor(rng.random(size=(n, N)) < p).astype(np.int8)
    X = rep ^ flips
    return X

def estimate_params(X):
    # X – массив shape (N, 2)
    N = X.shape[0]
    M_hat = X.mean(axis=0)
    # B_hat = (1/N) * sum (x_i - M_hat)(x_i - M_hat)^T
    diff = X - M_hat
    B_hat = diff.T @ diff / N
    return M_hat, B_hat