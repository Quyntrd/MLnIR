import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det, inv

from utility import *

LOG_PATH = "logs\\"
np.random.seed(0)  # чтобы результаты воспроизводились
N = 200

M1 = np.array([0.0, 1.0])
M2 = np.array([1.0, -1.0])
M3 = np.array([-2.0, 1.0])
B1 = np.array([[2.0, 0.0],
               [0.0, 1.0]])

B2 = np.array([[1.0, -0.4],
               [-0.4, 2.0]])

B3 = np.array([[1.5, 0.9],
               [0.9, 1.5]])
B_equal = np.array([[2.0, 0.8],
                    [0.8, 1.5]])

X1 = simulate_normal(M1, B_equal, N)
X2 = simulate_normal(M2, B_equal, N)

# Сохранение выборок
np.save(LOG_PATH + "normal_eq_1.npy", X1)
np.save(LOG_PATH + "normal_eq_2.npy", X2)

# График
plt.figure(figsize=(6, 6))
plt.scatter(X1[:, 0], X1[:, 1], s=10, alpha=0.6, label="Класс 1")
plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.6, label="Класс 2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Две выборки N=200 с равными корреляционными матрицами")
plt.grid(True)
plt.tight_layout()
plt.show()

def estimate_params(X):
    # X – массив shape (N, 2)
    N = X.shape[0]
    M_hat = X.mean(axis=0)
    # B_hat = (1/N) * sum (x_i - M_hat)(x_i - M_hat)^T
    diff = X - M_hat
    B_hat = diff.T @ diff / N
    return M_hat, B_hat

M1_est, B1_est = estimate_params(X1)
M2_est, B2_est = estimate_params(X2)

rho_b_eq = bhattacharyya_distance(M1_est, B1_est, M2_est, B2_est)
rho_m_eq = mahalanobis_distance(M1_est, M2_est, B_equal)

print("== Равные корреляционные матрицы ==")
print("Исходные M1:", M1, "M2:", M2)
print("Оценки M1^:", M1_est, "M2^:", M2_est)
print("Оценки B1^:\n", B1_est)
print("Оценки B2^:\n", B2_est)
print("Расстояние Бхатачария:", rho_b_eq)
print("Расстояние Махаланобиса:", rho_m_eq)

#---Для 3 выборок с разными корреляционными матрицами---
X1_3 = simulate_normal(M1, B1, N)
X2_3 = simulate_normal(M2, B2, N)
X3_3 = simulate_normal(M3, B3, N)

np.save(LOG_PATH + "normal_3_1.npy", X1_3)
np.save(LOG_PATH + "normal_3_2.npy", X2_3)
np.save(LOG_PATH + "normal_3_3.npy", X3_3)

plt.figure(figsize=(6, 6))
plt.scatter(X1_3[:, 0], X1_3[:, 1], s=10, alpha=0.6, label="Класс 1")
plt.scatter(X2_3[:, 0], X2_3[:, 1], s=10, alpha=0.6, label="Класс 2")
plt.scatter(X3_3[:, 0], X3_3[:, 1], s=10, alpha=0.6, label="Класс 3")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Три выборки N=200 с разными корреляционными матрицами")
plt.grid(True)
plt.tight_layout()
plt.show()

M1_3_est, B1_est_3 = estimate_params(X1_3)
M2_3_est, B2_est_3 = estimate_params(X2_3)
M3_3_est, B3_est_3 = estimate_params(X3_3)

pairs = [
    ("1-2", M1_3_est, B1_est_3, M2_3_est, B2_est_3),
    ("1-3", M1_3_est, B1_est_3, M3_3_est, B3_est_3),
    ("2-3", M2_3_est, B2_est_3, M3_3_est, B3_est_3),
]

print("== Разные корреляционные матрицы ==")
for name, Ma, Ba, Mb, Bb in pairs:
    rho_b = bhattacharyya_distance(Ma, Ba, Mb, Bb)
    print(f"Пара {name}: расстояние Бхатачария = {rho_b}")

p1 = np.array([0.5, 0.3, 0.5, 0.3, 0.5])  # Вектор 1: p=0.3 на 2 и 4 компоненте
p2 = np.array([0.3, 0.3, 0.3, 0.3, 0.3])  # Вектор 2: p=0.3 на всех компонентах (пример)

bin_vecs_1 = simulate_binary_vector(p1, N)
bin_vecs_2 = simulate_binary_vector(p2, N)

np.save(LOG_PATH + "binary_1.npy", bin_vecs_1)
np.save(LOG_PATH + "binary_2.npy", bin_vecs_2)

print("Частоты единиц по компонентам (вектор 1):", bin_vecs_1.mean(axis=0))
print("Частоты единиц по компонентам (вектор 2):", bin_vecs_2.mean(axis=0))
