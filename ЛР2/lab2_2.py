import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import norm

LOG_PATH = "logs\\"

def estimate_params(X):
    N = X.shape[0]
    M_hat = X.mean(axis=0)
    diff = X - M_hat
    B_hat = diff.T @ diff / N
    return M_hat, B_hat

# 1. Загрузка данных
X1 = np.load(LOG_PATH + "normal_eq_1.npy")
X2 = np.load(LOG_PATH + "normal_eq_2.npy")
N = X1.shape[0]

# 2. Оценки
M1_est, B1_est = estimate_params(X1)
M2_est, B2_est = estimate_params(X2)
B_avg = (B1_est + B2_est) / 2
B_inv = inv(B_avg)

# 3. Установка НЕРАВНЫХ априорных вероятностей
P1 = 0.8
P2 = 0.2

# 4. Расчет параметров границы
W = B_inv @ (M1_est - M2_est)
# Слагаемое np.log(P1 / P2) теперь существенно влияет на результат
w0 = -0.5 * (M1_est.T @ B_inv @ M1_est) + 0.5 * (M2_est.T @ B_inv @ M2_est) + np.log(P1 / P2)

# Функция классификации
def classify_linear(x, W, w0):
    return (x @ W + w0) > 0

# 5. Оценка ошибок
# Экспериментальная
pred1 = classify_linear(X1, W, w0)
pred2 = classify_linear(X2, W, w0)
errors1 = np.sum(pred1 == False)
errors2 = np.sum(pred2 == True)
P_err_exp = (errors1 + errors2) / (2 * N)

# Аналитическая (для случая равных матриц, но разных P)
# Формула усложняется: ошибка = P1*Ф(-d1) + P2*Ф(-d2)
d = np.sqrt((M1_est - M2_est).T @ B_inv @ (M1_est - M2_est))
d1 = d/2 + (1/d) * np.log(P1/P2)
d2 = d/2 - (1/d) * np.log(P1/P2)
P_err_analytic = P1 * norm.cdf(-d1) + P2 * norm.cdf(-d2)

# ================= ФОРМИРОВАНИЕ ОТЧЕТА =================
print("="*60)
print("ОТЧЕТ: ПУНКТ 2. РАВНЫЕ МАТРИЦЫ, НЕРАВНЫЕ ВЕРОЯТНОСТИ")
print("="*60)
print(f"Заданные вероятности: P(w1) = {P1}, P(w2) = {P2}")
print(f"Коэффициент смещения ln(P1/P2): {np.log(P1/P2):.4f}")
print(f"Итоговое уравнение: ({W[0]:.4f})*x_1 + ({W[1]:.4f})*x_2 + ({w0:.4f}) = 0\n")

print(f"Аналитическая вероятность ошибки: {P_err_analytic:.6f}")
print(f"Экспериментальная вероятность ошибки: {P_err_exp:.6f}")
print(f"Распределение ошибок: класс 1 -> {errors1}, класс 2 -> {errors2}\n")

# ================= ВИЗУАЛИЗАЦИЯ =================
plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], s=15, alpha=0.6, label=f"Класс 1 (P={P1})")
plt.scatter(X2[:, 0], X2[:, 1], s=15, alpha=0.6, label=f"Класс 2 (P={P2})")

x1_lim = plt.xlim()
x1_v = np.array(x1_lim)
x2_v = (-W[0] * x1_v - w0) / W[1]

plt.plot(x1_v, x2_v, color='green', linestyle='--', linewidth=2, label="Граница (P1=0.8)")
plt.title(f"Смещение границы при P(w1)={P1}, P(w2)={P2}")
plt.legend()
plt.grid(True)
plt.show()