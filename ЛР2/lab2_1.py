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

# 1. Загрузка выборок из первой лабораторной
try:
    X1 = np.load(LOG_PATH + "normal_eq_1.npy")
    X2 = np.load(LOG_PATH + "normal_eq_2.npy")
except FileNotFoundError:
    print("Файлы выборок не найдены. Убедитесь, что пути правильные.")
    exit()

N = X1.shape[0]

# 2. Оценки параметров
M1_est, B1_est = estimate_params(X1)
M2_est, B2_est = estimate_params(X2)

# Усредненная корреляционная матрица (т.к. предполагаем их равенство)
B_avg = (B1_est + B2_est) / 2
B_inv = inv(B_avg)

# 3. Априорные вероятности
P1 = 0.5
P2 = 0.5

# 4. Вычисление коэффициентов линейной границы W^T * X + w0 = 0
W = B_inv @ (M1_est - M2_est)
w0 = -0.5 * (M1_est.T @ B_inv @ M1_est) + 0.5 * (M2_est.T @ B_inv @ M2_est) + np.log(P1 / P2)

# Функция классификатора (если h(x) > 0, то класс 1, иначе класс 2)
def classify_linear(x, W, w0):
    return (x @ W + w0) > 0

# 5. Оценка вероятности ошибки
# Аналитическая (через расстояние Махаланобиса)
D2 = (M1_est - M2_est).T @ B_inv @ (M1_est - M2_est)
D = np.sqrt(D2)
P_err_analytic = norm.cdf(-D / 2)

# Экспериментальная (доля неверно классифицированных)
pred1 = classify_linear(X1, W, w0) # Для первого класса ждем True
pred2 = classify_linear(X2, W, w0) # Для второго класса ждем False

errors1 = np.sum(pred1 == False)
errors2 = np.sum(pred2 == True)
P_err_exp = (errors1 + errors2) / (2 * N)

# ================= ФОРМИРОВАНИЕ ОТЧЕТА =================
print("="*60)
print("ОТЧЕТ: ПУНКТ 1. РАВНЫЕ МАТРИЦЫ, РАВНЫЕ ВЕРОЯТНОСТИ")
print("="*60)
print(f"Оценка M1: {M1_est}")
print(f"Оценка M2: {M2_est}")
print(f"Усредненная матрица ковариации B_avg:\n{B_avg}\n")

print("Уравнение разделяющей границы: W_1*x_1 + W_2*x_2 + w0 = 0")
print(f"Веса W: {W}")
print(f"Смещение w0: {w0:.4f}")
print(f"Итоговое уравнение: ({W[0]:.4f})*x_1 + ({W[1]:.4f})*x_2 + ({w0:.4f}) = 0\n")

print(f"Расстояние Махаланобиса D: {D:.4f}")
print(f"Аналитическая вероятность ошибки: {P_err_analytic:.6f}")
print(f"Экспериментальная вероятность ошибки: {P_err_exp:.6f} ({errors1+errors2} ошибок из {2*N})\n")

# ================= ПОСТРОЕНИЕ ГРАФИКА =================
plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], s=15, alpha=0.6, label="Класс 1")
plt.scatter(X2[:, 0], X2[:, 1], s=15, alpha=0.6, label="Класс 2")

# Построение разделяющей прямой x2 = (-W[0]*x1 - w0) / W[1]
x1_min, x1_max = plt.xlim()
x1_vals = np.array([x1_min, x1_max])
x2_vals = (-W[0] * x1_vals - w0) / W[1]

plt.plot(x1_vals, x2_vals, color='red', linewidth=2, label="Байесовская граница")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Разделяющая граница: равные матрицы, P(w1)=P(w2)=0.5")
plt.legend()
plt.grid(True)
plt.xlim(x1_min, x1_max)
plt.ylim(plt.ylim()) # Сохраняем масштаб по y до отрисовки прямой
plt.show()