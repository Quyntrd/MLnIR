import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv, det

# =====================================================================
# НАСТРОЙКИ И ИСХОДНЫЕ ДАННЫЕ (ИЗ ЛАБ. РАБОТЫ 1)
# =====================================================================
LOG_PATH = "logs"
os.makedirs(LOG_PATH, exist_ok=True)
N = 200

# Математические ожидания
M0 = np.array([1.0, 0.0])      # Класс 0 (в 1-й работе M1)
M1 = np.array([-1.0, -1.0])    # Класс 1 (в 1-й работе M2)
M2 = np.array([1.0, 2.0])      # Класс 2 (в 1-й работе M3)

# Равные корреляционные матрицы (Задания 1 и 2)
B_eq = np.array([[1.0, 0.2],
                 [0.2, 1.0]])

# Неравные корреляционные матрицы (Задание 3)
B0 = np.array([[1.0, -0.9],[-0.9, 1.0]])
B1 = np.array([[1.0,  0.0],[ 0.0, 1.0]])
B2 = np.array([[1.0,  0.9], [ 0.9, 1.0]])

# Прототипы для бинарных векторов (Задание 4)
proto1 = np.array([
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,1]
], dtype=int).flatten()

proto2 = np.array([
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0,0,0],
    [0,1,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0,0],
    [0,1,1,1,1,1,0,0,0]
], dtype=int).flatten()

# Загрузка или генерация данных
rng = np.random.default_rng(7)

def load_or_generate_normal(filename, M, B):
    path = os.path.join(LOG_PATH, filename)
    if os.path.exists(path):
        return np.load(path)
    return rng.multivariate_normal(M, B, N)

def load_or_generate_binary(filename, proto):
    path = os.path.join(LOG_PATH, filename)
    if os.path.exists(path):
        return np.load(path)
    p_noise = np.where(proto == 1, 0.7, 0.3)
    return (rng.random(size=(N, len(proto))) < p_noise).astype(int)

X_eq_0 = load_or_generate_normal("normal_eq_1.npy", M0, B_eq)
X_eq_1 = load_or_generate_normal("normal_eq_2.npy", M1, B_eq)

X_3_0 = load_or_generate_normal("normal_3_1.npy", M0, B0)
X_3_1 = load_or_generate_normal("normal_3_2.npy", M1, B1)
X_3_2 = load_or_generate_normal("normal_3_3.npy", M2, B2)

X_bin_0 = load_or_generate_binary("binary_1.npy", proto1)
X_bin_1 = load_or_generate_binary("binary_2.npy", proto2)

print("="*60)
print("СОДЕРЖИМОЕ ОТЧЕТА ПО ЛАБОРАТОРНОЙ РАБОТЕ 2")
print("="*60)


# =====================================================================
# ЗАДАНИЕ 1 и 2: Равные корреляционные матрицы (Байес, Минимакс, Н-П)
# =====================================================================
print("\n--- Задания 1 и 2: Равные ковариационные матрицы ---")

# Аналитические вычисления (Байес)
B_inv = inv(B_eq)
w = B_inv @ (M1 - M0)
b_bayes = -0.5 * (M1 - M0).T @ B_inv @ (M1 + M0)

rho2 = (M1 - M0).T @ B_inv @ (M1 - M0)
rho = np.sqrt(rho2)

# Ошибки Байеса
p0_bayes = norm.cdf(-rho / 2)
p1_bayes = norm.cdf(-rho / 2)

# Ошибки Байеса (эксперимент)
Z0_eq = X_eq_0 @ w + b_bayes
Z1_eq = X_eq_1 @ w + b_bayes
e0_bayes = np.mean(Z0_eq > 0)
e1_bayes = np.mean(Z1_eq < 0)

# Нейман-Пирсон
p0_star = 0.05
lambda_tilde_NP = -0.5 * rho**2 + rho * norm.ppf(1 - p0_star)
p0_np = 1 - norm.cdf((lambda_tilde_NP - (-0.5 * rho**2)) / rho)
p1_np = norm.cdf((lambda_tilde_NP - (0.5 * rho**2)) / rho)

e0_np = np.mean(Z0_eq > lambda_tilde_NP)
e1_np = np.mean(Z1_eq < lambda_tilde_NP)

print("1. Аналитические выражения:")
print(f"  Байесовская граница: d(x) = {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + ({b_bayes:.4f}) = 0")
print("  Минимаксная граница: совпадает с байесовской (для равных априорных вероятностей и штрафов).")
print(f"  Граница Неймана-Пирсона: d(x) = {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + ({b_bayes:.4f}) = {lambda_tilde_NP:.4f}")

print("\n2. Вероятности ошибок (Байесовский и Минимаксный):")
print(f"  Аналитические: p0 = {p0_bayes:.4f}, p1 = {p1_bayes:.4f}, Суммарная = {0.5*p0_bayes + 0.5*p1_bayes:.4f}")
print(f"  Эксперимент:   e0 = {e0_bayes:.4f}, e1 = {e1_bayes:.4f}, Суммарная = {0.5*e0_bayes + 0.5*e1_bayes:.4f}")

print("\n3. Вероятности ошибок (Неймана-Пирсона, задан p0* = 0.05):")
print(f"  Аналитические: p0 = {p0_np:.4f}, p1 = {p1_np:.4f}")
print(f"  Эксперимент:   e0 = {e0_np:.4f}, e1 = {e1_np:.4f}")

# Отрисовка
plt.figure(figsize=(8, 6))
plt.scatter(X_eq_0[:, 0], X_eq_0[:, 1], alpha=0.5, label=r'Класс $\Omega_0$')
plt.scatter(X_eq_1[:, 0], X_eq_1[:, 1], alpha=0.5, label=r'Класс $\Omega_1$')

min_x = min(np.min(X_eq_0[:, 0]), np.min(X_eq_1[:, 0])) - 1
max_x = max(np.max(X_eq_0[:, 0]), np.max(X_eq_1[:, 0])) + 1
x_vals = np.linspace(min_x, max_x, 100)

y_bayes = (-w[0] * x_vals - b_bayes) / w[1]
y_np = (-w[0] * x_vals - b_bayes + lambda_tilde_NP) / w[1]

plt.plot(x_vals, y_bayes, 'k-', linewidth=2, label='Байес / Минимакс')
plt.plot(x_vals, y_np, 'r--', linewidth=2, label='Нейман-Пирсон')
plt.xlabel('x1'); plt.ylabel('x2')
plt.title("Задания 1 и 2: Равные ковариационные матрицы")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================================
# ЗАДАНИЕ 3: Неравные корреляционные матрицы (3 класса)
# =====================================================================
print("\n--- Задание 3: Неравные ковариационные матрицы (3 класса) ---")

def d_quad(x, M, B):
    B_inv = inv(B)
    diff = x - M
    return -0.5 * np.sum(diff @ B_inv * diff, axis=1) - 0.5 * np.log(det(B))

print("1. Аналитические выражения классификаторов (d_l(x) = x^T Q_l x + L_l^T x + C_l):")
for i, (M, B) in enumerate(zip([M0, M1, M2],[B0, B1, B2])):
    Q = -0.5 * inv(B)
    L = inv(B) @ M
    C = -0.5 * M.T @ inv(B) @ M - 0.5 * np.log(det(B))
    print(f"  Класс {i}:")
    print(f"    Q = {Q.tolist()}")
    print(f"    L = {L.tolist()}")
    print(f"    C = {C:.4f}")

# Экспериментальная оценка (возьмем классы 0 и 1 для оценки погрешности)
d0_x0 = d_quad(X_3_0, M0, B0)
d1_x0 = d_quad(X_3_0, M1, B1)
d2_x0 = d_quad(X_3_0, M2, B2)
pred_x0 = np.argmax(np.column_stack([d0_x0, d1_x0, d2_x0]), axis=1)
e0_3 = np.mean(pred_x0 != 0)

d0_x1 = d_quad(X_3_1, M0, B0)
d1_x1 = d_quad(X_3_1, M1, B1)
d2_x1 = d_quad(X_3_1, M2, B2)
pred_x1 = np.argmax(np.column_stack([d0_x1, d1_x1, d2_x1]), axis=1)
e1_3 = np.mean(pred_x1 != 1)

print("\n2. Экспериментальные оценки и относительная погрешность (N=200):")
for i, e in enumerate([e0_3, e1_3]):
    print(f"  Класс {i}: Ошибка e{i} = {e:.4f}")
    if e > 0:
        eps = np.sqrt((1 - e) / (e * N))
        N_req = (1 - e) / (e * 0.05**2)
        print(f"    Относительная погрешность: {eps:.4f}")
        print(f"    Требуемый объем выборки для погрешности <= 5%: {int(np.ceil(N_req))}")
    else:
        print("    Относительная погрешность неприменима (эмпирическая ошибка равна 0).")

# Отрисовка
plt.figure(figsize=(8, 6))
min_x = min(np.min(X_3_0[:, 0]), np.min(X_3_1[:, 0]), np.min(X_3_2[:, 0])) - 1
max_x = max(np.max(X_3_0[:, 0]), np.max(X_3_1[:, 0]), np.max(X_3_2[:, 0])) + 1
min_y = min(np.min(X_3_0[:, 1]), np.min(X_3_1[:, 1]), np.min(X_3_2[:, 1])) - 1
max_y = max(np.max(X_3_0[:, 1]), np.max(X_3_1[:, 1]), np.max(X_3_2[:, 1])) + 1

xx, yy = np.meshgrid(np.linspace(min_x, max_x, 200), np.linspace(min_y, max_y, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z_grid = np.argmax(np.column_stack([d_quad(grid, M0, B0), d_quad(grid, M1, B1), d_quad(grid, M2, B2)]), axis=1)
Z_grid = Z_grid.reshape(xx.shape)

plt.contourf(xx, yy, Z_grid, alpha=0.3, cmap='Pastel1')
plt.scatter(X_3_0[:, 0], X_3_0[:, 1], alpha=0.6, label=r'Класс $\Omega_0$')
plt.scatter(X_3_1[:, 0], X_3_1[:, 1], alpha=0.6, label=r'Класс $\Omega_1$')
plt.scatter(X_3_2[:, 0], X_3_2[:, 1], alpha=0.6, label=r'Класс $\Omega_2$')
plt.contour(xx, yy, Z_grid, colors='k', linewidths=0.5)
plt.xlabel('x1'); plt.ylabel('x2')
plt.title("Задание 3: Байесовские границы для 3-х классов (неравные ковариации)")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================================
# ЗАДАНИЕ 4: Бинарные вектора признаков
# =====================================================================
print("\n--- Задание 4: Бинарные векторы признаков ---")

# Истинные вероятности генерации (с учетом шума 30%)
p0 = np.where(proto1 == 1, 0.7, 0.3)
p1 = np.where(proto2 == 1, 0.7, 0.3)

# Веса для логарифма отношения правдоподобия
wi = np.log((p1 * (1 - p0)) / (p0 * (1 - p1)))
w0 = np.sum(np.log((1 - p1) / (1 - p0)))

# Аналитическая ошибка с использованием ЦПТ
m0 = np.sum(p0 * wi) + w0
m1 = np.sum(p1 * wi) + w0
sigma0 = np.sqrt(np.sum(p0 * (1 - p0) * wi**2))
sigma1 = np.sqrt(np.sum(p1 * (1 - p1) * wi**2))

# Если Z > 0, принимаем класс 1. Для класса 0 ошибка если Z > 0
p0_bin_analyt = norm.cdf(m0 / sigma0)
p1_bin_analyt = norm.cdf(-m1 / sigma1)

# Экспериментальная ошибка
Z0_bin = X_bin_0 @ wi + w0
Z1_bin = X_bin_1 @ wi + w0

e0_bin = np.mean(Z0_bin > 0)
e1_bin = np.mean(Z1_bin < 0)

print("1. Параметры классификатора:")
print(f"  Порог w0: {w0:.4f}")
print("  Компоненты вектора весов w_i отображены на графике.")

print("\n2. Вероятности ошибок (Бинарные вектора):")
print(f"  Аналитические: p0 = {p0_bin_analyt:.4f}, p1 = {p1_bin_analyt:.4f}")
print(f"  Эксперимент:   e0 = {e0_bin:.4f}, e1 = {e1_bin:.4f}")

# Отрисовка
plt.figure(figsize=(6, 5))
plt.imshow(wi.reshape(9, 9), cmap='coolwarm', aspect='auto')
plt.colorbar(label='Вес компоненты $w_i$')
plt.title("Задание 4: Компоненты вектора весов (изображение 9x9)")
plt.axis('off')
plt.show()
