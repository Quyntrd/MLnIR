from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from scipy.stats import norm

from utility import simulate_normal, simulate_binary_vector, bhattacharyya_distance, mahalanobis_distance

OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
N = 200

M1 = np.array([1.0, 0.0])
M2 = np.array([-2.0, -2.0])
M3 = np.array([1.0, 2.0])

B_equal = np.array([[1.0, 0.2],
                    [0.2, 1.0]])
B1 = np.array([[1.0, -0.9],
               [-0.9, 1.0]])
B2 = np.array([[1.0, 0.0],
               [0.0, 1.0]])
B3 = np.array([[1.0, 0.9],
               [0.9, 1.0]])

proto1 = np.array([
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,0,1,0,0],
    [1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,1],
], dtype=np.int8)

proto2 = np.array([
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0,0,0],
    [0,1,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0,0],
    [0,1,1,1,1,1,0,0,0],
], dtype=np.int8)

p1_proto = proto1.flatten()
p2_proto = proto2.flatten()

def ensure_data() -> Dict[str, np.ndarray]:
    files = {
        'eq1': LOG_DIR / 'normal_eq_1.npy',
        'eq2': LOG_DIR / 'normal_eq_2.npy',
        '3_1': LOG_DIR / 'normal_3_1.npy',
        '3_2': LOG_DIR / 'normal_3_2.npy',
        '3_3': LOG_DIR / 'normal_3_3.npy',
        'b1': LOG_DIR / 'binary_1.npy',
        'b2': LOG_DIR / 'binary_2.npy',
    }

    if all(p.exists() for p in files.values()):
        return {
            'X1': np.load(files['eq1']),
            'X2': np.load(files['eq2']),
            'X1_3': np.load(files['3_1']),
            'X2_3': np.load(files['3_2']),
            'X3_3': np.load(files['3_3']),
            'bin1': np.load(files['b1']),
            'bin2': np.load(files['b2']),
        }

    X1 = simulate_normal(M1, B_equal, N)
    X2 = simulate_normal(M2, B_equal, N)
    X1_3 = simulate_normal(M1, B1, N)
    X2_3 = simulate_normal(M2, B2, N)
    X3_3 = simulate_normal(M3, B3, N)
    bin1 = simulate_binary_vector(p1_proto, N, 0.3, rng).T
    bin2 = simulate_binary_vector(p2_proto, N, 0.3, rng).T

    np.save(files['eq1'], X1)
    np.save(files['eq2'], X2)
    np.save(files['3_1'], X1_3)
    np.save(files['3_2'], X2_3)
    np.save(files['3_3'], X3_3)
    np.save(files['b1'], bin1)
    np.save(files['b2'], bin2)

    return {
        'X1': X1, 'X2': X2, 'X1_3': X1_3, 'X2_3': X2_3, 'X3_3': X3_3,
        'bin1': bin1, 'bin2': bin2,
    }

def classify_linear(X: np.ndarray, w: np.ndarray, w0: float) -> np.ndarray:
    return (X @ w + w0 >= 0).astype(int)

def plot_linear_boundary(ax, w, w0, xlim, ylim, label, style='-'):
    xs = np.linspace(xlim[0], xlim[1], 200)
    if abs(w[1]) < 1e-12:
        x = -w0 / w[0]
        ax.axvline(x, linestyle=style, label=label)
        return
    ys = -(w[0] * xs + w0) / w[1]
    ax.plot(xs, ys, style, linewidth=2, label=label)

def plot_quadratic_boundaries(ax, Ms, Bs):
    x_min, x_max = -5.5, 4.5
    y_min, y_max = -4.5, 4.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.stack([xx, yy], axis=-1)
    scores = []
    for M, B in zip(Ms, Bs):
        Bi = inv(B)
        const = -0.5 * np.log(det(B))
        s = const - 0.5 * np.einsum('...i,ij,...j->...', grid - M, Bi, grid - M)
        scores.append(s)
    scores = np.stack(scores, axis=-1)
    ax.contour(xx, yy, scores[..., 0] - scores[..., 1], levels=[0], colors='k', linewidths=2)
    ax.contour(xx, yy, scores[..., 0] - scores[..., 2], levels=[0], colors='k', linewidths=2, linestyles='--')
    ax.contour(xx, yy, scores[..., 1] - scores[..., 2], levels=[0], colors='k', linewidths=2, linestyles=':')
    return x_min, x_max, y_min, y_max

def binary_bayes_predict(X: np.ndarray, proto_a: np.ndarray, proto_b: np.ndarray, p: float = 0.3, prior_a: float = 0.5, prior_b: float = 0.5) -> np.ndarray:
    qa1 = np.where(proto_a == 1, 1 - p, p)
    qb1 = np.where(proto_b == 1, 1 - p, p)
    log_ratio = X @ np.log((qa1 * (1 - qb1)) / ((1 - qa1) * qb1)) + np.sum(np.log((1 - qa1) / (1 - qb1))) + math.log(prior_a / prior_b)
    return (log_ratio < 0).astype(int)

def binary_analytic_error(proto_a: np.ndarray, proto_b: np.ndarray, p: float = 0.3) -> float:
    d = int(np.sum(proto_a != proto_b))
    if d == 0:
        return 0.0
    threshold = d / 2
    pe = 0.0
    for k in range(0, d + 1):
        prob = math.comb(d, k) * ((1 - p) ** k) * (p ** (d - k))
        if k < threshold:
            pe += prob
        elif d % 2 == 0 and k == threshold:
            pe += 0.5 * prob
    return float(pe)

def approx_relative_error(p_hat: float, n: int) -> float:
    if p_hat <= 0:
        return float('inf')
    return math.sqrt((1 - p_hat) / (p_hat * n))

def required_n_for_relative_error(p_hat: float, eps: float = 0.05) -> int:
    if p_hat <= 0:
        return math.inf
    return int(math.ceil((1 - p_hat) / (p_hat * eps ** 2)))

def lab2(data: Dict[str, np.ndarray]):
    X1, X2 = data['X1'], data['X2']
    X1_3, X2_3, X3_3 = data['X1_3'], data['X2_3'], data['X3_3']
    bin1, bin2 = data['bin1'], data['bin2']

    M0, M1_ = M1, M2
    B = B_equal
    w = inv(B) @ (M1_ - M0)
    w0_bayes = -0.5 * (M1_ @ inv(B) @ M1_ - M0 @ inv(B) @ M0)
    w0_minimax = w0_bayes

    mu0 = float(w @ M0 + w0_bayes)
    mu1 = float(w @ M1_ + w0_bayes)
    sigma = float(math.sqrt(w @ B @ w))

    p0_b = 1 - norm.cdf((0 - mu0) / sigma)
    p1_b = norm.cdf((0 - mu1) / sigma)
    R_b = 0.5 * (p0_b + p1_b)

    p0_m = p0_b
    p1_m = p1_b
    R_m = R_b

    alpha = 0.05
    t_np = mu0 + sigma * norm.ppf(1 - alpha)
    p0_np = 1 - norm.cdf((t_np - mu0) / sigma)
    p1_np = norm.cdf((t_np - mu1) / sigma)
    R_np = 0.5 * (p0_np + p1_np)

    pred_b_0 = classify_linear(X1, w, w0_bayes)
    pred_b_1 = classify_linear(X2, w, w0_bayes)
    p0_b_emp = float(np.mean(pred_b_0 == 1))
    p1_b_emp = float(np.mean(pred_b_1 == 0))

    pred_np_0 = (X1 @ w + w0_bayes >= t_np).astype(int)
    pred_np_1 = (X2 @ w + w0_bayes >= t_np).astype(int)
    p0_np_emp = float(np.mean(pred_np_0 == 1))
    p1_np_emp = float(np.mean(pred_np_1 == 0))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1[:, 0], X1[:, 1], s=12, alpha=0.6, label='Класс Ω0')
    ax.scatter(X2[:, 0], X2[:, 1], s=12, alpha=0.6, label='Класс Ω1')
    plot_linear_boundary(ax, w, w0_bayes, (-5.5, 4.5), (-5.0, 4.5), 'Байес', '-')
    plot_linear_boundary(ax, w, t_np, (-5.5, 4.5), (-5.0, 4.5), 'Нейман–Пирсон', '--')
    ax.set_title('Лабораторная 2: равные корреляционные матрицы')
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'lab2_equal_boundaries.png', dpi=160); plt.close(fig)

    M0_, M1__, M2_ = M1, M2, M3
    B0_, B1__, B2_ = B1, B2, B3

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1_3[:, 0], X1_3[:, 1], s=12, alpha=0.6, label='Класс Ω0')
    ax.scatter(X2_3[:, 0], X2_3[:, 1], s=12, alpha=0.6, label='Класс Ω1')
    ax.scatter(X3_3[:, 0], X3_3[:, 1], s=12, alpha=0.6, label='Класс Ω2')
    x_min, x_max, y_min, y_max = plot_quadratic_boundaries(ax, [M0_, M1__, M2_], [B0_, B1__, B2_])
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_title('Лабораторная 2: байесовские границы для трёх классов')
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / 'lab2_unequal_boundaries.png', dpi=160); plt.close(fig)

    def pair_err(Xa, Xb, Ma, Ba, Mb, Bb):
        Bi_a = inv(Ba); Bi_b = inv(Bb)
        def pred(X):
            ga = -0.5 * np.log(det(Ba)) - 0.5 * np.einsum('...i,ij,...j->...', X - Ma, Bi_a, X - Ma)
            gb = -0.5 * np.log(det(Bb)) - 0.5 * np.einsum('...i,ij,...j->...', X - Mb, Bi_b, X - Mb)
            return (gb >= ga).astype(int)
        pa = float(np.mean(pred(Xa) == 1))
        pb = float(np.mean(pred(Xb) == 0))
        return pa, pb, 0.5 * (pa + pb)

    pair_results = {
        '0-1': pair_err(X1_3, X2_3, M0_, B0_, M1__, B1__),
        '0-2': pair_err(X1_3, X3_3, M0_, B0_, M2_, B2_),
        '1-2': pair_err(X2_3, X3_3, M1__, B1__, M2_, B2_),
    }
    key_worst = max(pair_results, key=lambda k: pair_results[k][2])
    p_worst = pair_results[key_worst][2]
    rel_err_worst = approx_relative_error(p_worst, N)
    n_5 = required_n_for_relative_error(p_worst, 0.05)

    d_hamming = int(np.sum(p1_proto != p2_proto))
    p_bin_analytic = binary_analytic_error(p1_proto, p2_proto, p=0.3)
    pred_bin1 = binary_bayes_predict(bin1, p1_proto, p2_proto, p=0.3)
    pred_bin2 = binary_bayes_predict(bin2, p1_proto, p2_proto, p=0.3)
    p_bin_emp_0 = float(np.mean(pred_bin1 == 1))
    p_bin_emp_1 = float(np.mean(pred_bin2 == 0))
    p_bin_emp = 0.5 * (p_bin_emp_0 + p_bin_emp_1)

    mean_img1 = bin1.mean(axis=0).reshape(9, 9)
    mean_img2 = bin2.mean(axis=0).reshape(9, 9)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mean_img1, cmap='gray'); axes[0].set_title('Средняя частота proto1'); axes[0].axis('off')
    axes[1].imshow(mean_img2, cmap='gray'); axes[1].set_title('Средняя частота proto2'); axes[1].axis('off')
    fig.tight_layout(); fig.savefig(OUT / 'lab2_binary_means.png', dpi=160); plt.close(fig)

    results = {
        'equal_cov': {
            'bayes': {'p0': p0_b, 'p1': p1_b, 'R': R_b, 'p0_emp': p0_b_emp, 'p1_emp': p1_b_emp},
            'minimax': {'p0': p0_m, 'p1': p1_m, 'R': R_m},
            'np': {'alpha': alpha, 'p0': p0_np, 'p1': p1_np, 'R': R_np, 'p0_emp': p0_np_emp, 'p1_emp': p1_np_emp},
            'boundary': {'w': w.tolist(), 'w0_bayes': float(w0_bayes), 't_np': float(t_np)},
        },
        'unequal_cov': {
            'pair_errors': {k: {'p0': v[0], 'p1': v[1], 'R': v[2]} for k, v in pair_results.items()},
            'worst_pair': key_worst,
            'worst_pair_rel_err': rel_err_worst,
            'n_for_5pct': n_5,
        },
        'binary': {
            'hamming_distance': d_hamming,
            'analytic_p': p_bin_analytic,
            'empirical': {'p0': p_bin_emp_0, 'p1': p_bin_emp_1, 'R': p_bin_emp},
        },
    }
    return results

def write_report_text(lab2_res):
    lines = []
    lines.append('Лабораторная 2')
    lines.append('---------------')
    eq = lab2_res['equal_cov']
    lines.append(f"Байес: p0={eq['bayes']['p0']:.4f}, p1={eq['bayes']['p1']:.4f}, R={eq['bayes']['R']:.4f}")
    lines.append(f"Минимакс: p0={eq['minimax']['p0']:.4f}, p1={eq['minimax']['p1']:.4f}, R={eq['minimax']['R']:.4f}")
    lines.append(f"Нейман–Пирсон (alpha=0.05): p0={eq['np']['p0']:.4f}, p1={eq['np']['p1']:.4f}, R={eq['np']['R']:.4f}")
    lines.append('')
    un = lab2_res['unequal_cov']
    lines.append('Неравные ковариации, пары классов:')
    for k, v in un['pair_errors'].items():
        lines.append(f"  {k}: p0={v['p0']:.4f}, p1={v['p1']:.4f}, R={v['R']:.4f}")
    lines.append(f"Худшая пара: {un['worst_pair']}, относительная погрешность при N=200: {un['worst_pair_rel_err']:.4f}, N для 5%: {un['n_for_5pct']}")
    b = lab2_res['binary']
    lines.append('')
    lines.append(f"Бинарные векторы: Hamming={b['hamming_distance']}, аналитическая ошибка={b['analytic_p']:.6f}, экспериментальная={b['empirical']['R']:.6f}")
    lines.append('')
    lines.append('Файлы графиков:')
    lines.append('lab2_equal_boundaries.png')
    lines.append('lab2_unequal_boundaries.png')
    lines.append('lab2_binary_means.png')
    (OUT / 'results.txt').write_text('\n'.join(lines), encoding='utf-8')

def main():
    data = ensure_data()
    lab2_res = lab2(data)
    write_report_text(lab2_res)
    print(f'Готово. Результаты сохранены в: {OUT}')

if __name__ == '__main__':
    main()

# =============================================================================
# Аналитическая часть (кратко, по формулам из методички)
#
# Лабораторная 2.
# 1) Для нормально распределённого вектора X ~ N(M_l, B_l) байесовский
#    классификатор строится по дискриминантным функциям
#    g_l(x) = ln P(Ω_l) - 1/2 ln|B_l| - 1/2 (x - M_l)^T B_l^{-1} (x - M_l).
#    Решение: выбрать класс с максимальным g_l(x).
#
# 2) При равных корреляционных матрицах B_0 = B_1 = B
#    граница Байеса линейна:
#    d_l(x) = M_l^T B^{-1} x - 1/2 M_l^T B^{-1} M_l + ln P(Ω_l).
#
# 3) Для простейшей матрицы потерь минимаксный классификатор для двух классов
#    совпадает с байесовским, если априорные вероятности выбраны как наиболее
#    неблагоприятные.
#
# 4) Для критерия Неймана–Пирсона выбирается порог, обеспечивающий заданную
#    вероятность ошибки первого рода p_0^*.
#
# 5) Для бинарных независимых признаков:
#    P(X=x|Ω_l) = Π_i p_{li}^{x_i} (1-p_{li})^{1-x_i},
#    а решающее правило сводится к сравнению логарифма отношения правдоподобия
#    с порогом.
# =============================================================================
