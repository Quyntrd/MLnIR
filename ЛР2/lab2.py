
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from scipy.stats import norm

from utility import (
    simulate_normal,
    simulate_binary_vector,
    estimate_params,
    bhattacharyya_distance,
    mahalanobis_distance,
)

# -----------------------------------------------------------------------------
# Output folders
# -----------------------------------------------------------------------------
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT / "logs"
LOG_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
np.random.seed(0)

N = 200
N_ERR = 50000

M1 = np.array([1.0, 0.0])
M2 = np.array([-1.0, -1.0])
M3 = np.array([1.0, 2.0])

B_equal = np.array([[1.0, 0.1],
                    [0.1, 1.0]])

B1 = np.array([[1.0, -0.1],
               [-0.1, 1.0]])
B2 = np.array([[1.0, 0.1],
               [0.1, 1.0]])
B3 = np.array([[1.0, 0.7],
               [0.7, 1.0]])

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
        "eq1": LOG_DIR / "normal_eq_1.npy",
        "eq2": LOG_DIR / "normal_eq_2.npy",
        "3_1": LOG_DIR / "normal_3_1.npy",
        "3_2": LOG_DIR / "normal_3_2.npy",
        "3_3": LOG_DIR / "normal_3_3.npy",
        "b1": LOG_DIR / "binary_1.npy",
        "b2": LOG_DIR / "binary_2.npy",
    }

    if all(p.exists() for p in files.values()):
        return {
            "X1": np.load(files["eq1"]),
            "X2": np.load(files["eq2"]),
            "X1_3": np.load(files["3_1"]),
            "X2_3": np.load(files["3_2"]),
            "X3_3": np.load(files["3_3"]),
            "bin1": np.load(files["b1"]),
            "bin2": np.load(files["b2"]),
        }

    X1 = simulate_normal(M1, B_equal, N)
    X2 = simulate_normal(M2, B_equal, N)
    X1_3 = simulate_normal(M1, B1, N)
    X2_3 = simulate_normal(M2, B2, N)
    X3_3 = simulate_normal(M3, B3, N)
    bin1 = simulate_binary_vector(p1_proto, N, 0.3, rng).T
    bin2 = simulate_binary_vector(p2_proto, N, 0.3, rng).T

    np.save(files["eq1"], X1)
    np.save(files["eq2"], X2)
    np.save(files["3_1"], X1_3)
    np.save(files["3_2"], X2_3)
    np.save(files["3_3"], X3_3)
    np.save(files["b1"], bin1)
    np.save(files["b2"], bin2)

    return {
        "X1": X1,
        "X2": X2,
        "X1_3": X1_3,
        "X2_3": X2_3,
        "X3_3": X3_3,
        "bin1": bin1,
        "bin2": bin2,
    }


def classify_linear(X: np.ndarray, w: np.ndarray, threshold: float) -> np.ndarray:
    """Return class index 1 if w^T x >= threshold, else 0."""
    return (X @ w >= threshold).astype(int)


def plot_linear_boundary(ax, w: np.ndarray, threshold: float, xlim, label, style='-'):
    xs = np.linspace(xlim[0], xlim[1], 250)
    if abs(w[1]) < 1e-12:
        x = threshold / w[0]
        ax.axvline(x, linestyle=style, linewidth=2, label=label)
        return
    ys = (threshold - w[0] * xs) / w[1]
    ax.plot(xs, ys, style, linewidth=2, label=label)


def gaussian_score(X: np.ndarray, M: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Bayes score for a normal class with equal priors:
    g(x) = -1/2 ln|B| - 1/2 (x-M)^T B^{-1} (x-M)
    """
    Bi = inv(B)
    return -0.5 * np.log(det(B)) - 0.5 * np.einsum("...i,ij,...j->...", X - M, Bi, X - M)


def plot_quadratic_regions(ax, Ms, Bs, xlim=(-5.5, 4.5), ylim=(-4.5, 4.5)):
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 500),
        np.linspace(ylim[0], ylim[1], 500),
    )
    grid = np.stack([xx, yy], axis=-1)

    scores = []
    for M, B in zip(Ms, Bs):
        Bi = inv(B)
        const = -0.5 * np.log(det(B))
        s = const - 0.5 * np.einsum("...i,ij,...j->...", grid - M, Bi, grid - M)
        scores.append(s)

    scores = np.stack(scores, axis=-1)
    labels = np.argmax(scores, axis=-1)

    ax.contourf(
        xx, yy, labels,
        levels=[-0.5, 0.5, 1.5, 2.5],
        alpha=0.22
    )

    ax.contour(xx, yy, scores[..., 0] - scores[..., 1], levels=[0], colors="k", linewidths=1.8)
    ax.contour(xx, yy, scores[..., 0] - scores[..., 2], levels=[0], colors="k", linewidths=1.8, linestyles="--")
    ax.contour(xx, yy, scores[..., 1] - scores[..., 2], levels=[0], colors="k", linewidths=1.8, linestyles=":")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return labels


def binary_bayes_predict(
    X: np.ndarray,
    proto_a: np.ndarray,
    proto_b: np.ndarray,
    p: float = 0.3,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> np.ndarray:
    """
    Bayes classifier for binary vectors under independent bit flips.
    Returns 0 for proto_a, 1 for proto_b.
    """
    pa = np.where(proto_a == 1, 1.0 - p, p)
    pb = np.where(proto_b == 1, 1.0 - p, p)

    llr = (
        X @ np.log(pa / pb)
        + (1.0 - X) @ np.log((1.0 - pa) / (1.0 - pb))
        + math.log(prior_a / prior_b)
    )
    return (llr < 0).astype(int)


def binary_analytic_error(proto_a: np.ndarray, proto_b: np.ndarray, p: float = 0.3) -> float:
    """
    Exact Bayes error for equal priors and independent bit flips.
    If the prototypes differ in d positions, the error is the tail of Bin(d, 1-p).
    """
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


def relative_error_estimate(p_hat: float, n: int) -> float:
    """Approximate relative standard error of Bernoulli proportion estimate."""
    if p_hat <= 0:
        return float("inf")
    return math.sqrt((1.0 - p_hat) / (p_hat * n))


def required_n_for_relative_error(p_hat: float, eps: float = 0.05) -> int:
    """N needed so that relative error does not exceed eps."""
    if p_hat <= 0:
        return math.inf
    return int(math.ceil((1.0 - p_hat) / (p_hat * eps**2)))

def pairwise_gaussian_errors(Xa, Xb, Ma, Ba, Mb, Bb):
    """
    Empirical Bayes error for a pair of normal classes with equal priors.
    """
    score_a_a = gaussian_score(Xa, Ma, Ba)
    score_b_a = gaussian_score(Xa, Mb, Bb)
    pred_a = (score_b_a >= score_a_a).astype(int)  # 1 means class b

    score_a_b = gaussian_score(Xb, Ma, Ba)
    score_b_b = gaussian_score(Xb, Mb, Bb)
    pred_b = (score_b_b >= score_a_b).astype(int)

    p_a_to_b = float(np.mean(pred_a == 1))
    p_b_to_a = float(np.mean(pred_b == 0))
    return p_a_to_b, p_b_to_a, 0.5 * (p_a_to_b + p_b_to_a)


def lab2(data: Dict[str, np.ndarray]):
    X1, X2 = data["X1"], data["X2"]
    X1_3, X2_3, X3_3 = data["X1_3"], data["X2_3"], data["X3_3"]
    bin1, bin2 = data["bin1"], data["bin2"]

    X1_eval = simulate_normal(M1, B_equal, N_ERR)
    X2_eval = simulate_normal(M2, B_equal, N_ERR)
    X1_3_eval = simulate_normal(M1, B1, N_ERR)
    X2_3_eval = simulate_normal(M2, B2, N_ERR)
    X3_3_eval = simulate_normal(M3, B3, N_ERR)
    bin1_eval = simulate_binary_vector(p1_proto, N_ERR, 0.3, rng).T
    bin2_eval = simulate_binary_vector(p2_proto, N_ERR, 0.3, rng).T

    M0, M1_ = M1, M2
    B = B_equal

    B_inv = inv(B)
    w = B_inv @ (M1_ - M0)

    prior0 = 0.5
    prior1 = 0.5

    w0_bayes = -0.5 * (M1_ @ B_inv @ M1_ - M0 @ B_inv @ M0) + math.log(prior1 / prior0)

    mu0 = float(w @ M0 + w0_bayes)
    mu1 = float(w @ M1_ + w0_bayes)
    sigma = float(math.sqrt(w @ B @ w))

    # -------------------------------------------------
    # Bayes classifier
    # -------------------------------------------------

    tau_b = 0.0

    p0_b = 1.0 - norm.cdf((tau_b - mu0) / sigma)
    p1_b = norm.cdf((tau_b - mu1) / sigma)

    R_b = prior0 * p0_b + prior1 * p1_b

    # -------------------------------------------------
    # Minimax classifier
    # -------------------------------------------------

    tau_mm = 0.5 * (mu0 + mu1)

    p0_m = 1.0 - norm.cdf((tau_mm - mu0) / sigma)
    p1_m = norm.cdf((tau_mm - mu1) / sigma)

    R_m = 0.5 * (p0_m + p1_m)

    # Neyman-Pearson with p0* = 0.05
    alpha = 0.05
    tau_np = mu0 + sigma * norm.ppf(1.0 - alpha)
    p0_np = 1.0 - norm.cdf((tau_np - mu0) / sigma)
    p1_np = norm.cdf((tau_np - mu1) / sigma)
    R_np = 0.5 * (p0_np + p1_np)

    # Empirical estimates on the generated sample
    gX1 = X1_eval @ w + w0_bayes
    gX2 = X2_eval @ w + w0_bayes

    p0_b_emp = float(np.mean(gX1 >= tau_b))
    p1_b_emp = float(np.mean(gX2 < tau_b))

    p0_m_emp = float(np.mean(gX1 >= tau_mm))
    p1_m_emp = float(np.mean(gX2 < tau_mm))

    p0_np_emp = float(np.mean(gX1 >= tau_np))
    p1_np_emp = float(np.mean(gX2 < tau_np))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1[:, 0], X1[:, 1], s=12, alpha=0.6, label="Класс Ω0")
    ax.scatter(X2[:, 0], X2[:, 1], s=12, alpha=0.6, label="Класс Ω1")
    plot_linear_boundary(ax, w, 0.0 - w0_bayes, (-5.5, 4.5), "Байес", "-")
    plot_linear_boundary(ax, w, tau_mm - w0_bayes, (-5.5, 4.5), "Минимакс", "--")
    plot_linear_boundary(ax, w, tau_np - w0_bayes, (-5.5, 4.5), "Нейман–Пирсон", ":")
    ax.set_title("Лабораторная 2: равные корреляционные матрицы")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lab2_equal_boundaries.png", dpi=160)
    plt.close(fig)

    Ms = [M1, M2, M3]
    Bs = [B1, B2, B3]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X1_3[:, 0], X1_3[:, 1], s=12, alpha=0.6, label="Класс Ω0")
    ax.scatter(X2_3[:, 0], X2_3[:, 1], s=12, alpha=0.6, label="Класс Ω1")
    ax.scatter(X3_3[:, 0], X3_3[:, 1], s=12, alpha=0.6, label="Класс Ω2")
    plot_quadratic_regions(ax, Ms, Bs)
    ax.set_title("Лабораторная 2: байесовские границы для трех классов")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "lab2_unequal_boundaries.png", dpi=160)
    plt.close(fig)

    pair_results = {
        "0-1": pairwise_gaussian_errors(X1_3_eval, X2_3_eval, M1, B1, M2, B2),
        "0-2": pairwise_gaussian_errors(X1_3_eval, X3_3_eval, M1, B1, M3, B3),
        "1-2": pairwise_gaussian_errors(X2_3_eval, X3_3_eval, M2, B2, M3, B3),
    }

    pair_summary = {}
    for k, (p0, p1, R) in pair_results.items():
        p_rel = relative_error_estimate(R, N)
        n_5 = required_n_for_relative_error(R, 0.05)
        pair_summary[k] = {
            "p0": p0,
            "p1": p1,
            "R": R,
            "rel_err": p_rel,
            "n_for_5pct": n_5,
        }

    worst_pair = max(pair_summary, key=lambda k: pair_summary[k]["R"])

    p_bin = 0.3
    bin_pred1 = binary_bayes_predict(bin1_eval, p1_proto, p2_proto, p=p_bin)
    bin_pred2 = binary_bayes_predict(bin2_eval, p1_proto, p2_proto, p=p_bin)

    bin_emp_0 = float(np.mean(bin_pred1 == 1))
    bin_emp_1 = float(np.mean(bin_pred2 == 0))
    bin_emp_R = 0.5 * (bin_emp_0 + bin_emp_1)

    bin_analytic = binary_analytic_error(p1_proto, p2_proto, p=p_bin)
    hamming_d = int(np.sum(p1_proto != p2_proto))

    mean_img1 = bin1.mean(axis=0).reshape(9, 9)
    mean_img2 = bin2.mean(axis=0).reshape(9, 9)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mean_img1, cmap="gray")
    axes[0].set_title("Средняя частота proto1")
    axes[0].axis("off")

    axes[1].imshow(mean_img2, cmap="gray")
    axes[1].set_title("Средняя частота proto2")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(OUT / "lab2_binary_means.png", dpi=160)
    plt.close(fig)

    results = {
        "equal_cov": {
            "bayes": {
                "p0": p0_b,
                "p1": p1_b,
                "R": R_b,
                "p0_emp": p0_b_emp,
                "p1_emp": p1_b_emp,
                "threshold": 0.0,
            },
            "minimax": {
                "p0": p0_m,
                "p1": p1_m,
                "R": R_m,
                "p0_emp": p0_m_emp,
                "p1_emp": p1_m_emp,
                "threshold": tau_mm,
            },
            "np": {
                "alpha": alpha,
                "p0": p0_np,
                "p1": p1_np,
                "R": R_np,
                "p0_emp": p0_np_emp,
                "p1_emp": p1_np_emp,
                "threshold": tau_np,
            },
            "w": w.tolist(),
            "w0": float(w0_bayes),
        },
        "unequal_cov": {
            "pairwise": pair_summary,
            "worst_pair": worst_pair,
        },
        "binary": {
            "hamming_distance": hamming_d,
            "analytic_R": bin_analytic,
            "empirical": {
                "p0": bin_emp_0,
                "p1": bin_emp_1,
                "R": bin_emp_R,
            },
            "p": p_bin,
        },
    }
    return results


def write_report_text(res):
    lines = []
    lines.append("Лабораторная 2")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1) Две нормальные выборки с равными корреляционными матрицами")
    eq = res["equal_cov"]
    lines.append(f"Bayes:     p0={eq['bayes']['p0']:.6f}, p1={eq['bayes']['p1']:.6f}, R={eq['bayes']['R']:.6f}")
    lines.append(f"Minimax:   p0={eq['minimax']['p0']:.6f}, p1={eq['minimax']['p1']:.6f}, R={eq['minimax']['R']:.6f}")
    lines.append(f"NP (0.05): p0={eq['np']['p0']:.6f}, p1={eq['np']['p1']:.6f}, R={eq['np']['R']:.6f}")
    lines.append(f"w = {np.array(eq['w'])}")
    lines.append(f"w0 = {eq['w0']:.6f}")
    lines.append("")

    lines.append("2) Три нормальные выборки с неравными корреляционными матрицами")
    for k, v in res["unequal_cov"]["pairwise"].items():
        lines.append(
            f"pair {k}: p0={v['p0']:.6f}, p1={v['p1']:.6f}, R={v['R']:.6f}, "
            f"rel_err(N=200)={v['rel_err']:.6f}, N_5%={v['n_for_5pct']}"
        )
    lines.append(f"worst pair: {res['unequal_cov']['worst_pair']}")
    lines.append("")

    lines.append("3) Бинарные векторы")
    b = res["binary"]
    lines.append(f"Hamming distance = {b['hamming_distance']}")
    lines.append(f"Analytic Bayes error = {b['analytic_R']:.8f}")
    lines.append(f"Empirical p0 = {b['empirical']['p0']:.8f}")
    lines.append(f"Empirical p1 = {b['empirical']['p1']:.8f}")
    lines.append(f"Empirical R  = {b['empirical']['R']:.8f}")
    lines.append("")

    lines.append("Файлы графиков:")
    lines.append("lab2_equal_boundaries.png")
    lines.append("lab2_unequal_boundaries.png")
    lines.append("lab2_binary_means.png")
    lines.append("")
    lines.append("Сохраненные выборки:")
    for name in [
        "normal_eq_1.npy",
        "normal_eq_2.npy",
        "normal_3_1.npy",
        "normal_3_2.npy",
        "normal_3_3.npy",
        "binary_1.npy",
        "binary_2.npy",
    ]:
        lines.append(name)

    (OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    data = ensure_data()
    res = lab2(data)
    write_report_text(res)
    print(f"Done. Results saved to: {OUT}")


if __name__ == "__main__":
    main()

# =============================================================================
# Аналитическая часть по лабораторной 2
#
# 1) Нормальные классы.
# Если X ~ N(M_l, B_l), то байесовская дискриминантная функция для класса l:
# g_l(x) = ln P(Ω_l) - 1/2 ln|B_l| - 1/2 (x - M_l)^T B_l^{-1} (x - M_l).
#
# Решение: выбрать класс с максимальным g_l(x).
#
# Для случая B_0 = B_1 = B и равных априорных вероятностей:
# g(x) = w^T x + w0,
# w = B^{-1}(M_1 - M_0),
# w0 = -1/2(M_1^T B^{-1} M_1 - M_0^T B^{-1} M_0).
#
# Байесовское правило: класс 1, если g(x) >= 0.
#
# 2) Минимаксный классификатор.
# Для двух классов и простейшей матрицы потерь минимаксный классификатор
# можно получить как байесовский с "наименее благоприятным" порогом.
# В случае равной ковариационной матрицы граница остаётся линейной:
# g(x) = w^T x + w0 >= tau_mm,
# где tau_mm выбирается так, чтобы p01 = p10.
# Для нормального распределения проекции это даёт:
# tau_mm = (mu0 + mu1)/2,
# mu0 = w^T M0 + w0,
# mu1 = w^T M1 + w0.
#
# 3) Критерий Неймана–Пирсона.
# Порог выбирается из условия p0 = p0*:
# tau_np = mu0 + sigma * Phi^{-1}(1 - p0*),
# где sigma^2 = w^T B w.
#
# 4) Бинарные векторы.
# Пусть компоненты независимы и каждая с вероятностью p инвертируется.
# Тогда:
# P(X = x | Ω_l) = Π_i q_{li}^{x_i}(1 - q_{li})^{1 - x_i},
# где q_{li} = 1 - p, если эталонная компонента равна 1,
# и q_{li} = p, если эталонная компонента равна 0.
#
# При равных априорных вероятностях решение сводится к сравнению логарифма
# отношения правдоподобия с нулем.
#
# Если прототипы различаются в d позициях, точная вероятность ошибки:
# P_err = Σ_{k < d/2} C(d, k)(1-p)^k p^{d-k} + 0.5 * [tie term if d even].
# =============================================================================
