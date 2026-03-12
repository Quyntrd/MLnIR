import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Настройки для графиков
plt.rcParams['figure.figsize'] = (10, 8)

def calc_bayes_error(y_true, y_pred):
    """Экспериментальная оценка вероятности ошибки."""
    return np.mean(y_true != y_pred)

def plot_decision_boundary(X, y, classifier, title):
    """Визуализация решающей границы."""
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = classifier(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm, s=20)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)

# --- 1. Случай равных корреляционных матриц (Линейный классификатор) ---
def solve_linear_case(X1, X2, M1, M2, B):
    """
    Построение границы при B1 = B2 = B.
    Решающее правило (стр. 18): d(x) = W.T * x + w0
    """
    B_inv = np.linalg.inv(B)
    
    # W = B^-1 * (M1 - M2) -- вектор весов (формула 2.15-2.17 в теории)
    W = B_inv @ (M1 - M2)
    
    # w0 = -0.5 * (M1.T * B^-1 * M1 - M2.T * B^-1 * M2)
    w0 = -0.5 * (M1.T @ B_inv @ M1 - M2.T @ B_inv @ M2)
    
    def classify(X):
        # Линейная дискриминантная функция
        return (X @ W + w0 > 0).astype(int)
    
    return classify

# --- 2. Случай разных корреляционных матриц (Квадратичный классификатор) ---
def solve_quadratic_case(X_list, M_list, B_list, P_prior=None):
    """
    Построение границы при B1 != B2.
    Решающее правило на основе максимума апостериорной вероятности.
    """
    if P_prior is None:
        P_prior = [1/len(M_list)] * len(M_list)
        
    def classify(X):
        scores = []
        for i in range(len(M_list)):
            # f(x) = 1/((2pi)^n/2 * |B|^1/2) * exp(-0.5 * (x-M).T * B^-1 * (x-M))
            # Используем логарифм плотности для численной стабильности
            rv = multivariate_normal(M_list[i], B_list[i])
            # ln(P(wi|x)) ~ ln(f(x|wi)) + ln(P(wi))
            scores.append(rv.logpdf(X) + np.log(P_prior[i]))
        
        return np.argmax(scores, axis=0)
    
    return classify

# --- 3. Классификатор для бинарных векторов ---
def solve_binary_case(X, p1, p2):
    """
    Байесовский классификатор для бинарных признаков (стр. 27, п. 4).
    p1, p2 - векторы вероятностей появления '1' в каждой компоненте.
    """
    def classify(X):
        # Отношение правдоподобия для независимых признаков Бернулли
        # L(x) = Product [ (pi^xi) * (1-pi)^(1-xi) ]
        log_lik1 = X @ np.log(p1) + (1 - X) @ np.log(1 - p1)
        log_lik2 = X @ np.log(p2) + (1 - X) @ np.log(1 - p2)
        return (log_lik1 > log_lik2).astype(int)
    
    return classify

# --- Пример вызова и визуализации ---
if __name__ == "__main__":
    # Заглушка данных (в реальности загрузить из .npy)
    N = 200
    M1, B1 = np.array([1, 1]), np.array([[1, 0.5], [0.5, 1]])
    M2, B2 = np.array([-1, -1]), np.array([[1.5, -0.3], [-0.3, 1.5]])
    
    X1 = np.random.multivariate_normal(M1, B1, N)
    X2 = np.random.multivariate_normal(M2, B2, N)
    X = np.vstack([X1, X2])
    y = np.array([0]*N + [1]*N)
    
    # Квадратичный классификатор
    quad_clf = solve_quadratic_case([M1, M2], [B1, B2])
    
    plt.figure()
    plot_decision_boundary(X, y, quad_clf, "Байесовская граница (неравные матрицы)")
    plt.show()
    
    # Оценка ошибки
    y_pred = quad_clf(X)
    error = calc_bayes_error(y, y_pred)
    print(f"Экспериментальная вероятность ошибки: {error:.4f}")