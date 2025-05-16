import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, pearsonr

np.random.seed(0)

# === ПАРАМЕТРЫ ===
n = 475  # объём выборки
mu_X, mu_Y = 0, 0
sigma_X, sigma_Y = 1, 1
alpha = 0.001  # уровень значимости

# === ТЕСТОВЫЕ КОРРЕЛЯЦИИ ===
correlations = [-0.9, -0.5, 0.0, 0.5, 0.9]

for rho in correlations:
    print(f"\n=== Коэффициент корреляции ρ = {rho} ===")

    # Ковариационная матрица
    cov_matrix = [
        [sigma_X**2, rho * sigma_X * sigma_Y],
        [rho * sigma_X * sigma_Y, sigma_Y**2]
    ]

    # Генерация двумерной нормальной выборки
    data = np.random.multivariate_normal([mu_X, mu_Y], cov_matrix, size=n)
    X = data[:, 0]
    Y = data[:, 1]

    # 3.1 Точечные оценки и корреляция
    mean_X, mean_Y = np.mean(X), np.mean(Y)
    var_X, var_Y = np.var(X, ddof=1), np.var(Y, ddof=1)
    r_xy, p_val = pearsonr(X, Y)

    print(f"Оценка M(X): {mean_X:.4f}, Var(X): {var_X:.4f}")
    print(f"Оценка M(Y): {mean_Y:.4f}, Var(Y): {var_Y:.4f}")
    print(f"Оценка корреляции r: {r_xy:.4f}, p-value: {p_val:.4f}")

    # 3.2 Проверка гипотезы о независимости
    t_stat = r_xy * np.sqrt((n - 2) / (1 - r_xy**2))
    t_crit = norm.ppf(1 - alpha / 2)
    print(f"t-статистика: {t_stat:.4f}, критическое значение: ±{t_crit:.4f}")
    if abs(t_stat) < t_crit:
        print("Гипотеза о независимости ПРИНИМАЕТСЯ.")
    else:
        print("Гипотеза о независимости ОТВЕРГАЕТСЯ.")

    # 3.3 Построение графика рассеяния
    plt.figure(figsize=(5, 5))
    plt.scatter(X, Y, alpha=0.5, label=f"ρ = {rho}")
    plt.title(f"Диаграмма рассеяния (ρ = {rho})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"scatter_rho_{str(rho).replace('.', '')}.png", dpi=300)

print("\nАнализ завершён.")
