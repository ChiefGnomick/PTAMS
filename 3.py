import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr

# Исходные данные
n = 475
alpha = 0.001
correlations = [-0.9, -0.5, 0.0, 0.5, 0.9]

for rho in correlations:
    print(f"\n=== Коэффициент корреляции ρ = {rho} ===")

    # Загрузка сохранённой выборки
    file_name = f"bivariate_sample_rho_{str(rho).replace('.', '').replace('-', 'm')}.npy"
    data = np.load(file_name)
    X = data[:, 0]
    Y = data[:, 1]

    # 3.1 Точечные оценки
    mean_X, mean_Y = np.mean(X), np.mean(Y)
    var_X, var_Y = np.var(X, ddof=1), np.var(Y, ddof=1)
    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
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

    # 3.3 Уравнения регрессий
    a = cov_XY / var_X
    b = mean_Y - a * mean_X
    c = cov_XY / var_Y
    d = mean_X - c * mean_Y

    print(f"Уравнение регрессии Y на X: Y = {a:.4f}·X + {b:.4f}")
    print(f"Уравнение регрессии X на Y: X = {c:.4f}·Y + {d:.4f}")

    # Построение графика с линиями регрессий
    plt.figure(figsize=(6, 6))
    plt.scatter(X, Y, alpha=0.4, label='Данные')

    x_vals = np.linspace(min(X), max(X), 100)
    y_on_x = a * x_vals + b
    y_vals = np.linspace(min(Y), max(Y), 100)
    x_on_y = c * y_vals + d

    plt.plot(x_vals, y_on_x, color='red', label='Y на X')
    plt.plot(x_on_y, y_vals, color='green', linestyle='--', label='X на Y')
    plt.title(f"Диаграмма рассеяния и регрессии (ρ = {rho})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"regression_rho_{str(rho).replace('.', '')}.png", dpi=300)
    plt.close()

print("\nАнализ завершён.")
