import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, t

# === Настройки === #
alpha = 0.05
n = 475
mu_X, mu_Y = 1.8, -4.5
sigma_X, sigma_Y = np.sqrt(9.5), np.sqrt(3.61)

# === 3.0 Загрузка выборки === #
data = np.load("bivariate_sample_task3.npy")
X, Y = data[:, 0], data[:, 1]

# === 3.1 Точечные оценки === #
mean_X = np.mean(X)
mean_Y = np.mean(Y)
var_X = np.var(X, ddof=1)
var_Y = np.var(Y, ddof=1)
r_xy = np.corrcoef(X, Y)[0, 1]

print("3.1 Точечные оценки:")
print(f"  M[X] = {mean_X:.4f},  D[X] = {var_X:.4f}")
print(f"  M[Y] = {mean_Y:.4f},  D[Y] = {var_Y:.4f}")
print(f"  r(X,Y) = {r_xy:.4f}")

# === 3.2 Проверка гипотезы независимости === #
k = int(np.sqrt(n))
x_bins = np.linspace(min(X), max(X), k)
y_bins = np.linspace(min(Y), max(Y), k)

contingency_table, _, _ = np.histogram2d(X, Y, bins=[x_bins, y_bins])

# Удаление нулевых строк и столбцов
nonzero_rows = ~np.all(contingency_table == 0, axis=1)
nonzero_cols = ~np.all(contingency_table == 0, axis=0)
clean_table = contingency_table[np.ix_(nonzero_rows, nonzero_cols)]

if clean_table.shape[0] < 2 or clean_table.shape[1] < 2:
    print("\n3.2 Недостаточно данных для критерия χ²: таблица слишком мала после очистки.")
else:
    chi2_stat, p_val, dof, _ = chi2_contingency(clean_table)
    print("\n3.2 Проверка независимости (хи-квадрат):")
    print(f"  χ² = {chi2_stat:.4f}, df = {dof}, p = {p_val:.4f}")
    if p_val > alpha:
        print("  Гипотеза о независимости ПРИНИМАЕТСЯ.")
    else:
        print("  Гипотеза о независимости ОТВЕРГАЕТСЯ.")

# === Проверка значимости коэффициента корреляции === #
t_stat = r_xy * np.sqrt((n - 2) / (1 - r_xy**2))
t_crit = t.ppf(1 - alpha / 2, df=n - 2)

print("\nКритерий значимости корреляции:")
print(f"  t = {t_stat:.4f}, критическое значение t = ±{t_crit:.4f}")
if abs(t_stat) > t_crit:
    print("  Корреляция статистически ЗНАЧИМА.")
else:
    print("  Корреляция статистически НЕзначима.")

# === 3.3 Влияние корреляции на выборочные характеристики === #
rhos = [-0.9, -0.5, 0.0, 0.5, 0.9]
print("\n3.3 Анализ влияния корреляции:")
print("  ρ\tM[X]\tD[X]\tM[Y]\tD[Y]\tr выборки")

for rho in rhos:
    cov = rho * sigma_X * sigma_Y
    cov_matrix = [[sigma_X**2, cov],
                  [cov, sigma_Y**2]]
    
    sample = np.random.multivariate_normal([mu_X, mu_Y], cov_matrix, size=n)
    X_rho, Y_rho = sample[:, 0], sample[:, 1]
    
    mean_X_rho = np.mean(X_rho)
    mean_Y_rho = np.mean(Y_rho)
    var_X_rho = np.var(X_rho, ddof=1)
    var_Y_rho = np.var(Y_rho, ddof=1)
    r_sample = np.corrcoef(X_rho, Y_rho)[0, 1]
    
    print(f"  {rho:+.1f}\t{mean_X_rho:.2f}\t{var_X_rho:.2f}\t{mean_Y_rho:.2f}\t{var_Y_rho:.2f}\t{r_sample:+.3f}")
    
    # Визуализация для каждого случая
    plt.figure(figsize=(6, 5))
    plt.scatter(X_rho, Y_rho, alpha=0.6, edgecolors='k')
    plt.title(f"Диаграмма рассеяния (ρ = {rho})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"scatter_rho_{str(rho).replace('.', '').replace('-', 'm')}.png", dpi=200, bbox_inches="tight")
    plt.close()

# === Финальная визуализация исходной выборки === #
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.6, edgecolors='k')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Диаграмма рассеяния X и Y (исходная выборка)")
plt.grid(True)
plt.savefig("scatter_task3.png", dpi=300, bbox_inches="tight")
