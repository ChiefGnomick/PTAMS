import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Загрузка выборки
data = np.load("bivariate_sample_task3.npy")
X, Y = data[:, 0], data[:, 1]
n = len(X)
alpha = 0.05


# --- 3.1. Точечные оценки ---
mean_X = np.mean(X)
mean_Y = np.mean(Y)
var_X = np.var(X, ddof=1)
var_Y = np.var(Y, ddof=1)

# Коэффициент корреляции Пирсона
r_xy = np.corrcoef(X, Y)[0, 1]

print(f"3.1 Точечные оценки:")
print(f"  M[X] = {mean_X:.4f},  D[X] = {var_X:.4f}")
print(f"  M[Y] = {mean_Y:.4f},  D[Y] = {var_Y:.4f}")
print(f"  r(X,Y) = {r_xy:.4f}")

# --- 3.2 Проверка гипотезы независимости (с обработкой нулей) ---
k = int(np.sqrt(n))
x_bins = np.linspace(min(X), max(X), k)
y_bins = np.linspace(min(Y), max(Y), k)

contingency_table, _, _ = np.histogram2d(X, Y, bins=[x_bins, y_bins])

# Удалим строки и столбцы, содержащие только нули
nonzero_rows = ~np.all(contingency_table == 0, axis=1)
nonzero_cols = ~np.all(contingency_table == 0, axis=0)
clean_table = contingency_table[np.ix_(nonzero_rows, nonzero_cols)]

# Проверим размер таблицы
if clean_table.shape[0] < 2 or clean_table.shape[1] < 2:
    print("\n3.2 Недостаточно данных для критерия χ²: после очистки осталась слишком малая таблица.")
else:
    chi2_stat, p_val, dof, _ = chi2_contingency(clean_table)

    print(f"\n3.2 Проверка независимости (хи-квадрат):")
    print(f"  χ² = {chi2_stat:.4f}, df = {dof}, p = {p_val:.4f}")
    if p_val > alpha:
        print("  Гипотеза о независимости ПРИНИМАЕТСЯ.")
    else:
        print("  Гипотеза о независимости ОТВЕРГАЕТСЯ.")

# --- 3.3 Влияние корреляции ---
# Здесь можно варьировать rho и построить графики (см. дальше при необходимости)

# --- Визуализация ---
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.6, edgecolors='k')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Диаграмма рассеяния X и Y")
plt.grid(True)
plt.savefig("scatter_task3.png", dpi=300, bbox_inches="tight")
