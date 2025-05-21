import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Загрузка выборки
Y = np.load("power_sample.npy")

# Исходные параметры
a_real = 4
alpha = 0.05
gamma = 0.92
N = len(Y)

# --- 2.1 Гистограмма и плотность ---
plt.figure(figsize=(10, 6))
count, bins, _ = plt.hist(Y, bins='auto', density=True, alpha=0.6, edgecolor='black', color='lightgreen', label='Гистограмма')

# Теоретическая плотность: f(x) = a * x^(a-1)
x_vals = np.linspace(0, 1, 500)
f_x = a_real * x_vals**(a_real - 1)
plt.plot(x_vals, f_x, 'r-', lw=2, label='Теоретическая плотность')

plt.title("Гистограмма и теоретическая плотность")
plt.xlabel("Y")
plt.ylabel("Плотность")
plt.grid(True)
plt.legend()
plt.savefig("histogram_2.png", dpi=300, bbox_inches="tight")

# --- 2.2 Выборочные оценки ---
mean_Y = np.mean(Y)
var_Y = np.var(Y, ddof=1)
print(f"\n2.2 Среднее: {mean_Y:.4f}")
print(f"Дисперсия: {var_Y:.4f}")

# --- 2.3 Метод моментов ---
# Математическое ожидание: M[X] = a / (a + 1)
# Решаем уравнение: mean_Y = a / (a + 1)
# => a = mean_Y / (1 - mean_Y)
a_moment = mean_Y / (1 - mean_Y)
print(f"\n2.3 Оценка параметра a методом моментов: {a_moment:.4f}")

# --- 2.4 Доверительные интервалы ---
from scipy.stats import norm

z = norm.ppf(1 - alpha / 2)
ci_mean = (mean_Y - z * np.sqrt(var_Y / N), mean_Y + z * np.sqrt(var_Y / N))

chi2_left = chi2.ppf(alpha / 2, df=N - 1)
chi2_right = chi2.ppf(1 - alpha / 2, df=N - 1)
ci_var = ((N - 1) * var_Y / chi2_right, (N - 1) * var_Y / chi2_left)

print(f"\n2.4 Доверительный интервал для среднего: [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")
print(f"Доверительный интервал для дисперсии: [{ci_var[0]:.4f}, {ci_var[1]:.4f}]")

# --- 2.5 Критерий Пирсона ---
k = int(np.sqrt(N))
observed_freq, bin_edges = np.histogram(Y, bins=k)
expected_freq = np.array([
    N * (bin_edges[i+1]**a_real - bin_edges[i]**a_real)
    for i in range(k)
])

chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
df = k - 1 - 1  # Один параметр a оценивался
chi2_crit = chi2.ppf(1 - alpha, df)

print(f"\n2.5 χ²-статистика: {chi2_stat:.4f}")
print(f"Критическое значение χ² (df={df}): {chi2_crit:.4f}")
if chi2_stat < chi2_crit:
    print("Гипотеза о распределении ПРИНИМАЕТСЯ.")
else:
    print("Гипотеза ОТВЕРГАЕТСЯ.")
