import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Загрузка сохранённой выборки
X = np.load("norm_sample.npy")

# Исходные данные
n = len(X)
mu = 1.8
sigma2 = 9.5
sigma = np.sqrt(sigma2)
gamma = 0.999
alpha = 0.001

plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(X, bins='auto', density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Гистограмма')
x_axis = np.linspace(min(X), max(X), 500)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), 'r-', lw=2, label='Теоретическая плотность')
plt.title('Гистограмма и теоретическая плотность нормального распределения')
plt.xlabel('X')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True)
plt.savefig("histogram_1.png", dpi=300, bbox_inches="tight")

# 1.2 Выборочные характеристики
x_mean = np.mean(X)
x_var_biased = np.var(X)        # Смещённая
x_var_unbiased = np.var(X, ddof=1)  # Несмещённая

print(f"\n1.2 Выборочное среднее: {x_mean:.4f}")
print(f"Смещённая дисперсия (ММП): {x_var_biased:.4f}")
print(f"Несмещённая дисперсия: {x_var_unbiased:.4f}")

# 1.3 ММП-оценки (совпадают со смещёнными)
print(f"\n1.3 ММП-оценка ожидания: {x_mean:.4f}")
print(f"ММП-оценка дисперсии: {x_var_biased:.4f} (смещённая)")

# 1.4 Доверительные интервалы
# Для среднего (z-распределение)
z = norm.ppf(1 - alpha / 2)
ci_mean = (x_mean - z * sigma / np.sqrt(n), x_mean + z * sigma / np.sqrt(n))

# Для дисперсии (хи-квадрат распределение)
chi2_left = chi2.ppf(alpha / 2, df=n - 1)
chi2_right = chi2.ppf(1 - alpha / 2, df=n - 1)
ci_var = ((n - 1) * x_var_unbiased / chi2_right, (n - 1) * x_var_unbiased / chi2_left)

print(f"\n1.4 Доверительный интервал для мат. ожидания (γ = {gamma}): [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")
print(f"Доверительный интервал для дисперсии (γ = {gamma}): [{ci_var[0]:.4f}, {ci_var[1]:.4f}]")

# 1.5 Критерий хи-квадрат на нормальность
# Количество интервалов: правило — не менее 5 наблюдений в интервале
k = int(np.sqrt(n))  # Например, Стёрджесс: k ~ sqrt(n)
observed_freq, bin_edges = np.histogram(X, bins=k)
expected_freq = np.array([
    n * (norm.cdf(bin_edges[i+1], mu, sigma) - norm.cdf(bin_edges[i], mu, sigma))
    for i in range(k)
])

chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
df = k - 1 - 2  # Количество степеней свободы (минус 2 параметра: mu и sigma)
chi2_crit = chi2.ppf(1 - alpha, df=df)

print(f"\n1.5 Статистика χ²: {chi2_stat:.4f}")
print(f"Критическое значение χ² при α={alpha}, df={df}: {chi2_crit:.4f}")
if chi2_stat < chi2_crit:
    print("Гипотеза о нормальности распределения ПРИНИМАЕТСЯ.")
else:
    print("Гипотеза о нормальности распределения ОТВЕРГАЕТСЯ.")
