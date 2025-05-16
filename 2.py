import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, chi2

# Исходные данные
n = 475
alpha = 0.001
gamma = 0.999

# Параметры экспоненциального распределения
true_lambda = 0.5  # плотность: f(y) = λ * exp(-λy)

# Генерация выборки
Y = np.random.exponential(scale=1 / true_lambda, size=n)

# 2.1 Гистограмма + теоретическая плотность
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(Y, bins='auto', density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Гистограмма')
x = np.linspace(0, np.max(Y), 500)
plt.plot(x, expon.pdf(x, scale=1 / true_lambda), 'r-', lw=2, label='Теоретическая плотность')
plt.title('Гистограмма и теоретическая плотность экспоненциального распределения')
plt.xlabel('Y')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True)
plt.savefig("histogram_2.png", dpi=300, bbox_inches="tight")

# 2.2 Точечные оценки
mean_Y = np.mean(Y)
var_Y = np.var(Y, ddof=1)
print(f"\n2.2 Точечные оценки:")
print(f"Выборочное среднее: {mean_Y:.4f}")
print(f"Выборочная дисперсия: {var_Y:.4f}")

# 2.3 Метод моментов
# Математическое ожидание эксп. распределения = 1/λ => λ = 1/среднее
lambda_mom = 1 / mean_Y
print(f"\n2.3 Метод моментов:")
print(f"Оценка λ методом моментов: {lambda_mom:.4f}")

# 2.4 Доверительный интервал для мат. ожидания и дисперсии
# M(Y) = 1/λ; D(Y) = 1/λ^2; mean_Y — оценка M(Y)
z = chi2.ppf(1 - alpha / 2, df=n)
ci_mean = (mean_Y / np.sqrt(z / n), mean_Y * np.sqrt(z / n))
ci_var = (var_Y * (n - 1) / chi2.ppf(1 - alpha / 2, n - 1),
          var_Y * (n - 1) / chi2.ppf(alpha / 2, n - 1))

print(f"\n2.4 Доверительные интервалы (γ = {gamma}):")
print(f"Мат. ожидание: [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")
print(f"Дисперсия: [{ci_var[0]:.4f}, {ci_var[1]:.4f}]")

# 2.5 Проверка гипотезы о виде распределения (χ²-критерий)
k = int(np.sqrt(n))  # Кол-во интервалов
observed_freq, bin_edges = np.histogram(Y, bins=k)
expected_freq = np.array([
    n * (expon.cdf(bin_edges[i+1], scale=1 / lambda_mom) - expon.cdf(bin_edges[i], scale=1 / lambda_mom))
    for i in range(k)
])
chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
df = k - 1 - 1  # Оценка 1 параметра (λ)
chi2_crit = chi2.ppf(1 - alpha, df)

print(f"\n2.5 Проверка гипотезы χ²:")
print(f"Статистика: {chi2_stat:.4f}")
print(f"Критическое значение при α={alpha}, df={df}: {chi2_crit:.4f}")
if chi2_stat < chi2_crit:
    print("Гипотеза о виде распределения ПРИНИМАЕТСЯ.")
else:
    print("Гипотеза ОТВЕРГАЕТСЯ.")
