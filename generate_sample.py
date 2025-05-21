import numpy as np

# --- 1. Нормальное распределение ---
n = 475
mu = 1.8
sigma2 = 9.5
sigma = np.sqrt(sigma2)
norm_sample = np.random.normal(loc=mu, scale=sigma, size=n)
np.save("norm_sample.npy", norm_sample)
print("Сохранено: norm_sample.npy")

# --- 2. Экспоненциальное распределение ---
a = 4
N = 300

# Инверсный метод: Y = U^(1/a), где U ~ U[0,1]
U = np.random.uniform(0, 1, size=N)
Y = U**(1/a)
np.save("power_sample.npy", Y)
print("Сохранено: power_sample.npy")

# --- 3. Двумерные нормальные выборки с разными корреляциями ---
