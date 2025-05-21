import numpy as np

# --- 1. Нормальное распределение ---
n = 475
mu = 1.8
sigma2 = 9.5
sigma = np.sqrt(sigma2)
norm_sample = np.random.normal(loc=mu, scale=sigma, size=n)
np.save("norm_sample.npy", norm_sample)
print("Сохранено: norm_sample.npy")

# --- 2. Степенное распределение Y = U^(1/a), где U ~ U[0,1] ---
a = 4
N = 300
U = np.random.uniform(0, 1, size=N)
Y = U**(1/a)
np.save("power_sample.npy", Y)
print("Сохранено: power_sample.npy")

# --- 3. Двумерное нормальное распределение с заданной корреляцией ---
# Параметры для X и Y
mu_X = 1.8
sigma2_X = 9.5
sigma_X = np.sqrt(sigma2_X)

mu_Y = -4.5
sigma2_Y = 3.61
sigma_Y = np.sqrt(sigma2_Y)

rho = -0.18  # коэффициент корреляции

# Ковариационная матрица
cov_XY = rho * sigma_X * sigma_Y
cov_matrix = [
    [sigma2_X, cov_XY],
    [cov_XY, sigma2_Y]
]

# Генерация выборки
bivariate_sample = np.random.multivariate_normal([mu_X, mu_Y], cov_matrix, size=n)
np.save("bivariate_sample_task3.npy", bivariate_sample)
print("Сохранено: bivariate_sample_task3.npy")
