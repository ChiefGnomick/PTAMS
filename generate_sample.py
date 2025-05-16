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
true_lambda = 0.5
exp_sample = np.random.exponential(scale=1 / true_lambda, size=n)
np.save("exp_sample.npy", exp_sample)
print("Сохранено: exp_sample.npy")

# --- 3. Двумерные нормальные выборки с разными корреляциями ---
mu_X, mu_Y = 0, 0
sigma_X, sigma_Y = 1, 1
correlations = [-0.9, -0.5, 0.0, 0.5, 0.9]

for rho in correlations:
    cov_matrix = [
        [sigma_X**2, rho * sigma_X * sigma_Y],
        [rho * sigma_X * sigma_Y, sigma_Y**2]
    ]
    bivariate_sample = np.random.multivariate_normal([mu_X, mu_Y], cov_matrix, size=n)
    file_name = f"bivariate_sample_rho_{str(rho).replace('.', '').replace('-', 'm')}.npy"
    np.save(file_name, bivariate_sample)
    print(f"Сохранено: {file_name}")
