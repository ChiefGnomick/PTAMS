import numpy as np
import pandas as pd

# 1. Одномерная нормальная выборка
X = np.load("norm_sample.npy")
df_norm = pd.DataFrame({'X': X})
df_norm.to_csv("norm_sample.csv", index=False)
print("Сохранено: norm_sample.csv")

# 2. Экспоненциальная выборка
Y = np.load("exp_sample.npy")
df_exp = pd.DataFrame({'Y': Y})
df_exp.to_csv("exp_sample.csv", index=False)
print("Сохранено: exp_sample.csv")

# 3. Двумерные нормальные выборки с разной корреляцией
correlations = [-0.9, -0.5, 0.0, 0.5, 0.9]

for rho in correlations:
    file_name = f"bivariate_sample_rho_{str(rho).replace('.', '').replace('-', 'm')}.npy"
    data = np.load(file_name)
    df_biv = pd.DataFrame({'X': data[:, 0], 'Y': data[:, 1]})
    csv_name = file_name.replace(".npy", ".csv")
    df_biv.to_csv(csv_name, index=False)
    print(f"Сохранено: {csv_name}")
