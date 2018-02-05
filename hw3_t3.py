import numpy as np
import scipy.stats

A = np.random.randn(100, 100)
col_means = np.apply_along_axis(np.mean, axis=0, arr=A)
col_std   = np.apply_along_axis(np.std , axis=0, arr=A)

ci_u = col_means+col_std*scipy.stats.t.ppf(0.9, 99) / (99.0**0.5)
ci_l = col_means-col_std*scipy.stats.t.ppf(0.9, 99) / (99.0**0.5)

is_in_interval = (0<=ci_u)&(0>=ci_l)

print(is_in_interval)
unique, counts = np.unique(is_in_interval, return_counts=True)
TF_num = dict(zip(unique, counts))
print(TF_num[True])