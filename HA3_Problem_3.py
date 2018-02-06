import numpy as np
import scipy.stats

t_stat = scipy.stats.t.ppf(0.95, 99)
#print('t_stat (90%, m = 100):', t_stat)
a = np.random.randn(100, 100)
mean_col = a.mean(axis=0)
std_col = a.std(axis=0)
mean_str = a.mean(axis=1)
std_str = a.std(axis=1)
zt_col = mean_col + t_stat * std_col / 10
zb_col = mean_col - t_stat * std_col / 10
zt_str = mean_str + t_stat * std_str / 10
zb_str = mean_str - t_stat * std_str / 10
tof_col = [(zb_col[i] < 0 < zt_col[i]) for i in range(len(zb_col))]
print(tof_col)
print(tof_col.count(1))
tof_str = [(zb_str[i] < 0 < zt_str[i]) for i in range(len(zb_str))]
print(tof_str)
print(tof_str.count(1))
