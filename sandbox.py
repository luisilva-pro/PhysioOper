import neurokit2 as nk
import matplotlib.pyplot as plt
# from dtw import dtw, accelerated_dtw
import numpy as np
from dtw_duarte import *

rsp_u = nk.rsp_simulate(duration=60, sampling_rate=250, respiratory_rate=15)[::10]
rsp_l = nk.rsp_simulate(duration=60, sampling_rate=250, respiratory_rate=15)[::10]

rsp_u = (rsp_u - np.mean(rsp_u)) / np.std(rsp_u)
rsp_l = (rsp_l - np.mean(rsp_l)) / np.std(rsp_l)

plt.figure()
d1, cost_matrix1, acc_cost_matrix1, path1 = dtw(rsp_u, rsp_l, winlen=11)

plot_alignment(rsp_u, rsp_l, path1)

# plt.figure()
# plt.imshow(acc_cost_matrix1.T, origin='lower', cmap='bone', interpolation='nearest')
# plt.plot(path1[0], path1[1], 'w', linewidth=2)
# plt.xlabel('Chest')
# plt.ylabel('Abdominal')
# plt.title(f'DTW Minimum Path with minimum distance: {np.round(d1, 2)}')

plt.figure()
plt.plot(rsp_u)
plt.plot(rsp_l)

plt.show()

print(tam(path1, report='distance'))


