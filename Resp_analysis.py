# Import packages ----------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as si
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import neurokit2 as nk
from dtw import dtw, accelerated_dtw
from force import smooth, tam
import emd

sns.set()

# Load data ----------------------------------------------- #

filename1 = 'S03_TASK1_2021-11-11_12-19-12.txt'
filename2 = 'S03_TASK2_2021-11-11_12-35-07.txt'
filename3 = 'S03_TASK3_2021-11-11_12-55-04.txt'
filename_rest_pre = 'S03_REST_PRE_2021-11-11_11-15-07.txt'
filename_rest_pos = 'S03_REST_POS_2021-11-11_13-06-56.txt'

task1 = np.loadtxt(filename1)
task2 = np.loadtxt(filename2)
task3 = np.loadtxt(filename3)
rest_pre = np.loadtxt(filename_rest_pre)
rest_pos = np.loadtxt(filename_rest_pos)

task1 = task1[len(task1)-570000-20000: -20000, :]
task2 = task2[len(task2)-570000-20000: -20000, :]
task3 = task3[len(task3)-570000-20000: -20000, :]
rest_pre = rest_pre[-120000:, :]
rest_pos = rest_pos[-120000:, :]

# Define respiratory data #

resp_task_raw = {'Chest1': task1[:, 7], 'Chest2': task2[:, 7], 'Chest3': task3[:, 7],
            'Abd1': task1[:, 8], 'Abd2': task2[:, 8], 'Abd3': task3[:, 8]}

resp_rest_raw = {'Chest_pre': rest_pre[:, 7], 'Abd_pre': rest_pre[:, 8],
             'Chest_pos': rest_pos[:, 7], 'Abd_pos': rest_pos[:, 8]}

resp_task = pd.DataFrame(resp_task_raw)
resp_rest = pd.DataFrame(resp_rest_raw)
resp_t = resp_task.copy()
resp_r = resp_rest.copy()

n = 40
ds_task = len(resp_task[resp_task.columns[0]])/n
ds_task = int(ds_task)
ds_rest = len(resp_rest[resp_rest.columns[0]])/n
ds_rest = int(ds_rest)

resp_task_s = resp_task.copy()[0: ds_task]
resp_task_d = resp_task.copy()
resp_rest_s = resp_rest.copy()[0: ds_rest]
resp_rest_d = resp_rest.copy()

# Power Spectrum ----------------------------------------- #


def power(data, fs):
    ps = np.abs(np.fft.fft(data)) ** 2
    time = 1 / fs
    freqs = np.fft.fftfreq(data.size, time)
    idx = np.argsort(freqs)
    plt.plot(np.abs(freqs[idx]), ps[idx])

    return freqs, ps, idx

freqs, ps, idx = power(resp_rest['Chest_pos'], 1000)
print(np.median(freqs))

# Clean data --------------------------------------------- #

for ii in range(0, resp_task.shape[1]):
    resp_task[resp_task.columns[ii]] = nk.signal_filter(resp_task[resp_task.columns[ii]], lowcut=0.15, highcut=.45,
                                                  method='fir', order=5)

for ii in range(0, resp_rest.shape[1]):
    resp_rest[resp_rest.columns[ii]] = nk.signal_filter(resp_rest[resp_rest.columns[ii]], lowcut=0.15, highcut=.45,
                                                  method='fir', order=5)


# Downsample --------------------------------------------- #

for ii in range(0, resp_task.shape[1]):
    resp_task_s[resp_task.columns[ii]] = si.resample(resp_task[resp_task.columns[ii]], ds_task)

for ii in range(0, resp_rest.shape[1]):
    resp_rest_s[resp_rest.columns[ii]] = si.resample(resp_rest[resp_rest.columns[ii]], ds_rest)


# Signal decomposition -------------------------------------#
comp_task = []
comp_rest = []

for ii in range(0, resp_task_s.shape[1]):
    imf_task = emd.sift.ensemble_sift(resp_task_s[resp_task.columns[ii]])
    comp_task.append(imf_task)

for ii in range(0, resp_rest_s.shape[1]):
    imf_rest = emd.sift.ensemble_sift(resp_rest_s[resp_rest.columns[ii]])
    comp_rest.append(imf_rest)

emd.plotting.plot_imfs(comp_task[2][5000:8000], cmap=True)

# Synchronizarion ---------------------------------------- #

# Plot #

ax1 = plt.subplot(411)
ax1.plot(resp_task_s[resp_task.columns[2]][5000:8000])
ax2 = plt.subplot(412)
ax2.plot(comp_task[2][5000:8000, 3])
ax3 = plt.subplot(413)
ax3.plot(resp_task_s[resp_task.columns[3]][5000:8000])
ax4 = plt.subplot(414)
ax4.plot(comp_task[3][5000:8000, 3])


## Pearson ##

### Global ###

global_pearson = resp_s.corr()
pearson1 = np.round(global_pearson.iloc[0, 3], 3)
pearson2 = np.round(global_pearson.iloc[1, 4], 3)
pearson3 = np.round(global_pearson.iloc[2, 5], 3)
print(pearson1, pearson2, pearson3)

### Local ###

wind_time = np.round(np.linspace(0, len(resp_s), 10)).astype(int)

local_pearson = np.zeros((10, 3))
resp1 = resp_s[['Chest1', 'Abd1']]
resp2 = resp_s[['Chest2', 'Abd2']]
resp3 = resp_s[['Chest3', 'Abd3']]

for ii in range(1, 10):
    local_pearson[ii, 0] = resp1[wind_time[ii-1]:wind_time[ii]].corr().iloc[0, 1]
    local_pearson[ii, 1] = resp2[wind_time[ii-1]:wind_time[ii]].corr().iloc[0, 1]
    local_pearson[ii, 2] = resp3[wind_time[ii-1]:wind_time[ii]].corr().iloc[0, 1]

plt.plot(local_pearson[1:], linewidth=2)
plt.xticks(fontweight=8)
plt.ylabel('Pearson Coefficient')
plt.legend(['Baseline', 'Fatigue 1', 'Fatigue 2'])
plt.title('Pearson Local Correlation')

syc1 = nk.signal_synchrony(resp1['Chest1'], resp1['Abd1'], method="correlation", window_size=1420)
syc2 = nk.signal_synchrony(resp2['Chest2'], resp2['Abd2'], method="correlation", window_size=1420)
syc3 = nk.signal_synchrony(resp3['Chest3'], resp3['Abd3'], method="correlation", window_size=1420)

plt.figure()
plt.plot(syc1)
plt.plot(syc2)
plt.plot(syc3)

## Cross-correlation ##

cc1 = sm.tsa.stattools.ccf(resp1['Chest1'], resp1['Abd1'], adjusted=False)
cc2 = sm.tsa.stattools.ccf(resp2['Chest2'], resp2['Abd2'], adjusted=False)
cc3 = sm.tsa.stattools.ccf(resp3['Chest3'], resp3['Abd3'], adjusted=False)

cc_value1 = max(cc1)
cc_value2 = max(cc2)
cc_value3 = max(cc3)

cc_index1 = np.argmax(cc1)
cc_index2 = np.argmax(cc2)
cc_index3 = np.argmax(cc3)

print(cc_value1, cc_value2, cc_value3)
print(cc_index1, cc_index2, cc_index3)

## Instantaneous phase ##

n2 = 400
ip_ch1 = np.angle(si.hilbert(resp_d[resp.columns[0]]), deg=False)
ip_ab1 = np.angle(si.hilbert(resp_d[resp.columns[1]]), deg=False)
phase_synchrony1 = 1-np.sin(np.abs(ip_ch1-ip_ab1)/2)
phase_synchrony1 = smooth(phase_synchrony1, n2)
plt.figure()
plt.plot(phase_synchrony1[50:-50], linewidth=2)

ip_ch2 = np.angle(si.hilbert(resp_d[resp.columns[2]]), deg=False)
ip_ab2 = np.angle(si.hilbert(resp_d[resp.columns[3]]), deg=False)
phase_synchrony2 = 1-np.sin(np.abs(ip_ch2-ip_ab2)/2)
phase_synchrony2 = smooth(phase_synchrony2, n2)
plt.plot(phase_synchrony2[50:-50], linewidth=2)

ip_ch3 = np.angle(si.hilbert(resp_d[resp.columns[4]]), deg=False)
ip_ab3 = np.angle(si.hilbert(resp_d[resp.columns[5]]), deg=False)
phase_synchrony3 = 1-np.sin(np.abs(ip_ch3-ip_ab3)/2)
phase_synchrony3 = smooth(phase_synchrony3, n2)
plt.plot(phase_synchrony3[50:-50], linewidth=2)


## look at ##

plt.figure(figsize=[20, 10])

ax1 = plt.subplot(331)
ax1.plot(resp_s[resp.columns[0]])
ax1.plot(resp_s[resp.columns[1]])
ax2 = plt.subplot(334, sharex=ax1)
ax2.plot(ip_ch1)
ax2.plot(ip_ab1)
ax3 = plt.subplot(337, sharex=ax1)
ax3.plot(phase_synchrony1)
plt.ylim(0, 1)

ax4 = plt.subplot(332, sharex=ax1)
ax4.plot(resp_s[resp.columns[2]])
ax4.plot(resp_s[resp.columns[3]])
ax5 = plt.subplot(335, sharex=ax1)
ax5.plot(ip_ch2)
ax5.plot(ip_ab2)
ax6 = plt.subplot(338, sharex=ax1)
ax6.plot(phase_synchrony2)
plt.ylim(0, 1)

ax7 = plt.subplot(333, sharex=ax1)
ax7.plot(resp_s[resp.columns[2]])
ax7.plot(resp_s[resp.columns[3]])
ax8 = plt.subplot(336, sharex=ax1)
ax8.plot(ip_ch3)
ax8.plot(ip_ab3)
ax9 = plt.subplot(339, sharex=ax1)
ax9.plot(phase_synchrony3)
plt.ylim(0, 1)


## Dynamic Time Warping ##

ch1 = np.array(resp_s[resp_raw.columns[0]])
ab1 = np.array(resp_s[resp_raw.columns[1]])
d1, cost_matrix1, acc_cost_matrix1, path1 = accelerated_dtw(ch1, ab1, dist='euclidean')
plt.figure()
plt.imshow(acc_cost_matrix1.T, origin='lower', cmap='bone', interpolation='nearest')
plt.plot(path1[0], path1[1], 'w', linewidth=2)
plt.xlabel('Chest')
plt.ylabel('Abdominal')
plt.title(f'DTW Minimum Path with minimum distance: {np.round(d1, 2)}')
plt.show()

ch2 = np.array(resp_s[resp_raw.columns[2]])
ab2 = np.array(resp_s[resp_raw.columns[3]])
d2, cost_matrix2, acc_cost_matrix2, path2 = accelerated_dtw(ch2, ab2, dist='euclidean')
plt.figure()
plt.imshow(acc_cost_matrix2.T, origin='lower', cmap='bone', interpolation='nearest')
plt.plot(path2[0], path2[1], 'w', linewidth=2)
plt.xlabel('Chest')
plt.ylabel('Abdominal')
plt.title(f'DTW Minimum Path with minimum distance: {np.round(d2, 2)}')
plt.show()

ch3 = np.array(resp_s[resp_raw.columns[4]])
ab3 = np.array(resp_s[resp_raw.columns[5]])
d3, cost_matrix3, acc_cost_matrix3, path3 = accelerated_dtw(ch3, ab3, dist='euclidean')
plt.figure()
plt.imshow(acc_cost_matrix3.T, origin='lower', cmap='bone', interpolation='nearest')
plt.plot(path3[0], path3[1], 'w', linewidth=2)
plt.xlabel('Chest')
plt.ylabel('Abdominal')
plt.title(f'DTW Minimum Path with minimum distance: {np.round(d3, 2)}')
plt.show()

tam1 = tam(path1)
tam2 = tam(path2)
tam3 = tam(path3)

sns.heatmap(acc_cost_matrix1, annot=True)

