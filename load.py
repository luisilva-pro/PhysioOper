# Thoracic and Abdominal Breathing during Repetitive Tasks
# Import Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as si
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import neurokit2 as nk
from dtw import dtw, accelerated_dtw
import emd
import glob
import os

sns.set()

# Import data

file_list = glob.glob(os.path.join(os.getcwd(), '/Users/luissilva/PycharmProjects/PhysioOper/data', "*.txt"))
file_list = np.sort(file_list)
print(file_list)

data = []

for file_path in file_list:
    load = np.loadtxt(file_path)
    data.append(load)

info = []

for file_path in file_list:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        info.append(lines[1][19:21])

# Prepare data

id_value = 200000

for ii in range(0, len(data)):
    if len(data[ii]) > 200000:
        data[ii] = data[ii][len(data[ii]) - 570000 - 20000: -20000, :]
    else:
        data[ii] = data[ii][-120000:, :]

resp_data = []

for ii in range(0, len(data)):
    if info[ii] == 'AD':
        resp = np.array([data[ii][:, 7], data[ii][:, 8]])
        resp_data.append(resp)
    elif info[ii] == 'DC':
        resp = np.array([data[ii][:, 16], data[ii][:, 17]])
        resp_data.append(resp)

# Filter data

filtered_resp = resp_data.copy()
col = resp_data[1].shape[0]

for ii in range(0, len(filtered_resp)):
    for jj in range(0, col):
        filtered_resp[ii][jj, :] = nk.signal_filter(resp_data[ii][jj, :], lowcut=0.15, highcut=.45,
                                                  method='fir', order=5)

# Downsample data

ds_resp = []

for ii in range(0, len(filtered_resp)):
    ds = filtered_resp[ii][:, ::25]
    ds_resp.append(ds)

ds_resp[1].shape

# Signal decomposition

comp_resp = []

for ii in range(0, len(ds_resp)):
    for jj in range(0, col):
        imf_resp = emd.sift.ensemble_sift(ds_resp[ii][jj, :], max_imfs=5)
        comp_resp.append(imf_resp)

emd.plotting.plot_imfs(comp_resp[2][5000:8000, :], cmap=True)

emd.plotting.plot_imfs(comp_resp[12][5000:8000, :], cmap=True)

plt.plot(comp_resp[12][5000:8000, 3] + comp_resp[12][5000:8000, 2])

emd.plotting.plot_imfs(comp_resp[9][1000:4000, :], cmap=True)

plt.plot(comp_resp[9][1000:4000,2] + comp_resp[9][1000:4000,3])

# Normalization

# Signal itself

z_ds_resp = ds_resp.copy()

for ii in range(0, len(ds_resp)):
    for jj in range(0, col):
        z_ds_resp[ii][jj, :] = (ds_resp[ii][jj, :] - np.mean(ds_resp[ii][jj, :])) / np.std(ds_resp[ii][jj, :])

# Reconstructed signal of the decomposition

z_comp_resp = comp_resp.copy()

for ii in range(0, len(comp_resp)):
    for jj in range(0, 5):
        z_comp_resp[ii][jj, :] = (comp_resp[ii][jj, :] - np.mean(comp_resp[ii][jj, :])) / np.std(comp_resp[ii][jj, :])

# Synchronization
# The synchronization analysis comprises four different methods: 1) Pearson coefficient in segmented and overlapping window; 2) Cross correlation; 3) Dynamic time warping; 4) Instantaneous phase.
# Pearson coefficient in segmented and overlapping window

# Cross-correlation

# Signal itself

xcor = np.zeros((len(z_ds_resp), 1))
xlag = np.zeros((len(z_ds_resp), 1))

for n in range(0, len(ds_resp)):
    corr = si.correlate(z_ds_resp[n][0, :], z_ds_resp[n][1, :], mode='full', method='auto')
    lags = si.correlation_lags(len(z_ds_resp[n][0, :]), len(z_ds_resp[n][1, :]))
    corr /= np.max(corr_arr)
    pos = np.argmax(corr)
    xcor[n] = np.max(corr)
    xlag[n] = len(corr) / 2 - pos

# Reconstructed signal of the decomposition

xcor_d = np.zeros((len(z_comp_resp), 1))
xlag_d = np.zeros((len(z_comp_resp), 1))

tasks = [ii for ii in range(len(z_comp_resp)) if ii % 2 == 0]

for n in tasks:
    corr = si.correlate((comp_resp[n][:, 2] + comp_resp[n][:, 3]),
                        (comp_resp[n + 1][:, 2] + comp_resp[n + 1][:, 3]), mode='full', method='auto')
    lags = si.correlation_lags(len(comp_resp[n][:, 2] + comp_resp[n][:, 3]),
                               len(comp_resp[n + 1][:, 2] + comp_resp[n + 1][:, 3]))
    corr /= np.max(corr_arr)
    pos = np.argmax(corr)
    xcor_d[n] = np.max(corr)
    xlag_d[n] = len(corr) / 2 - pos

xlag_d = xlag_d[xlag_d != 0]

xlag = abs(xlag)
xlag_d = abs(xlag_d)

task1 = np.mean([xlag[0], xlag[5]])
task2 = np.mean([xlag[1], xlag[6]])
task3 = np.mean([xlag[2], xlag[7]])

plt.bar(['Trial1', 'Trial2', 'Trial3'], [task1, task2, task3], color=['green','yellow', 'red'])
plt.title('Delay among Trials')
plt.ylabel('Samples')

# Instantaneous phase synchrony
## In the following two methods the signal that will be used is converted by Hilbert-Huang transform¶



plt.subplot(331)
plt.subplot(332)
plt.subplot(335)


