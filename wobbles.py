# Import Packages #

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
import biosignalsnotebooks as bsnb

sns.set()

# Import Data #

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

# Prepare Data #

id_value = 200000

for ii in range(0, len(data)):
    if len(data[ii]) > 200000:
        data[ii] = data[ii][len(data[ii]) - 570000 - 20000: -20000, :]
    else:
        data[ii] = data[ii][-120000:, :]

acc_data = []

for ii in range(0, len(data)):
    if info[ii] == 'AD':
        acc = np.array([data[ii][:, 11], data[ii][:, 12], data[ii][:, 13]])
        acc_data.append(acc)
    elif info[ii] == 'DC':
        print(ii)
        acc = np.array([data[ii][:, 2], data[ii][:, 3], data[ii][:, 4]])
        acc_data.append(acc)

# Load calib File #

calib_name = 'Acc_calib_new.txt'
calib = np.loadtxt(calib_name)

calib = np.array([calib[:, 2], calib[:, 3], calib[:, 4]])

# Calibration Procedure #


def convertAcc(signal, calib):
    acc_sig = (signal - np.min(calib) / (np.max(calib) - np.min(calib))) * 2 - 1
    return acc_sig


## Conversion ##

acc_conv = acc_data.copy()

for ii in range(0, len(acc_data)):
    acc1 = acc_conv[ii]
    for jj in range(0, len(acc)):
        acc_conv[ii][jj, :] = convertAcc(acc1[jj, :], calib[jj, :])

## Baseline shift ##

acc_det = acc_conv.copy()

for ii in range(0, len(acc_conv)):
    acc2 = acc_det[ii]
    for jj in range(0, len(acc2)):
        acc_det[ii][jj, :] = acc2[jj, :] - np.mean(acc2[jj, :])

## Filtering data #

wind = 101

acc_fil = acc_det.copy()

for ii in range(0, len(acc_det)):
    acc3 = acc_fil[ii]
    for jj in range(0, len(acc3)):
        acc_fil[ii][jj, :] = si.savgol_filter(acc3[jj, :], wind, 1)

## Wobbles Quantification ##

def power(data, fs, fig=1):
    ps = np.abs(np.fft.fft(data)) ** 2
    time = 1 / fs
    freqs = np.fft.fftfreq(data.size, time)
    idx = np.argsort(freqs)

    if fig == 1:
        plt.plot(np.abs(freqs[idx]), ps[idx])

    return freqs, ps, idx


## Filter data with a bandPass filter between 10 and 20 Hz ##

acc_1020 = acc_fil.copy()
sr = 1000
lowF = 10
highF = 20
order = 4
b, a = si.butter(order, np.array([lowF, highF])/(sr/2.), btype='bandpass')

for ii in range(0, len(acc_fil)):
    acc4 = acc_fil[ii]
    for jj in range(0, len(acc4)):
        acc_1020[ii][jj, :] = bsnb.bandpass(acc4[jj, :], lowF, highF, order=3, use_filtfilt=True)

subj = int(len(info) / 5)
acc_trap1020 = np.zeros((subj*5, 3))

for ii in range(0, len(acc_trap1020)):
    freqs_x, ps_x, idx_x = power(acc_1020[ii][0, :], 1000)
    freqs_y, ps_y, idx_y = power(acc_1020[ii][1, :], 1000)
    freqs_z, ps_z, idx_z = power(acc_1020[ii][2, :], 1000)

    acc_trap1020[ii, 0] = np.trapz(ps_x[idx_x], freqs_x[idx_x]) * 10e-12
    acc_trap1020[ii, 1] = np.trapz(ps_y[idx_y], freqs_y[idx_y]) * 10e-12
    acc_trap1020[ii, 2] = np.trapz(ps_z[idx_z], freqs_z[idx_z]) * 10e-12

mom = [ii for ii in range(len(acc_fil)) if ii % 5 == 2]
plt.figure(figsize=[10, 8])

for ii in range(0, 3):
    c = acc_trap1020[:, ii]

    c1 = []
    c2 = []
    c3 = []

    for n in mom:
        t1 = c[n]
        c1.append(t1)
        t2 = c[n+1]
        c2.append(t2)
        t3 = c[n+2]
        c3.append(t3)

    plt.subplot(3, 1, ii + 1)

    labels = ['Trial 1', 'Trial 2', 'Trial 3']
    plt.boxplot([c1, c2, c3], notch=True, labels=labels, whiskerprops=dict(linestyle='--', linewidth=2.0))
    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=12)
    if ii == 0:
        plt.ylabel('X', fontweight='bold', fontsize=14)
    elif ii == 1:
        plt.ylabel('Y', fontweight='bold', fontsize=14)
    elif ii == 2:
        plt.ylabel('Z', fontweight='bold', fontsize=14)

## Identifying Wobbles by cycle

from matrixprofile import *
signal = acc_1020[3][2, :]
win_size = 9450
mp_dist, mp_idx = matrixProfile.stomp(signal, win_size)
smooth_dist = smooth(mp_dist, 100)


def look_at_acc(x, y, z):
    plt.figure(figsize=[10,6])
    ax1 = plt.subplot(311)
    ax1.plot(x)
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.plot(y)
    ax3 = plt.subplot(313, sharex=ax1)
    ax3.plot(z)

n = 3
look_at_acc(acc_data[n][0, :], acc_data[n][1, :], acc_data[n][2, :])