# Import packages #

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as si
import scipy.stats as ss
import novainstrumentation as ni
import pandas as pd
import seaborn as sns
import emd
import biosignalsnotebooks as bsnb

from force import force_calib, force_onset, convertAcc, \
    smooth, convertToMilivolts, sync, sync_with_delay

sns.set()

# Load file #

filename = 'EMD_test_2021-10-27_09-16-40.txt'
file = np.loadtxt(filename)

# Parameters #

sr = 1000
wind_sm = 101

# Organize data #

calib_name = 'Acc_calib_new.txt'
calib = np.loadtxt(calib_name)

x_calib = calib[:, 2]
y_calib = calib[:, 3]
z_calib = calib[:, 4]

names = {'ForceSq': file[:, 5], 'ForceLg': file[:, 6], 'X_hub': file[:, 2], 'Y_hub': file[:, 3], 'Z_hub': file[:, 4],
         'EMG_tri': file[:, 8], 'X_tri': file[:, 9], 'Y_tri': file[:, 10], 'Z_tri': file[:, 11],
         'EMG_bic': file[:, 13], 'X_bic': file[:, 14], 'Y_bic': file[:, 15], 'Z_bic': file[:, 16]}
data = pd.DataFrame(names)

# Preprocessing #

## Force #

### Conversion ###

force_conv_sq = force_calib(data['ForceSq'])
force_conv_lg = force_calib(data['ForceLg'])

### Filtering ###

# 'Order equal to one is similar to the smooth function with the same window length' #

force_f_sq = si.savgol_filter(force_conv_sq, wind_sm, 1)
force_f_lg = si.savgol_filter(force_conv_lg, wind_sm, 1)

### Normalization ###

force_n_sq = force_f_sq / np.max(force_f_sq)
force_n_lg = force_f_lg / np.max(force_f_lg)

force_sq = force_n_sq
force_lg = force_n_lg

## Accelerometer ##

### Conversion ###

acc_conv_hub = np.zeros((len(file), 3))
acc_conv_tri = np.zeros((len(file), 3))
acc_conv_bic = np.zeros((len(file), 3))

for ii in range(2, 5):
    acc_conv_hub[:, ii-2] = convertAcc(file[:, ii], calib[:, ii])

for ii in range(9, 12):
    acc_conv_tri[:, ii-9] = convertAcc(file[:, ii], calib[:, ii-7])

for ii in range(14, 17):
    acc_conv_bic[:, ii-14] = convertAcc(file[:, ii], calib[:, ii-12])

### Baseline shift ###

for ii in range(0, 3):
    acc_conv_hub[:, ii] = acc_conv_hub[:, ii] - np.max(acc_conv_hub[:, ii])

for ii in range(0, 3):
    acc_conv_tri[:, ii] = acc_conv_tri[:, ii] - np.max(acc_conv_tri[:, ii])

for ii in range(0, 3):
    acc_conv_bic[:, ii] = acc_conv_bic[:, ii] - np.max(acc_conv_bic[:, ii])

### Filtering ###

acc_f_hub = np.zeros_like(acc_conv_hub)
acc_f_tri = np.zeros_like(acc_conv_tri)
acc_f_bic = np.zeros_like(acc_conv_bic)

for ii in range(0, 3):
    acc_f_hub[:, ii] = si.savgol_filter(acc_conv_hub[:, ii], wind_sm, 1)

for ii in range(0, 3):
    acc_f_tri[:, ii] = si.savgol_filter(acc_conv_tri[:, ii], wind_sm, 1)

for ii in range(0, 3):
    acc_f_bic[:, ii] = si.savgol_filter(acc_conv_bic[:, ii], wind_sm, 1)

acc_hub = acc_f_hub
acc_tri = acc_f_tri
acc_bic = acc_f_bic


## EMG ##

### Converstions ###

bicep_conv = convertToMilivolts(data['EMG_bic'])
tricep_conv = convertToMilivolts(data['EMG_tri'])

### Baseline Shift ###

bicep_conv = bicep_conv - np.mean(bicep_conv)
tricep_conv = tricep_conv - np.mean(tricep_conv)

### Filtering ###

bicep_f = ni.bandpass(bicep_conv, 10, 499, order=4, fs=sr, use_filtfilt=True)
tricep_f = ni.bandpass(tricep_conv, 10, 499, order=4, fs=sr, use_filtfilt=True)

biceps = bicep_f
triceps = tricep_f


# Synchronization #

x_hub = acc_hub[:, 0]
y_hub = acc_hub[:, 1]
z_hub = acc_hub[:, 2]
x_tri = acc_tri[:, 0] * (-1)
x_tri, delay_tri = sync(y_hub, x_tri)
y_tri = sync_with_delay(acc_tri[:, 1], delay_tri)
z_tri = sync_with_delay(acc_tri[:, 2], delay_tri)
emg_tri = sync_with_delay(triceps, delay_tri)

y_hub = acc_hub[:, 1]
x_bic = acc_bic[:, 0] * (-1)
x_bic, delay_bic = sync(y_hub, x_bic)
y_bic = sync_with_delay(acc_bic[:, 1], delay_bic)
z_bic = sync_with_delay(acc_bic[:, 2], delay_bic)
emg_bic = sync_with_delay(biceps, delay_bic)

# Cut the first 20 seconds #

# Onset determination #

## Force ##

### Threshold method ###

thr_lg = np.mean(force_lg[:2000]) + 0.04

ind_lg = (force_lg > thr_lg).astype(int)

ind_diff_lg = np.concatenate([[0], ind_lg[1:] - ind_lg[:-1]])

loc_lg = np.where(ind_diff_lg == 1)[0]

### Hilbert method ###

force_sq_h = si.hilbert(force_sq)
force_sq_im = np.diff(np.imag(force_sq_h))
force_sq_im = smooth(force_sq_im, 50)
force_sq_im = (force_sq_im - np.mean(force_sq_im))/ np.std(force_sq_im)
loc_sq = si.find_peaks(force_sq_im, height=2, distance=2250)[0]

dist = np.zeros(len(loc_sq))
onsets_sq = np.zeros(len(loc_sq))
jj = 0

for ii in loc_sq:
    dist[jj] = np.argmin(force_sq_im[ii-250: ii])
    onsets_sq[jj] = ii - 250 + dist[jj]
    jj = jj + 1

onsets_sq = onsets_sq.astype(int)

ax1 = plt.subplot(211)
ax1.scatter(onsets_sq, force_sq_im[onsets_sq], color='red')
ax1.plot(force_sq_im)
ax1.plot(force_sq*10)
ax1.vlines(onsets_sq, -5, 10, colors='red')

ax2 = plt.subplot(212, sharex=ax1)
ax2.scatter(loc_lg, ind_diff_lg[loc_lg], color='red')
ax2.plot(ind_diff_lg)
ax2.plot(ind_lg)
ax2.plot(force_lg)
ax2.vlines(onsets_sq, -0.9, 0.9, colors='red')

## Onset determination ##

ref = force_sq_im[41000:-4000]
z_h = si.hilbert(z_hub[40000:-4000])
z_i = np.imag(z_h)
z_imf = emd.sift.mask_sift(z_i)

emd.plotting.plot_imfs(z_imf, cmap=True)

comp = z_imf[:, 1] + z_imf[:, 2] + z_imf[:, 3] + z_imf[:, 4] + z_imf[:, 5]
comp = comp[1001:] / np.max(comp[1001:])
comp = abs(comp)
plt.plot(comp)
plt.plot(ref)

onsets = si.find_peaks(comp, height=0.15)[0]
dif_onsets = np.diff(onsets)
peaks_dif = np.concatenate(([0], dif_onsets))
first_peak = peaks_dif > 1400
onsets_1 = onsets[first_peak]
onsets_1 = np.concatenate(([onsets[0]], onsets_1))
plt.scatter(onsets_1, comp[onsets_1], color='red')
plt.plot(comp*6)
plt.plot(ref)
plt.vlines(onsets_1, -5, 10)

plt.plot(onsets, 'o-')
plt.plot(dif_onsets, 'o-')



a = si.morlet(len(comp)-15000, w=5.0, s=1.0)
b = si.convolve(comp, a, 'same')
plt.plot(b)
plt.plot(ref*20)

## EMG preprocessing ##

### TKEO ###


def tkeo(signal):
    tkeo = []
    for i in range(0, len(signal)):
        if i == 0 or i == len(signal) - 1:
            tkeo.append(signal[i])
        else:
            tkeo.append(np.power(signal[i], 2) - (signal[i + 1] * signal[i - 1]))

    return tkeo


tkeo_use = 1

if tkeo_use == 1:
    emg_bic_tkeo = tkeo(bicep_f)
    emg_tri_tkeo = tkeo(tricep_f)
else:
    emg_bic_tkeo = bicep_f
    emg_tri_tkeo = tricep_f

emg_bic_tkeo = np.array(emg_bic_tkeo)
emg_tri_tkeo = np.array(emg_tri_tkeo)

emg_bic_s = abs(emg_bic_tkeo)
emg_tri_s = abs(emg_tri_tkeo)

sm = 1

if sm == 1:
    emg_bic_s = smooth(emg_bic_s, wind)
    emg_tri_s = smooth(emg_tri_s, wind)

else:
    emg_bic_s = rms(emg_bic_s, wind)
    emg_tri_s = rms(emg_tri_s, wind)

## Force ##

force_s = smooth(force_conv, wind)

# Synchronization #

sync_delay_bic, x_hub, x_bic = bsnb.synchronise_signals(x_hub_s, (x_mb_bic_s*-1))
sync_delay_tri, x_hub, x_tri = bsnb.synchronise_signals(x_hub_s, (x_mb_tri_s*-1))

# Padding #

v_mb_bic = np.zeros(sync_delay_bic)
v_mb_tri = np.zeros(sync_delay_tri)

x_cut_mb_bic = x_bic[: -sync_delay_bic]
x_cut_mb_tri = x_tri[: -sync_delay_tri]
y_cut_mb_bic = y_mb_bic_s[: -sync_delay_bic]
y_cut_mb_tri = y_mb_tri_s[: -sync_delay_tri]
z_cut_mb_bic = z_mb_bic_s[: -sync_delay_bic]
z_cut_mb_tri = z_mb_tri_s[: -sync_delay_tri]
emg_cut_mb_bic = emg_bic_s[: -sync_delay_bic]
emg_cut_mb_tri = emg_tri_s[: -sync_delay_tri]

x_bic_s = x_cut_mb_bic
x_tri_s = x_cut_mb_tri
y_bic_s = np.concatenate((v_mb_bic, y_cut_mb_bic))
y_tri_s = np.concatenate((v_mb_tri, y_cut_mb_tri))
z_bic_s = np.concatenate((v_mb_bic, z_cut_mb_bic))
z_tri_s = np.concatenate((v_mb_tri, z_cut_mb_tri))
emg_bic_s = np.concatenate((v_mb_bic, emg_cut_mb_bic))
emg_tri_s = np.concatenate((v_mb_tri, emg_cut_mb_tri))

# Normalize #

cut = 15000

x_hub = x_hub[cut:] / np.max(x_hub[cut:])
x_bic = x_bic_s[cut:] / np.max(x_bic_s[cut:])
x_tri = x_tri_s[cut:] / np.max(x_tri_s[cut:])
y_hub = y_hub_s[cut:] / np.max(y_hub_s[cut:])
y_bic = y_bic_s[cut:] / np.max(y_bic_s[cut:])
y_tri = y_tri_s[cut:] / np.max(y_tri_s[cut:])
z_hub = z_hub_s[cut:] / np.max(z_hub_s[cut:])
z_bic = z_bic_s[cut:] / np.max(z_bic_s[cut:])
z_tri = z_tri_s[cut:] / np.max(z_tri_s[cut:])
emg_bic = emg_bic_s[cut:] / np.max(emg_bic_s[cut:])
emg_tri = emg_tri_s[cut:] / np.max(emg_tri_s[cut:])
force = force_s[cut:] / np.max(force_s[cut:])


# Force onset determination #

force_H = si.hilbert(force)
force_H_imag = np.imag(force_H)
force_H_dif = -1 * np.diff(force_H_imag)
force_H_dif = force_H_dif / np.max(force_H_dif)
force_H_dif = smooth(force_H_dif, 20)
force_onsets = si.find_peaks(force_H_dif, height=0.25, distance=3000)
plt.scatter(force_onsets[0], force_H_dif[force_onsets[0]], color='red')
plt.plot(force_H_dif)



