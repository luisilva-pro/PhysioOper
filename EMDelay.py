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

from force import force_calib, force_onset

sns.set()

# Load file #

filename = 'Test_Force_2021-10-26_10-44-47.txt'
file = np.loadtxt(filename)

# Parameters #

sr = 1000
wind_sm = 101

# EMD delay part #

calib_name = 'Acc_calib_new.txt'
calib = np.loadtxt(calib_name)

x_calib = calib[:, 2]
y_calib = calib[:, 3]
z_calib = calib[:, 4]

names = {'Force': file[:, 5], 'X_hub': file[:, 2], 'Y_hub': file[:, 3], 'Z_hub': file[:, 4],
         'EMG_Bic': file[:, 7], 'X_mb_bic': file[:, 8], 'Y_mb_bic': file[:, 9], 'Z_mb_bic': file[:, 10],
         'EMG_Tri': file[:, 12], 'X_mb_tri': file[:, 13], 'Y_mb_tri': file[:, 14], 'Z_mb_tri': file[:, 15]}
data = pd.DataFrame(names)

# Force #

channel_force = 6
force = file[:17500, channel_force]

## Conversion ##

force_conv = force_calib(force)

## Filtering ##

# Order equal to one is similar to the smooth function with the same window length #

force_f = si.savgol_filter(force_conv, wind_sm, 1)

## Normalization ##

force_n = force_f / np.max(force_f)

## Force onset ##

thr = np.mean(force_n[:2000]) + .05
ind = (force_n > thr).astype(int)

# Determine onset #

ind_diff = np.concatenate([[0], ind[1:] - ind[:-1]])
diff_loc = np.where(ind_diff == 1)[0]

plt.scatter(diff_loc, ind_diff[diff_loc], color='red')
plt.plot(ind_diff)
plt.plot(ind)
plt.plot(force_n)

# Accelerometer #

## Conversion ##

acc_n = 4
acc = file[:17500, acc_n]

acc_conv = convertAcc(acc, z_calib)

## Baseline shift ##

acc_conv = acc_conv - np.mean(acc_conv)

## Filtering ##

acc_f = si.savgol_filter(acc_conv, wind_sm, 1)

## Onset determination ##

acc_h = si.hilbert(acc_f)
acc_i = np.imag(acc_h)
imf = emd.sift.mask_sift(acc_i)

emd.plotting.plot_imfs(imf, cmap=True)

comp = imf[:, 1] + imf[:, 2]

onsets = si.find_peaks(comp, height=)[0]

x_hub = convertAcc(data['X_hub'], x_calib)
y_hub = convertAcc(data['Y_hub'], y_calib)
z_hub = convertAcc(data['Z_hub'], z_calib)
x_mb_bic = convertAcc(data['X_mb_bic'], x_calib)
x_mb_tri = convertAcc(data['X_mb_tri'], x_calib)
y_mb_bic = convertAcc(data['Y_mb_bic'], y_calib)
y_mb_tri = convertAcc(data['Y_mb_tri'], y_calib)
z_mb_bic = convertAcc(data['Z_mb_bic'], z_calib)
z_mb_tri = convertAcc(data['Z_mb_tri'], z_calib)

x_hub = x_hub - np.mean(x_hub)
y_hub = z_hub - np.mean(y_hub)
z_hub = z_hub - np.mean(z_hub)
x_mb_bic = x_mb_bic - np.mean(x_mb_bic)
x_mb_tri = x_mb_tri - np.mean(x_mb_tri)
y_mb_bic = y_mb_bic - np.mean(y_mb_bic)
y_mb_tri = y_mb_tri - np.mean(y_mb_tri)
z_mb_bic = z_mb_bic - np.mean(z_mb_bic)
z_mb_tri = z_mb_tri - np.mean(z_mb_tri)


## EMG ##

def convertToMilivolts(x):
    return ((((x / 2 ** 16) - (1 / 2)) * 3) / 1000) * 1e3


bicep = convertToMilivolts(data['EMG_Bic'])
tricep = convertToMilivolts(data['EMG_Tri'])

bicep = bicep - np.mean(bicep)
tricep = tricep - np.mean(tricep)

bicep_f = ni.bandpass(bicep, 10, 499, order=4, fs=sr, use_filtfilt=True)
tricep_f = ni.bandpass(tricep, 10, 499, order=4, fs=sr, use_filtfilt=True)


# Force conversion #


def force_calib(force_signal):
    # Step 1: Calculate voltage output of the sensor #
    volt_signal = (3 * force_signal) / (2 ** 16)

    # Step 2: Calculate sensor conductance ###
    cond_signal = volt_signal / ((6 - volt_signal) * 47)

    # Step 3: Acquire calibration signal ###
    result = ss.linregress(volt_signal, cond_signal)

    # Step 4: Convert data #
    converted_force = cond_signal / abs(result.slope)

    return converted_force


force_conv = force_calib(data['Force'])


## ACC smooth ##

def smooth(y, win_size):
    window = np.ones(win_size) / win_size
    y_smooth = np.convolve(y, window, mode='same')

    return y_smooth


def rms(signal, win_size):
    signal2 = np.power(signal, 2)
    window = np.ones(win_size) / float(win_size)
    rms_signal = np.sqrt(np.convolve(signal2, window, 'valid'))

    return rms_signal


wind = 101

x_hub_s = smooth(x_hub, wind)
x_mb_bic_s = smooth(x_mb_bic, wind)
x_mb_tri_s = smooth(x_mb_tri, wind)
y_hub_s = smooth(y_hub, wind)
y_mb_bic_s = smooth(y_mb_bic, wind)
y_mb_tri_s = smooth(y_mb_tri, wind)
z_hub_s = smooth(z_hub, wind)
z_mb_bic_s = smooth(z_mb_bic, wind)
z_mb_tri_s = smooth(z_mb_tri, wind)


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



