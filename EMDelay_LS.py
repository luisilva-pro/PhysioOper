# Import packages #

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as si
import scipy.stats as ss
import novainstrumentation as ni
import pandas as pd
import seaborn as sns
import emd

sns.set()

# Load file #

mb_name = 'EMD_S02_2021-10-20_13-00-15.txt'
mb = np.loadtxt(mb_name)

# Parameters #

sr = 1000

# EMD delay part #

calib_name = 'Acc_calib_new.txt'
calib = np.loadtxt(calib_name)

z_calib = calib[:, 4]
names = {'Z_hub_mb': mb[:, 4], 'Force': mb[:, 5], 'Z_hub': mb[:, 8], 'EMG_Bic': mb[:, 10], 'Z_mb_bic': mb[:, 13],
         'EMG_Tri': mb[:, 15], 'Z_mb_tri': mb[:, 18]}
data = pd.DataFrame(names)


# Conversions #

## Acc ##


def convertAcc(signal, calib):
    acc_sig = (signal - np.min(calib) / (np.max(calib) - np.min(calib))) * 2 - 1
    return acc_sig


z_hub = convertAcc(data['Z_hub'], z_calib)
z_hub_mb = convertAcc(data['Z_hub_mb'], z_calib)
z_mb_bic = convertAcc(data['Z_mb_bic'], z_calib)
z_mb_tri = convertAcc(data['Z_mb_tri'], z_calib)

z_hub = z_hub - np.mean(z_hub)
z_hub_mb = z_hub_mb - np.mean(z_hub_mb)
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

z_hub_s = smooth(z_hub, wind)
z_hub_mb_s = smooth(z_hub_mb, wind)
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

# After reference decision #

ref_hub_mb = 4760
ref_mb_bic = 4239
ref_mb_tri = 4358
d_mb_bic = ref_hub_mb - ref_mb_bic
d_mb_tri = ref_hub_mb - ref_mb_tri

# Padding #

v_mb_bic = np.zeros(d_mb_bic)
v_mb_tri = np.zeros(d_mb_tri)

z_cut_mb_bic = z_mb_bic_s[: -d_mb_bic]
z_cut_mb_tri = z_mb_tri_s[: -d_mb_tri]
emg_cut_mb_bic = emg_bic_s[: -d_mb_bic]
emg_cut_mb_tri = emg_tri_s[: -d_mb_tri]

z_bic_s = np.concatenate((v_mb_bic, z_cut_mb_bic))
z_tri_s = np.concatenate((v_mb_tri, z_cut_mb_tri))
emg_bic_s = np.concatenate((v_mb_bic, emg_cut_mb_bic))
emg_tri_s = np.concatenate((v_mb_tri, emg_cut_mb_tri))

# Normalize #

cut = 15000

z_hub_mb = z_hub_mb_s[cut:] / np.max(z_hub_mb_s[cut:])
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

# EMG Onset Determination #

emg_bic_n = emg_bic / np.max(emg_bic)
emg_tri_n = emg_tri / np.max(emg_tri)

thr = .01
ind_bic = (emg_bic_n > thr).astype(int)
ind_tri = (emg_tri_n > thr).astype(int)

ind_diff_bic = np.concatenate([[0], ind_bic[1:] - ind_bic[:-1]])
ind_diff_tri = np.concatenate([[0], ind_tri[1:] - ind_tri[:-1]])

triceps_onsets = si.find_peaks(ind_diff_tri, distance=4000)
plt.scatter(triceps_onsets[0], ind_diff_tri[triceps_onsets[0]], color='red')
plt.plot(ind_diff_tri)

# bic_onsets = np.where(ind_diff_bic == 1)[0]
# tri_onsets = np.where(ind_diff_tri == 1)[0]

# ACC onset #

acc = np.array([z_hub, z_bic, z_tri])

acc_H = np.zeros_like(acc, dtype='complex_')
acc_H_imag = np.zeros_like(acc)
acc_H_dif = np.zeros((3, acc.shape[1]-1))

plt.plot(force)

for ii in range(0, 3):
    acc_H[ii] = si.hilbert(acc[ii])
    acc_H_imag[ii] = np.imag(acc_H[ii])
    plt.plot(acc_H_imag[ii])

plt.legend(['Force', 'Wrist', 'Biceps', 'Triceps'])

imf_acc0 = emd.sift.mask_sift(acc_H_imag[0])
imf_acc1 = emd.sift.mask_sift(acc_H_imag[1])
imf_acc2 = emd.sift.mask_sift(acc_H_imag[2])

# emd.plotting.plot_imfs(imf_acc0, cmap=True)

comp = np.array([imf_acc0[:, 1] + imf_acc0[:, 2],
                 imf_acc1[:, 1] + imf_acc1[:, 2],
                 imf_acc2[:, 1], imf_acc2[:, 2]])

plt.figure(figsize=[10, 8])
title_fig = np.array(['Wrist', 'Biceps', 'Triceps'])

for ii in range(0, 3):
    plt.subplot(3, 1, ii+1)
    plt.title(title_fig[ii])
    plt.plot(force_H_dif)
    plt.plot(ind_diff_tri)
    plt.plot(acc[ii])
    plt.plot(acc_H_imag[ii])
    plt.plot(comp[ii] * 2)
    plt.xticks([])


def acc_onset1(ts, height, amp_visual, dist, distance=None):
    z_t_onsets = si.find_peaks(ts, height=height, distance=distance)[0]
    peaks_dif = np.diff(z_t_onsets)
    peaks_dif = np.concatenate(([0], peaks_dif))
    first_peak = peaks_dif > dist
    z_t_onsets2 = z_t_onsets[first_peak]
    z_t_onsets2 = np.concatenate(([z_t_onsets[0]], z_t_onsets2))
    plt.scatter(z_t_onsets2, ts[z_t_onsets2], color='red')
    plt.plot(ts)
    plt.plot(force_H_dif * amp_visual)
    plt.plot(ind_diff_tri * amp_visual)

    return peaks_dif, z_t_onsets, z_t_onsets2


def acc_onset2(ts, height, amp_visual, dist, distance=None):
    onsets = si.find_peaks(ts, height=height)[0]
    peaks_dif = np.diff(onsets)
    peaks_dif = np.concatenate(([0], peaks_dif))
    first_peak = peaks_dif > dist
    onsets_1 = onsets[first_peak]
    onsets_1 = np.concatenate(([onsets[0]], onsets_1))
    index = np.where(first_peak)[0]
    onsets_2 = onsets[index + 1]
    onsets_2 = np.concatenate(([onsets[1]], onsets_2))
    plt.scatter(onsets_1, ts[onsets_1], color='red')
    plt.scatter(onsets_2, ts[onsets_2], color='blue')
    plt.plot(ts)
    plt.plot(force_H_dif * amp_visual)
    plt.plot(ind_diff_tri * amp_visual)
    onsets_mean = (onsets_1 + onsets_2) / 2

    return peaks_dif, onsets, onsets_mean

pad = np.zeros(1000)

# Parameters for Acc wrist #

thr20 = 0.3
amp_visual0 = 5
dist0 = 1000
ts0 = abs(comp[0] ** 2 * 100)
ts0 = np.concatenate((pad, ts0[1000:-1000], pad))

peaks_dif0, z_t_onsets0, z_t_onsets20 = acc_onset1(ts=ts0, height=thr20, amp_visual=amp_visual0, dist=dist0)

plt.plot(z_t_onsets0)
plt.plot(peaks_dif0)
plt.plot(z_t_onsets0, 'o')
plt.plot(peaks_dif0, 'o')

# Parameters for Acc Biceps #

thr21 = 0.3
amp_visual1 = 5
dist1 = 1000
ts1 = abs(comp[1] ** 2 * 100)
ts1 = np.concatenate((pad, ts1[1000:-1000], pad))

peaks_dif1, z_t_onsets1, z_t_onsets21 = acc_onset(ts=ts1, height=thr21, amp_visual=amp_visual1, dist=dist1)

# Parameters for Acc Triceps #

thr22 = .4
amp_visual2 = 5
dist2 = 1000
ts2 = abs(comp[2] ** 2 * 100)
ts2 = np.concatenate((pad, ts2[1000:-1000], pad))
ts2 = smooth(ts2, 10)

peaks_dif2, onsets2, onsets2_mean = acc_onset2(ts=ts2, height=thr22, amp_visual=amp_visual2, dist=dist2)



onsets = si.find_peaks(ts2, height=thr22)[0]
peaks_dif = np.diff(onsets)
peaks_dif = np.concatenate(([0], peaks_dif))
first_peak = peaks_dif > dist2
onsets_1 = onsets[first_peak]
onsets_1 = np.concatenate(([onsets[0]], onsets_1))
index = np.where(first_peak)[0]
onsets_2 = onsets[index + 1]
onsets_2 = np.concatenate(([onsets[1]], onsets_2))
plt.scatter(onsets_1, ts2[onsets_1], color='red')
plt.scatter(onsets_2, ts2[onsets_2], color='blue')
plt.plot(ts2)
plt.plot(force_H_dif * amp_visual2)
plt.plot(ind_diff_tri * amp_visual2)
z_t_onsets2 = (onsets_1 +onsets_2) / 2

plt.plot(onsets)
plt.plot(peaks_dif)
plt.plot(onsets, 'o')
plt.plot(peaks_dif, 'o')