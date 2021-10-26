
# Import packages #

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as si
import scipy.stats as ss
import novainstrumentation as ni
import pandas as pd

# Import files #

force_name = 'S00_Force1_2021-10-13_11-39-04.txt'
force_raw = np.loadtxt(force_name)

mvc_name = 'S00_MVC_2021-10-13_11-52-25.txt'
mvc_raw = np.loadtxt(mvc_name)

trial_name = 'S00_Trial1_2021-10-13_12-01-46.txt'
trial_raw = np.loadtxt(trial_name)

fp_name = 'S00_FatigueProtocol_2021-10-13_12-22-12.txt'
fp_raw = np.loadtxt(fp_name)

mb_name = 'Acc_MB_2021-10-14_15-46-44.txt'
mb = np.loadtxt(mb_name)

# "sensor": ["XYZ", "XYZ", "XYZ", "FSRIII", "ECG", "RIP", "CUSTOM/SpO2"] #
# Parameters #

sr = 1000

# EMD delay part #

calib_name = 'Calib_Acc.txt'
calib = np.loadtxt(calib_name)

z_calib = calib[:, 4]
z_emd = force_raw[:, 4]
force_emd = force_raw[:, 5]
emg_emd = force_raw[:, 15]

delay_dic = {'Z': z_emd, 'Force': force_emd, 'EMG': emg_emd}
emd_frame = pd.DataFrame(delay_dic)


# General functions #


def smooth(y, win_size):

    window = np.ones(win_size) / win_size
    y_smooth = np.convolve(y, window, mode='same')

    return y_smooth


# TKEO #


def tkeo(signal):
    tkeo = []
    for i in range(0, len(signal)):
        if i == 0 or i == len(signal) - 1:
            tkeo.append(signal[i])
        else:
            tkeo.append(np.power(signal[i], 2) - (signal[i + 1] * signal[i - 1]))

    return tkeo


## Force units conversion ##

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


force_conv = force_calib(emd_frame['Force'])

force_smooth = smooth(force_conv, 101)

## Acc calibration ##

z = emd_frame['Z'] - np.mean(emd_frame['Z'])
z_calib = z_calib - np.mean(z_calib)


def convertAcc(signal, calib):
    acc_sig = ((signal - np.min(calib) / (np.max(calib) - np.min(calib)))) * 2 - 1

    return acc_sig


acc_sig_emd = convertAcc(z, z_calib)
acc_conv = si.savgol_filter(acc_sig_emd, 101, 4)
acc_conv = acc_conv - np.mean(acc_conv)
acc_tkeo = tkeo(acc_conv)

acc_pos = np.zeros_like(acc_conv)

for ii in range(1, len(acc_conv)+1):
    if acc_pos[ii-1] < 100:
        acc_pos[ii-1] = 0

acc_tkeo = tkeo(acc_sig2)

## EMG preprocessing ##

emg_emd = emd_frame['EMG'] - np.mean(emd_frame['EMG'])


def convertToMilivolts(x):
    return ((((x / 2**16) - (1/2)) * 3) / 1000) * 1e3


emg_conv = convertToMilivolts(emg_emd)

## EMG bandpass, TKEO, rectification and smoothing ##

emg_conv_f = ni.bandpass(emg_conv, 10, 499, order=4, fs=sr, use_filtfilt=True)

emg_tkeo = tkeo(emg_conv_f)
emg_tkeo = np.array(emg_tkeo)

## Smooth EMG ##

emg_smooth = smooth(emg_tkeo, 101)

##

# Look at force data #

ax1 = plt.subplot(311)
ax1.plot(force_conv)
ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(acc_conv)
ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(emg_conv_f)

# Fatigue Protocol #
## ["XYZ", "XYZ", "XYZ", "ECG", "RIP", "OXI"] ##

fp_frame = pd.DataFrame(fp_raw)


def spO2_convert(signal):

    spO2_converted = (signal * 1) / (1.2 * 2 ** 16)

    return spO2_converted

spO2_conv = spO2_convert(spO2)


