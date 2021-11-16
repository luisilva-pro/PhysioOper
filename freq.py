# Import packages #

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as si
from scipy.integrate import cumtrapz
import biosignalsnotebooks as bsnb

# Import file #

filename1 = 'EDA_hand_test1_2021-10-29_15-02-58.txt'
filename2 = 'EDA_palmas_forearm_2021-10-29_15-56-17.txt'
filename3 = 'EDA_palmas_arm_2021-10-29_15-57-37.txt'
data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)
data3 = np.loadtxt(filename3)

ax1 = plt.subplot(311)
ax1.plot(data1[:, 5])
ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(data2[:, 5])
ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(data3[:, 5])

# EMG Activations #

signal = data[:50000, 10]
signal = signal - np.mean(signal)
plt.plot(signal)
sr = 1000
activation_begin, activation_end = bsnb.detect_emg_activations(signal, sample_rate=sr,
                                                               threshold_level=2, smooth_level=10)[:2]
plt.vlines(activation_begin, -28000, 28000, colors='red', linestyle='dashed', linewidth=2)
plt.vlines(activation_end, -28000, 28000, colors='red', linestyle='dashed', linewidth=2)

# Iteration along muscular activations

median_freq_data = []
median_freq_time = []

for activation in range(0, len(activation_begin)):
    processing_window = signal[activation_begin[activation]:activation_end[activation]]
    central_point = (activation_begin[activation] + activation_end[activation]) / 2
    median_freq_time += [central_point / sr]

    # Processing window power spectrum (PSD) generation
    freqs, power = si.periodogram(processing_window, fs=sr)

    # Median power frequency determination
    area_freq = cumtrapz(power, freqs, initial=0)
    total_power = area_freq[-1]
    median_freq_data += [freqs[np.where(area_freq >= total_power / 2)[0][0]]]

plt.plot(median_freq_time, median_freq_data)

plt.plot(freqs, power)

