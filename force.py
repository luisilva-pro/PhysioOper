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


def force_onset(signal, height=0.25, distance=3000):

    # Normalization #
    force = signal / np.max(signal)

    # Force onset determination #
    force_H = si.hilbert(force)
    force_H_imag = np.imag(force_H)
    force_H_inv = -1 * np.diff(force_H_imag)
    force_H_dif = force_H_inv / np.max(force_H_inv)
    force_onsets = si.find_peaks(force_H_dif, height=height, distance=distance)

    return force_onsets, force_H_dif


def smooth(y, win_size):
    window = np.ones(win_size) / win_size
    y_smooth = np.convolve(y, window, mode='same')

    return y_smooth


def convertAcc(signal, calib):
    acc_sig = (signal - np.min(calib) / (np.max(calib) - np.min(calib))) * 2 - 1
    return acc_sig



