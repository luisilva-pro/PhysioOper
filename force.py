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


def convertToMilivolts(x):
    return ((((x / 2 ** 16) - (1 / 2)) * 3) / 1000) * 1e3


def rms(signal, win_size):
    signal2 = np.power(signal, 2)
    window = np.ones(win_size) / float(win_size)
    rms_signal = np.sqrt(np.convolve(signal2, window, 'valid'))

    return rms_signal


def sync(signal1, signal2):

    # Signal1 will be the reference, thus, the output is the synchronized signal2 #
    delay = bsnb.synchronise_signals(signal1, signal2)[0]

    if delay > 0:
        signal2_sync = signal2[delay:]
        signal2_sync = np.concatenate((signal2_sync, np.zeros(delay)))

    else:
        signal2_sync = np.concatenate((np.zeros(delay, signal2)))
        signal2_sync = signal2_sync[:-delay]

    return signal2_sync, delay


def sync_with_delay(signal, delay):

    # Synchronize signal having delay #

    if delay > 0:
        signal_sync = signal[delay:]
        signal_sync = np.concatenate((signal_sync, np.zeros(delay)))

    else:
        signal_sync = np.concatenate((np.zeros(delay, signal)))
        signal_sync = signal_sync[:-delay]

    return signal_sync


def tam(path, report='full'):
    """
    Calculates the Time Alignment Measurement (TAM) based on an optimal warping path
    between two time series.
    Reference: Folgado et. al, Time Alignment Measurement for Time Series, 2018.
    :param path: (ndarray)
                A nested array containing the optimal warping path between the
                two sequences.
    :param report: (string)
                A string containing the report mode parameter.
    :return:    In case ``report=instants`` the number of indexes in advance, delay and phase
                will be returned. For ``report=ratios``, the ratio of advance, delay and phase
                will be returned. In case ``report=distance``, only the TAM will be returned.
    """

    # Delay and advance counting
    delay = len(np.where(np.diff(path[0]) == 0)[0])
    advance = len(np.where(np.diff(path[1]) == 0)[0])

    # Phase counting
    incumbent = np.where((np.diff(path[0]) == 1) * (np.diff(path[1]) == 1))[0]
    phase = len(incumbent)

    # Estimated and reference time series duration.
    len_estimation = path[1][-1]
    len_ref = path[0][-1]

    p_advance = advance * 1. / len_ref
    p_delay = delay * 1. / len_estimation
    p_phase = phase * 1. / np.min([len_ref, len_estimation])

    if report == 'instants':
        return np.array([advance, delay, phase])

    if report == 'ratios':
        return np.array([advance, delay, phase])

    if report == 'distance':
        return p_advance + p_delay + (1 - p_phase)

    if report == 'full':
        return np.array([advance, delay, phase, p_advance + p_delay + (1 - p_phase)])