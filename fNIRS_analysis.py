import mne
import numpy as np
import matplotlib.pyplot as plt

mb_name = 'EMD_2021-10-23_13-14-19.txt'
mb = np.loadtxt(mb_name)

mne.io.read_raw_boxy()
raw_intensity = mne.io.read_raw_nirx(mb[:, 3], verbose=True)
raw_intensity.load_data()
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)