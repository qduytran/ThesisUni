import mne
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch, get_window

def welch_method(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    channel_name = raw.info['ch_names']
    channel_index = []
    channel_data = []
    channel_time = []
    frequencies = []
    psd = []
    for i in range(19):
        index = raw.ch_names.index(channel_name[i])
        data = raw[index, :][0] * 1e6
        time = raw[index, :][1] 

        channel_index.append(index)
        channel_data.append(data)
        channel_time.append(time)
        N = 1024  
        nfft = 2 ** int(np.ceil(np.log2(N)))
        freqs, power = welch(channel_data[i], fs=raw.info['sfreq'], nfft=nfft, window = get_window('hamming',256), noverlap=125, scaling='density')
        
        power = power.flatten()
        idx = (freqs >= 0) & (freqs <= 50)
        freqs = freqs[idx]
        power = power[idx]

        frequencies.append(freqs)
        psd.append(power)
    return frequencies, psd

# set_file = 'D:\\K66K1\\NCKH 2025\\Data\\AD\\sub-001_task-eyesclosed_eeg.set'
# frequencies, psd = welch_method(set_file)
# frequencies[17] = frequencies[17].flatten()
# psd[17] = psd[17].flatten()
# plt.plot(frequencies[17], 10 * np.log10(psd[17]))
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power Spectral Density (dB)")
# plt.title("Power Spectral Density using Welch's Method")
# plt.grid()
# plt.show()