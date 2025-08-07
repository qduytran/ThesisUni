import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import os

def process_eeg_data_from_folder(folder_path, n_channels=19):
    """
    Processes EEG data from all .set files in a folder to calculate mean PSD and variance for each channel.

    Args:
        folder_path (str): Path to the folder containing EEG data in EEGLAB format.

    Returns:
        tuple: Frequencies, list of mean PSDs (one per channel), list of standard deviations (one per channel), list of channel names.
    """
    all_subject_psds_per_channel = [[] for _ in range(n_channels)]
    channel_names = []

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".set")]
    
    for file_path in file_paths:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        
        if not channel_names:
            channel_names = raw.info['ch_names'][:n_channels]
        
        sfreq = raw.info['sfreq']
        
        for i in range(n_channels):
            index = raw.ch_names.index(channel_names[i])
            data = raw[index, :][0].flatten() * 1e6  # Convert to uV
            
            N = 1000
            nfft = 2 ** int(np.ceil(np.log2(N)))
            freqs, power = welch(data, fs=sfreq, nfft=nfft, window=get_window('hamming', 256), noverlap=125, scaling='density')
            
            all_subject_psds_per_channel[i].append(power)
    
    mean_psds = [np.mean(psds, axis=0) for psds in all_subject_psds_per_channel]
    std_psds = [np.std(psds, axis=0) for psds in all_subject_psds_per_channel]
    
    return freqs, mean_psds, std_psds, channel_names

def plot_psd_with_variance(frequencies, mean_psd, std_psd, channel_name, save_path):
    """
    Plots the mean PSD with a shaded area representing standard deviation and saves the image.

    Args:
        frequencies (array): Frequency values.
        mean_psd (array): Mean PSD values.
        std_psd (array): Standard deviation of PSD values.
        channel_name (str): Name of the EEG channel.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(4, 3))
    plt.plot(frequencies, mean_psd, color="#E91E63", linewidth=2, label="Mean PSD")
    plt.fill_between(frequencies, mean_psd - std_psd, mean_psd + std_psd, color="#E91E63", alpha=0.3, label="±1 Std Dev")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"PSD - Channel {channel_name}")
    plt.xlim(0, 50)
    plt.grid(True, color="#D3D3D3", linewidth=0.5)
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

folder_path = "D:\\K66K1\\NCKH 2025\\Data\\CN"
output_folder = "D:\\K66K1\\NCKH 2025\\Results"
os.makedirs(output_folder, exist_ok=True)

frequencies, mean_psds, std_psds, channel_names = process_eeg_data_from_folder(folder_path)

for i, channel_name in enumerate(channel_names):
    save_path = os.path.join(output_folder, f"cn_channel_{i+1}_{channel_name}.pdf")
    plot_psd_with_variance(frequencies, mean_psds[i], std_psds[i], channel_name, save_path)
