import mne
import numpy as np
import pandas as pd
import os
from scipy.signal import welch, get_window

def welch_method(file_path):
    """
    Tính toán PSD cho từng kênh từ một file .set và trả về tần số và PSD.

    Args:
        file_path (str): Đường dẫn đến file .set.

    Returns:
        tuple: Một tuple chứa hai list:
            - frequencies (list of numpy.ndarray): Danh sách các mảng tần số, mỗi mảng tương ứng với một kênh.
            - psd (list of numpy.ndarray): Danh sách các mảng PSD, mỗi mảng tương ứng với một kênh.
    """
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    channel_name = raw.info['ch_names']
    sfreq = raw.info['sfreq']

    frequencies = []
    psd = []

    for i in range(19):
        index = raw.ch_names.index(channel_name[i])
        data = raw[index, :][0].flatten() * 1e6 
        
        N = 1000
        nfft = 2 ** int(np.ceil(np.log2(N)))
        freqs, power = welch(data, fs=sfreq, nfft=nfft, window=get_window('hamming', 256), noverlap=125, scaling='density')

        frequencies.append(freqs)
        psd.append(power)

    return frequencies, psd

def process_set_files(folder_path, output_freq_csv, output_psd_csv):
    """
    Xử lý tất cả các file .set trong thư mục, tính toán PSD và lưu kết quả vào file CSV.

    Args:
        folder_path (str): Đường dẫn đến thư mục chứa các file .set.
        output_freq_csv (str): Đường dẫn để lưu file CSV chứa tần số.
        output_psd_csv (str): Đường dẫn để lưu file CSV chứa PSD.
    """

    all_psd_data = []
    first_file = True 

    for filename in os.listdir(folder_path):
        if filename.endswith(".set"):
            file_path = os.path.join(folder_path, filename)
            print(f"Đang xử lý file: {filename}")

            frequencies, psd = welch_method(file_path)

            if first_file:
                freq_indices = np.where(frequencies[0] <= 50)[0]
                freqs_to_save = frequencies[0][freq_indices] 

                df_freq = pd.DataFrame(freqs_to_save)
                df_freq.to_csv(output_freq_csv, index=False, header=False)
                first_file = False

            for channel_idx in range(len(psd)):
                psd_values = psd[channel_idx]
                psd_values_to_save = psd_values[freq_indices]  
                all_psd_data.append(psd_values_to_save) 

    df_psd = pd.DataFrame(all_psd_data)
    df_psd.to_csv(output_psd_csv, index=False, header=False) 

folder_path = "data_ad_clean/CN" 
output_freq_csv = "frequencies.csv"
output_psd_csv = "psd_data.csv"

process_set_files(folder_path, output_freq_csv, output_psd_csv)
print("Hoàn thành xử lý và lưu kết quả.")