import mne
import numpy as np
from scipy.signal import welch, get_window
import matplotlib.pyplot as plt
import pandas as pd
import os
from fooof import FOOOF

def fooof_tool(frequencies, psd):
    """
    Thực hiện phân tích FOOOF trên dữ liệu và trả về features, R^2 và MAE cho cả periodic và aperiodic fitting.

    Args:
        frequencies (list of numpy arrays): Danh sách các mảng tần số cho mỗi kênh.
        psd (list of numpy arrays): Danh sách các mảng PSD cho mỗi kênh.

    Returns:
        tuple: (features, r2_periodic, mae_periodic, r2_aperiodic, mae_aperiodic)
    """
    all_features = []
    r2_periodic_values = []
    mae_periodic_values = []
    r2_aperiodic_values = []
    mae_aperiodic_values = []

    for i in range(len(frequencies)):
        fm_periodic = FOOOF(peak_width_limits=[1, 12], max_n_peaks=1, min_peak_height=0.01,
                           peak_threshold=1, aperiodic_mode='fixed', verbose=False)
        fm_aperiodic = FOOOF(peak_width_limits=[1, 12], max_n_peaks=5, min_peak_height=0.01,
                           peak_threshold=1, aperiodic_mode='fixed', verbose=False)
        freq_range_periodic = [0.01, 15]
        freq_range_aperiodic = [2, 45]

        frequencies[i] = frequencies[i].flatten()
        psd[i] = psd[i].flatten()

        try:
            fm_periodic.fit(frequencies[i], psd[i], freq_range_periodic)
            fm_aperiodic.fit(frequencies[i], psd[i], freq_range_aperiodic)

            periodic_params = fm_periodic.get_params('peak_params')
            aperiodic_params = fm_aperiodic.get_params('aperiodic_params')

            periodic_params = periodic_params.flatten()

            if np.isnan(periodic_params).any():
                periodic_params = np.array([0, 0, 0])

            if aperiodic_params is not None:
                offset = aperiodic_params[0]
                exponent = aperiodic_params[1]
                aperiodic_features = np.array([offset, exponent]) 
            else:
                aperiodic_features = np.array([0, 0]) 

            features = np.concatenate((periodic_params, aperiodic_features))  
            all_features.extend(features)

            r2_periodic_values.append(fm_periodic.r_squared_)
            mae_periodic_values.append(fm_periodic.error_)
            r2_aperiodic_values.append(fm_aperiodic.r_squared_)
            mae_aperiodic_values.append(fm_aperiodic.error_)

        except Exception as e:
            print(f"FOOOF fit failed for channel {i + 1}: {e}")
            features = np.array([0, 0, 0, 0, 0])
            all_features.extend(features)

            r2_periodic_values.append(0)
            mae_periodic_values.append(0)
            r2_aperiodic_values.append(0)
            mae_aperiodic_values.append(0)

    return (np.array(all_features), np.array(r2_periodic_values),
            np.array(mae_periodic_values), np.array(r2_aperiodic_values),
            np.array(mae_aperiodic_values))

def welch_method(file_path):
    """
    Tính toán PSD và tần số bằng phương pháp Welch.

    Args:
        file_path (str): Đường dẫn đến file dữ liệu EEG.

    Returns:
        tuple: frequencies (list), psd (list)
    """
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    channel_name = raw.info['ch_names']
    frequencies = []
    psd = []
    for i in range(19):
        index = raw.ch_names.index(channel_name[i])
        data = raw[index, :][0] * 1e6
        N = 1000
        nfft = 2 ** int(np.ceil(np.log2(N)))
        freqs, power = welch(data, fs=raw.info['sfreq'], nfft=nfft, window = get_window('hamming',256), noverlap=125, scaling='density')
        frequencies.append(freqs)
        psd.append(power)
    return frequencies, psd

def create_data(set_files):
    """
    Tính toán features, frequencies, và PSD cho mỗi file.

    Args:
        set_files (list): Danh sách các đường dẫn đến file dữ liệu EEG.

    Returns:
        tuple: (df_X, frequencies_list, psd_list, r2_periodic_list, mae_periodic_list,
               r2_aperiodic_list, mae_aperiodic_list)
    """
    frequencies_list = []
    psd_list = []
    new_rows = []
    r2_periodic_list = []
    mae_periodic_list = []
    r2_aperiodic_list = []
    mae_aperiodic_list = []

    for set_file in set_files:
        frequencies, psd = welch_method(set_file)
        frequencies_list.append(frequencies)
        psd_list.append(psd)
        (features, r2_periodic_values, mae_periodic_values,
         r2_aperiodic_values, mae_aperiodic_values) = fooof_tool(frequencies, psd)
        new_rows.append(features)
        r2_periodic_list.append(r2_periodic_values)
        mae_periodic_list.append(mae_periodic_values)
        r2_aperiodic_list.append(r2_aperiodic_values)
        mae_aperiodic_list.append(mae_aperiodic_values)

    df_X = pd.DataFrame(new_rows)
    return (df_X, frequencies_list, psd_list, r2_periodic_list, mae_periodic_list,
            r2_aperiodic_list, mae_aperiodic_list)

def analyze_r2_mae(r2_periodic_list, mae_periodic_list, r2_aperiodic_list, mae_aperiodic_list):
    """
    Tính MAE và R^2 trung bình trên tất cả các kênh và vẽ hai bộ biểu đồ, một cho periodic fit và một cho aperiodic fit.

    Args:
        r2_periodic_list (list): Danh sách các mảng R^2 từ periodic fit.
        mae_periodic_list (list): Danh sách các mảng MAE từ periodic fit.
        r2_aperiodic_list (list): Danh sách các mảng R^2 từ aperiodic fit.
        mae_aperiodic_list (list): Danh sách các mảng MAE từ aperiodic fit.
    """

    all_r2_periodic = []
    all_mae_periodic = []
    all_r2_aperiodic = []
    all_mae_aperiodic = []

    for r2_values, mae_values in zip(r2_periodic_list, mae_periodic_list):
        all_r2_periodic.append(np.mean(r2_values))
        all_mae_periodic.append(np.mean(mae_values))

    for r2_values, mae_values in zip(r2_aperiodic_list, mae_aperiodic_list):
        all_r2_aperiodic.append(np.mean(r2_values))
        all_mae_aperiodic.append(np.mean(mae_values))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(all_r2_periodic, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Mean R^2 (Periodic Fit)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean R^2 (Periodic Fit)")

    plt.subplot(1, 2, 2)
    plt.hist(all_mae_periodic, bins=20, color='lightcoral', edgecolor='black')
    plt.xlabel("Mean MAE (Periodic Fit)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean MAE (Periodic Fit)")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(all_r2_aperiodic, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Mean R^2 (Aperiodic Fit)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean R^2 (Aperiodic Fit)")

    plt.subplot(1, 2, 2)
    plt.hist(all_mae_aperiodic, bins=20, color='lightcoral', edgecolor='black')
    plt.xlabel("Mean MAE (Aperiodic Fit)")
    plt.ylabel("Count")
    plt.title("Distribution of Mean MAE (Aperiodic Fit)")

    plt.tight_layout()
    plt.show()


def read_data(folder_paths):
    """
    Đọc danh sách các file .set từ các folder cho trước.

    Args:
        folder_paths (list): Danh sách các đường dẫn đến các folder chứa file .set.

    Returns:
        list: Danh sách các đường dẫn đầy đủ đến các file .set.
    """
    set_files = []
    for folder_path in folder_paths:
        files = []
        for f in os.listdir(folder_path):
            if f.endswith('.set'):
                files.append(os.path.join(folder_path, f))
        set_files.extend(files)
    return set_files


if __name__ == "__main__": 

    folder_paths = ['data_ad_clean/AD', 'data_ad_clean/CN'] 

    set_files = read_data(folder_paths)

    (df_X, frequencies_list, psd_list, r2_periodic_list, mae_periodic_list,
     r2_aperiodic_list, mae_aperiodic_list) = create_data(set_files)

    analyze_r2_mae(r2_periodic_list, mae_periodic_list, r2_aperiodic_list, mae_aperiodic_list)