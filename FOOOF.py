import numpy as np
from fooof import FOOOF
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt 

def detect_plateau_onset(freq, psd, f_start, f_range=10, thresh=0.05,
                         step=1, reverse=False,
                         ff_kwargs=dict(verbose=False, max_n_peaks=1)):
    """
    Detect the plateau of a power spectrum with 1/f exponent beta < threshold.
    """
    exp = np.inf
    fm = FOOOF(**ff_kwargs)
    max_freq = 50
    min_freq = 0

    while exp > thresh:
        if reverse:
            f_start -= step
            if f_start - f_range < min_freq:
                return 45 
            freq_range = [f_start - f_range, f_start]
        else:
            f_start += step
            if f_start + f_range > max_freq:
                return 45 
            freq_range = [f_start, f_start + f_range]

        try:
            fm.fit(freq, psd, freq_range)
            exp = fm.get_params('aperiodic_params', 'exponent')
        except Exception as e:
            print(f"FOOOF fitting failed at freq_range={freq_range}: {e}")
            return 45

    return f_start + f_range // 2

def fooof_tool(frequencies, psd):
    """
    Thực hiện FOOOF fit, trích xuất tham số periodic và aperiodic,
    tính toán lỗi khớp của thành phần không tuần hoàn theo phương pháp
    (phổ gốc - mô hình tuần hoàn) so với mô hình không tuần hoàn,
    và tùy chọn vẽ đồ thị so sánh cho mỗi kênh.

    LƯU Ý: Phương pháp tính lỗi này có hạn chế vì sai số trong mô hình tuần hoàn
    sẽ ảnh hưởng đến kết quả.

    Args:
        frequencies (list or array): Danh sách hoặc mảng các mảng tần số cho mỗi kênh.
        psd (list or array): Danh sách hoặc mảng các mảng phổ công suất (TUYẾN TÍNH)
                             tương ứng cho mỗi kênh.
        plot_each_channel (bool): Nếu True, vẽ đồ thị so sánh cho mỗi kênh.

    Returns:
        tuple: Chứa ba phần tử:
               (np.array(all_features): Mảng 2D các features [peak_params_flat, ap_params] cho mỗi kênh,
                r2s_list: Danh sách R2 score,
                mae_list: Danh sách MAE score)
    """
    mae_list = []
    r2s_list = []
    all_features = []

    num_channels = len(frequencies)
    if num_channels != len(psd):
        raise ValueError("Số lượng mảng tần số và mảng PSD phải bằng nhau.")

    for i in range(num_channels):
        print(f"Processing channel {i+1}/{num_channels}...")
        fm_periodic = FOOOF(peak_width_limits=[1, 12], max_n_peaks=1, min_peak_height=0.01,
                            peak_threshold=-5, aperiodic_mode='fixed', verbose=False)
        fm_aperiodic = FOOOF(peak_width_limits=[1, 12], max_n_peaks=5, min_peak_height=0.01,
                             peak_threshold=-5, aperiodic_mode='fixed', verbose=False)

        freqs_channel = np.array(frequencies[i]).flatten()
        power_linear = np.array(psd[i]).flatten()
         
        try:
            plateau_start = detect_plateau_onset(freqs_channel, power_linear, f_start=20) 

            freq_range_aperiodic = [2, plateau_start]
            freq_range_periodic = [0.01, 15]

            fm_periodic.fit(freqs_channel, power_linear, freq_range_periodic)
            fm_aperiodic.fit(freqs_channel, power_linear, freq_range_aperiodic)

            periodic_params = fm_periodic.get_params('peak_params')     
            aperiodic_params = fm_aperiodic.get_params('aperiodic_params') 

            periodic_params_flat = periodic_params.flatten()

            if np.isnan(periodic_params_flat).any():
                periodic_params_flat = np.array([0, 0, 0])
            features = np.concatenate((periodic_params_flat, aperiodic_params))
            all_features.append(features) 

            power_spectrum_log10_fitted = fm_aperiodic.power_spectrum
            aperiodic_fit = fm_aperiodic._ap_fit
            periodic_fit = fm_aperiodic._peak_fit 
            freqs_fitted = fm_aperiodic.freqs

            if power_spectrum_log10_fitted is not None and periodic_fit is not None and aperiodic_fit is not None and freqs_fitted is not None:
                psd_minus_periodic_model = power_spectrum_log10_fitted - periodic_fit

                mae = mean_absolute_error(psd_minus_periodic_model, aperiodic_fit)
                try:
                    r2s = r2_score(psd_minus_periodic_model, aperiodic_fit)
                except ValueError:
                    print(f"Cảnh báo: Không thể tính R2 score cho kênh {i+1}. Có thể do phương sai bằng 0.")
                    r2s = np.nan

                mae_list.append(mae)
                r2s_list.append(r2s)
                print("success")

            else:
                 print(f"FOOOF fit (_ap_fit/_peak_fit) có thể đã trả về None cho kênh {i+1}. Bỏ qua tính toán lỗi.")
                 mae_list.append(np.nan)
                 r2s_list.append(np.nan)


        except Exception as e:
            print(f"FOOOF fit thất bại cho kênh {i+1}: {e}")

    try:
        final_features = np.array(all_features)
    except ValueError as ve:
        print(f"Lỗi khi tạo mảng features cuối cùng: {ve}")
        final_features = all_features

    return final_features, r2s_list, mae_list