import matplotlib.pyplot as plt
import numpy as np 
import os 
from welch import welch_method
from fooof import FOOOF
from sklearn.metrics import mean_absolute_error, r2_score

folder_paths = ['D:\\K66K1\\NCKH 2025\\Data\\AD', 'D:\\K66K1\\NCKH 2025\\Data\\CN']
set_files = []
for folder_path in folder_paths:
    set_files.extend([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) if f.endswith('.set')
    ])

all_freqs = []
all_psds = []
for set_file in set_files:
    freqs, psd = welch_method(set_file)
    all_freqs.append(freqs)
    all_psds.append(psd)

f_upper_list = np.arange(15, 50, 1)  
r2s_by_f_upper = []
mae_by_f_upper = []
r2s_std_by_f_upper = []
mae_std_by_f_upper = []

for f_upper in f_upper_list:
    r2_all_files = []
    mae_all_files = []

    for freqs, psd in zip(all_freqs, all_psds):
        r2_temp = []
        mae_temp = []
        for i in range(19): 
            fm = FOOOF(peak_width_limits=[1, 12], max_n_peaks=5, min_peak_height=0.01,
                       peak_threshold=1, aperiodic_mode='fixed', verbose=False)
            try:
                fm.fit(freqs[i], psd[i], [2, f_upper])

                power_spectrum_log10_fitted = fm.power_spectrum
                aperiodic_fit = fm._ap_fit
                periodic_fit = fm._peak_fit 
                freqs_fitted = fm.freqs

                psd_minus_periodic_model = power_spectrum_log10_fitted - periodic_fit
                mae = mean_absolute_error(psd_minus_periodic_model, aperiodic_fit)
                r2s = r2_score(psd_minus_periodic_model, aperiodic_fit)
                
                mae_temp.append(mae)
                r2_temp.append(r2s)
            except:
                r2_temp.append(np.nan)
                mae_temp.append(np.nan)

        r2_mean = np.nanmean(r2_temp)
        mae_mean = np.nanmean(mae_temp)
        r2_all_files.append(r2_mean)
        mae_all_files.append(mae_mean)

    r2s_by_f_upper.append(np.nanmean(r2_all_files))
    mae_by_f_upper.append(np.nanmean(mae_all_files))
    r2s_std_by_f_upper.append(np.nanstd(r2_all_files))
    mae_std_by_f_upper.append(np.nanstd(mae_all_files))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(f_upper_list, r2s_by_f_upper, marker='o', label='Mean R²')
plt.fill_between(f_upper_list,
                 np.array(r2s_by_f_upper) - np.array(r2s_std_by_f_upper),
                 np.array(r2s_by_f_upper) + np.array(r2s_std_by_f_upper),
                 color='blue', alpha=0.2, label='±1 STD')
plt.xlabel('Upper bound of freq_range_aperiodic (Hz)')
plt.ylabel('Mean R² across all patients')
plt.title('R² vs Upper Frequency Bound')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f_upper_list, mae_by_f_upper, marker='o', color='orange', label='Mean MAE')
plt.fill_between(f_upper_list,
                 np.array(mae_by_f_upper) - np.array(mae_std_by_f_upper),
                 np.array(mae_by_f_upper) + np.array(mae_std_by_f_upper),
                 color='orange', alpha=0.2, label='±1 STD')
plt.xlabel('Upper bound of freq_range_aperiodic (Hz)')
plt.ylabel('Mean MAE across all patients')
plt.title('MAE vs Upper Frequency Bound')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("output/MAE_R2_comparision.pdf")
plt.show()

