import os
import scipy.io
import pandas as pd
import numpy as np
from FOOOF import fooof_tool
from welch import welch_method

def create_label(folder_paths):
    data = []
    set_files = read_data(folder_paths)
    for file_path in set_files:
        if 'D:\\K66K1\\NCKH 2025\\Data\\AD' in file_path:
            label = 1
        elif 'D:\\K66K1\\NCKH 2025\\Data\\CN' in file_path:
            label = 0
        else:
            label = 2
        data.append([file_path, label])
    df_Y = pd.DataFrame(data)
    df_Y = df_Y.iloc[:, 1:]
    return set_files, df_Y

def read_data(folder_paths):
    set_files = []
    for folder_path in folder_paths:
        files = []
        for f in os.listdir(folder_path):
            if f.endswith('.set'):
                files.append(os.path.join(folder_path, f))
        set_files.extend(files)
    return set_files

def create_data(set_files):
    frequencies = []
    psd = []
    new_rows = []
    all_r2s = []
    all_mae = []
    for set_file in set_files:
        frequencies, psd = welch_method(set_file)
        features, r2s_list, mae_list = fooof_tool(frequencies, psd)
        features_flat = np.array(features).flatten()
        new_rows.append(features_flat)
        all_r2s.append(r2s_list)
        all_mae.append(mae_list)
    print(all_r2s)
    df_X = pd.DataFrame(new_rows)
    return df_X, all_r2s, all_mae