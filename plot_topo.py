import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from mne import create_info

df = pd.read_csv('output/X_case2.csv', header=None)

exp_cols = list(range(3, 95, 5))
exp_data = df.iloc[:, exp_cols].values

exp_ad = exp_data[:36, :]
exp_cn = exp_data[36:, :]

exp_ad_mean = np.mean(exp_ad, axis=0)
exp_cn_mean = np.mean(exp_cn, axis=0)

from scipy.stats import ttest_ind
_, p_values = ttest_ind(exp_ad, exp_cn, axis=0, equal_var=False)
neg_log_p = -np.log10(p_values)

ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'O2']

montage = make_standard_montage('standard_1020')
info = create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
info.set_montage(montage)
pos_dict = info.get_montage().get_positions()['ch_pos']
pos = np.array([pos_dict[name][:2] for name in ch_names])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im_ad, cn_ad = plot_topomap(exp_ad_mean, pos, axes=axes[0], show=False, cmap="Reds")
plt.colorbar(im_ad, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.1)
axes[0].set_title("AD - Slope")

im_cn, cn_cn = plot_topomap(exp_cn_mean, pos, axes=axes[1], show=False, cmap="Reds")
plt.colorbar(im_cn, ax=axes[1], orientation="horizontal", fraction=0.05, pad=0.1)
axes[1].set_title("CN - Slope")

im_logp, cn_logp = plot_topomap(neg_log_p, pos, axes=axes[2], show=False, cmap="RdBu_r")
plt.colorbar(im_logp, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.1)
axes[2].set_title("-log10(p-value)")

plt.tight_layout()
fig.savefig('output/case2_topo.pdf', bbox_inches='tight')
plt.show()
