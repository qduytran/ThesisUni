import pandas as pd
import matplotlib.pyplot as plt
from utils import create_label, create_data
import matplotlib.pyplot as plt 

folder_paths = ['D:\\K66K1\\NCKH 2025\\Data\\AD', 'D:\\K66K1\\NCKH 2025\\Data\\CN']
set_files, df_Y = create_label(folder_paths)

df_X, r2s_list, mae_list = create_data(set_files) 
df_X.to_csv('output/X_case2.csv', header=False, index=False)

df_r2s = pd.DataFrame(r2s_list)
df_mae = pd.DataFrame(mae_list)
df_r2s.to_csv('output/R2_case2.csv', header=False, index=False)
df_mae.to_csv('output/MAE_case2.csv', header=False, index=False)
# df_Y.to_csv('Y.csv', header=False, index=False)
flat_list_r2s = [item for sublist in r2s_list for item in sublist] 
flat_list_mae = [item for sublist in mae_list for item in sublist] 
fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(14,6))
ax0.hist(flat_list_r2s)
ax0.set_xlabel('Variance explained (R^2)', fontsize=20)
ax0.set_ylabel('Count', size=20)
ax0.tick_params(labelsize=18)

ax1.hist(flat_list_mae)
ax1.set_xlabel('Mean absolute error (MAE)', fontsize=20)
ax1.set_ylabel('Count', size=20)
ax1.tick_params(labelsize=18)

fig.tight_layout()
# plt.savefig("results_thesis.pdf")
plt.show()