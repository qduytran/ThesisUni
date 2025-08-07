import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mae_case1 = pd.read_csv('MAE_case1.csv', header=None).values.flatten()
mae_case2 = pd.read_csv('MAE_case2.csv', header=None).values.flatten()
r2_case1 = pd.read_csv('R2_case1.csv', header=None).values.flatten()
r2_case2 = pd.read_csv('R2_case2.csv', header=None).values.flatten()

df_mae1 = pd.DataFrame({'Giá trị MAE': mae_case1, 'Trường hợp': 'Trường hợp 1'})
df_mae2 = pd.DataFrame({'Giá trị MAE': mae_case2, 'Trường hợp': 'Trường hợp 2'})
df_mae_combined = pd.concat([df_mae1, df_mae2])

df_r2_1 = pd.DataFrame({'Giá trị R²': r2_case1, 'Trường hợp': 'Trường hợp 1'})
df_r2_2 = pd.DataFrame({'Giá trị R²': r2_case2, 'Trường hợp': 'Trường hợp 2'})
df_r2_combined = pd.concat([df_r2_1, df_r2_2])

bins_mae = np.linspace(0, 0.25, 44)
bins_r2 = np.linspace(0, 1, 44)

sns.set_theme(style="whitegrid")

g_mae_hist = sns.FacetGrid(df_mae_combined, hue="Trường hợp", height=6, aspect=1.5, palette=['tab:blue', 'tab:orange'])
g_mae_hist.map(sns.histplot, "Giá trị MAE", bins=bins_mae, kde=True, alpha=0) 
g_mae_hist.add_legend(title="Trường hợp")
g_mae_hist.set_axis_labels("Giá trị MAE", "Số lượng")
g_mae_hist.fig.suptitle("Phân bố MAE giữa 2 trường hợp (sử dụng FaceGrid & histplot)", y=1.03)
plt.tight_layout()
plt.savefig("mae_comparison_facegrid_hist.pdf")
plt.savefig("mae_comparison_facegrid_hist.png") 
plt.close(g_mae_hist.fig) 

g_mae_kde = sns.FacetGrid(df_mae_combined, hue="Trường hợp", height=6, aspect=1.5, palette=['tab:blue', 'tab:orange'])
g_mae_kde.map(sns.kdeplot, "Giá trị MAE", fill=True, alpha=0, warn_singular=False) 
g_mae_kde.add_legend(title="Trường hợp")
g_mae_kde.set_axis_labels("Giá trị MAE", "Mật độ")
g_mae_kde.fig.suptitle("Phân bố MAE giữa 2 trường hợp (sử dụng FaceGrid & kdeplot)", y=1.03)
plt.tight_layout()
plt.savefig("mae_comparison_facegrid_kde.pdf")
plt.savefig("mae_comparison_facegrid_kde.png")
plt.close(g_mae_kde.fig)

g_r2_hist = sns.FacetGrid(df_r2_combined, hue="Trường hợp", height=6, aspect=1.5, palette=['tab:blue', 'tab:orange'])
g_r2_hist.map(sns.histplot, "Giá trị R²", bins=bins_r2, kde=True, alpha=0)
g_r2_hist.add_legend(title="Trường hợp")
g_r2_hist.set_axis_labels("Giá trị R²", "Số lượng")
g_r2_hist.fig.suptitle("Phân bố R² giữa 2 trường hợp (sử dụng FaceGrid & histplot)", y=1.03)
plt.tight_layout()
plt.savefig("r2_comparison_facegrid_hist.pdf")
plt.savefig("r2_comparison_facegrid_hist.png")
plt.close(g_r2_hist.fig)

g_r2_kde = sns.FacetGrid(df_r2_combined, hue="Trường hợp", height=6, aspect=1.5, palette=['tab:blue', 'tab:orange'])
g_r2_kde.map(sns.kdeplot, "Giá trị R²", fill=True, alpha=0, warn_singular=False)
g_r2_kde.add_legend(title="Trường hợp")
g_r2_kde.set_axis_labels("Giá trị R²", "Mật độ")
g_r2_kde.fig.suptitle("Phân bố R² giữa 2 trường hợp (sử dụng FaceGrid & kdeplot)", y=1.03)
plt.tight_layout()
plt.savefig("r2_comparison_facegrid_kde.pdf")
plt.savefig("r2_comparison_facegrid_kde.png")
plt.close(g_r2_kde.fig)

print("Đã tạo các biểu đồ phân bố sử dụng FaceGrid và lưu dưới dạng PDF/PNG.")