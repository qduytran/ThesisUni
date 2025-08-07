import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mae_case1 = pd.read_csv('output/MAE_case1 .csv', header=None).values.flatten()
mae_case2 = pd.read_csv('output/MAE_case2.csv', header=None).values.flatten()
r2_case1 = pd.read_csv('output/R2_case1.csv', header=None).values.flatten()
r2_case2 = pd.read_csv('output/R2_case2.csv', header=None).values.flatten()

def plot_grouped_histogram(data1, data2, bins, label1, label2, title, xlabel, output_filename):
    counts1, _ = np.histogram(data1, bins=bins)
    counts2, _ = np.histogram(data2, bins=bins)

    bar_width = 0.4
    x = np.arange(len(bins)-1)

    plt.figure(figsize=(10,6))
    plt.bar(x - bar_width/2, counts1, width=bar_width, label=label1, align='center')
    plt.bar(x + bar_width/2, counts2, width=bar_width, label=label2, align='center')

    plt.xticks(x, [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)], rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel("Số lượng")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(output_filename, format='pdf')
    plt.show()
    plt.close()

bins_mae = np.linspace(0, 0.25, 44)  
bins_r2 = np.linspace(0, 1, 44)  

plot_grouped_histogram(mae_case1, mae_case2, bins_mae, 'Trường hợp 1', 'Trường hợp 2',
                       title="So sánh MAE giữa 2 trường hợp",
                       xlabel="Giá trị MAE",
                       output_filename="output/mae_comparison.pdf")

plot_grouped_histogram(r2_case1, r2_case2, bins_r2, 'Trường hợp 1', 'Trường hợp 2',
                       title="So sánh R² giữa 2 trường hợp",
                       xlabel="Giá trị R²",
                       output_filename="output/r2_comparison.pdf")
