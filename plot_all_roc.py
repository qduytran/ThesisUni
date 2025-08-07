import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

file_info = [
    {"path": "output/rf_case1.csv", "label": "RF Case 1", "model_type": "RF", "case_type": "case1"},
    {"path": "output/rf_case2.csv", "label": "RF Case 2", "model_type": "RF", "case_type": "case2"},
    {"path": "output/svc_case1.csv", "label": "SVC Case 1", "model_type": "SVC", "case_type": "case1"},
    {"path": "output/svc_case2.csv", "label": "SVC Case 2", "model_type": "SVC", "case_type": "case2"},
]

model_colors = {
    "RF": "darkorange", 
    "SVC": "green"       
}

plt.rcParams['axes.linewidth'] = 1.2

plt.figure(figsize=(8, 7))

for info in file_info:
    try:
        df = pd.read_csv(info["path"])
        y_true = df["TrueLabel"]
        y_prob = df["PredictedProb"]

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        color = model_colors.get(info["model_type"], "black") 
        line_style = '-'  
        if info["case_type"] == "case2":
            line_style = '--'  

        plt.plot(fpr, tpr, color=color, lw=2.5,
                 label=f'{info["label"]}',
                 linestyle=line_style)
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{info['path']}'. Bỏ qua file này.")
    except KeyError:
        print(f"Lỗi: File '{info['path']}' không chứa cột 'TrueLabel' hoặc 'PredictedProb'. Bỏ qua file này.")
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý file '{info['path']}': {e}. Bỏ qua file này.")

plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle=':', label='Random (AUC = 0.500)')

plt.xlim([-0.03, 1.03])
plt.ylim([-0.03, 1.03])
plt.xticks(np.arange(0, 1.01, 0.2), fontsize=12)
plt.yticks(np.arange(0, 1.01, 0.2), fontsize=12)

plt.xlabel('False Positive Rate', fontsize=14, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=14, labelpad=10)
plt.title('Comparison of ROC Curves by Model and Case', fontsize=16, pad=15)

legend = plt.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='lightgray')

plt.grid(True, linestyle=':', linewidth=0.7, color='lightgray', alpha=0.7)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('output/roc_all.pdf', bbox_inches='tight')
plt.show()