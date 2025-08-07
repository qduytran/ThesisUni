import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

df_X = pd.read_csv("output/X_case2.csv", header=None)
df_Y = pd.read_csv(r"D:\K66K1\2425II_ELT4068\Code\Data\Y.csv", header=None)

# Feature selection
# columns_to_drop = [i for i in range(95) if i % 5 == 0 or i % 5 == 1 or i % 5 == 2]
# df_X = df_X.drop(df_X.columns[columns_to_drop], axis=1)

df_Y.columns = ["target"]
df = pd.concat([df_X, df_Y], axis=1)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

N_SPLITS = 5     
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_oof = []
all_true = []
accuracy_list = []
recall_list = []
specificity_list = []
f1_list = []
auc_list = []

all_probs = []
for i, (train_index, valid_index) in enumerate(skf.split(df, df["target"])):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
    print("#" * 25)

    X_train, y_train = df.iloc[train_index, :-1], df.iloc[train_index, -1]
    X_valid, y_valid = df.iloc[valid_index, :-1], df.iloc[valid_index, -1]

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_valid = scaler.transform(X_valid)

     ### HUẤN LUYỆN MÔ HÌNH
    # model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42, class_weight='balanced')
    # model = KNeighborsClassifier(n_neighbors=3) 
    # model = GaussianNB()
    # model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # model = SVC(random_state=42, class_weight='balanced', probability=True)
    # model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    ### KẾT THÚC HUẤN LUYỆN MÔ HÌNH 

    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1] 
    all_probs.append(y_prob)

    acc = accuracy_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    f1 = f1_score(y_valid, y_pred)
    fold_auc = roc_auc_score(y_valid, y_prob)

    accuracy_list.append(acc)
    recall_list.append(recall)
    specificity_list.append(specificity)
    f1_list.append(f1)
    auc_list.append(fold_auc)
    
    all_oof.append(y_pred)
    all_true.append(y_valid.values)

    del X_train, y_train, X_valid, y_valid, y_pred, y_prob
    gc.collect()

all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)
all_probs = np.concatenate(all_probs)
print("\n### Tổng kết kết quả trên các fold ###")
print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
print(f"Recall (Sensitivity): {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
print(f"Specificity: {np.mean(specificity_list):.4f} ± {np.std(specificity_list):.4f}")
print(f"F1-score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

final_cm = confusion_matrix(all_true, all_oof)
print("\nConfusion Matrix (Tổng hợp toàn bộ K-Fold):")
print(final_cm)

output_df = pd.DataFrame({
    "TrueLabel": all_true,
    "PredictedProb": all_probs
})
filename = "output/svc_case1.csv"
output_df.to_csv(filename, index=False)
print(f"ROC data đã được lưu vào: {filename}")

print("\n### Vẽ biểu đồ Boxplot các metrics ###")
print(recall_list, specificity_list, accuracy_list, auc_list)
data_to_plot = [recall_list, specificity_list, accuracy_list, auc_list]
labels = ['SEN', 'SPE', 'ACC', 'AUC']
plt.figure(figsize=(6, 5))
median_props = dict(linestyle='-', linewidth=1.5, color='red')
box_props = dict(facecolor='#ADD8E6', edgecolor='#0072BD', linewidth=1.5) 
whisker_props = dict(color='#0072BD', linewidth=1.5)
cap_props = dict(color='#0072BD', linewidth=1.5)
flier_props = dict(marker='+', markerfacecolor='red', markeredgecolor='red', markersize=8)
bp = plt.boxplot(data_to_plot,
                 labels=labels,
                 patch_artist=True,
                 medianprops=median_props,
                 flierprops=flier_props,
                 whiskerprops=whisker_props,
                 capprops=cap_props)
colors = ['#AACCFF'] * len(data_to_plot) 
edge_colors = ['#0072BD'] * len(data_to_plot) 
for patch, edge_color in zip(bp['boxes'], edge_colors):
    patch.set_facecolor('#AACCFF') 
    patch.set_edgecolor(edge_color) 
    patch.set_linewidth(1.5)

plt.ylim([-0.05, 1.05]) 
plt.grid(False)
plt.savefig("output/boxplot_metrics_svc_case1.pdf", bbox_inches ='tight')
plt.show()