import json
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_tree
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    roc_auc_score, cohen_kappa_score, matthews_corrcoef

data = pd.read_csv("dataset/output.csv")

owners_to_class = {
    100000000: 0,
    200000000: 0,
    500000000: 0,
    50000000: 0,
    20000000: 0,
    10000000: 0,
    5000000: 0,
    2000000: 0,
    1000000: 0,
    500000: 0,
    200000: 0,
    100000: 0,
    50000: 1,
    20000: 1
}
data['owners'] = data['owners'].map(owners_to_class)

# balance data
class_counts = data['owners'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

df_minority = data[data['owners'] == minority_class]
df_majority = data[data['owners'] == majority_class]

df_majority_sampled = df_majority.sample(len(df_minority), random_state=42)

data_balanced = pd.concat([df_majority_sampled, df_minority])
data_balanced['user_score_class'] = np.where(data_balanced['user_score'] > 80.64, 0, 1)

# target_column = data_balanced["user_score_class"]
target_column = data_balanced["user_score_class"]
data_balanced = data_balanced.drop(columns=["owners"])
data_balanced = data_balanced.drop(columns=["user_score"])
data_balanced = data_balanced.drop(columns=["user_score_class"])
# balance data



print("start training")

X_train, X_test, y_train, y_test = train_test_split(data_balanced, target_column, test_size=0.2, random_state=42)

model = XGBRegressor(objective='multi:softmax', num_class=2)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
y_pred = model.predict(X_test)

min_class = 0
max_class = 1
rounded_and_clipped_predictions = np.clip(np.round(y_pred), min_class, max_class)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print("cliped predtion mse")
print(mean_squared_error(y_test, rounded_and_clipped_predictions))

# Use rounded_and_clipped_predictions for accuracy calculation
accuracy = accuracy_score(y_test, rounded_and_clipped_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

if not os.path.exists('out/process/'):
    os.makedirs('out/process/')

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=rounded_and_clipped_predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values (Clipped)')
plt.title('True Values vs Predicted Values (Clipped)')
plt.savefig('out/process/scatter_plot.png')

residuals = y_test - rounded_and_clipped_predictions
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('out/process/Residual_Plot.png')

plt.figure(figsize=(8, 8))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.savefig('out/process/Distribution_of_Residuals.png')

plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.savefig('out/process/importance.png')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, rounded_and_clipped_predictions)

# Precision, Recall, F1-Score
precision = precision_score(y_test, rounded_and_clipped_predictions, average='weighted')
recall = recall_score(y_test, rounded_and_clipped_predictions, average='weighted')
f1 = f1_score(y_test, rounded_and_clipped_predictions, average='weighted')

# ROC-AUC
roc_auc = roc_auc_score(pd.get_dummies(y_test), y_pred.reshape(-1, 1), multi_class='ovr')

print("ROC-AUC: {:.2f}".format(roc_auc))

# Cohen's Kappa
kappa = cohen_kappa_score(y_test, rounded_and_clipped_predictions)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, rounded_and_clipped_predictions)

# Classification Report
class_report = classification_report(y_test, rounded_and_clipped_predictions)

print("Confusion Matrix:\n", conf_matrix)
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-Score: {:.2f}".format(f1))
print("ROC-AUC: {:.2f}".format(roc_auc))
print("Cohen's Kappa: {:.2f}".format(kappa))
print("MCC: {:.2f}".format(mcc))
print("Classification Report:\n", class_report)



output_dir = 'out/process/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "ROC-AUC": roc_auc,
    "Cohen's Kappa": kappa,
    "MCC": mcc
}

plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title('Métricas de Desempenho do Modelo')
plt.ylabel('Valor')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'metrics_bar_chart.png'))

class_report_dict = classification_report(y_test, rounded_and_clipped_predictions, output_dict=True)
df_class_report = pd.DataFrame(class_report_dict).transpose()

plt.figure(figsize=(12, 6))
sns.heatmap(df_class_report.iloc[:-1, :-1].T, annot=True, cmap='Blues')
plt.title('Relatório de Classificação')
plt.savefig(os.path.join(output_dir, 'classification_report.png'))




# booster = model.get_booster()
# print(len(booster.get_dump()))
#
# plt.figure(figsize=(10, 10), dpi=1200)
# plt.tight_layout()
# plot_tree(model, num_trees=1, rankdir='LR')
# plt.savefig(f'out/process/tree.png', dpi=1200, bbox_inches='tight')





