import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef

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
    100000: 1,
    20000: 1,
    50000: 1
}
data['owners'] = data['owners'].apply(lambda x: max(map(int, x.replace(',', '').split('..'))))
data['owners'] = data['owners'].map(owners_to_class)


class_counts = data['owners'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

df_minority = data[data['owners'] == minority_class]
df_majority = data[data['owners'] == majority_class]


df_majority_sampled = df_majority.sample(len(df_minority), random_state=42)

data_balanced = pd.concat([df_majority_sampled, df_minority])


target_column = data_balanced["owners"]
data_balanced = data_balanced.drop(columns=["owners"])
data_balanced = data_balanced.drop(columns=["user_score"])


print("start training")

X_train, X_test, y_train, y_test = train_test_split(data_balanced, target_column, test_size=0.2, random_state=42)

model = XGBRegressor(objective='multi:softmax', num_class=2)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
y_pred = model.predict(X_test)

min_class = min(owners_to_class.values())
max_class = max(owners_to_class.values())
rounded_and_clipped_predictions = np.clip(np.round(y_pred), min_class, max_class)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Use rounded_and_clipped_predictions for accuracy calculation
accuracy = accuracy_score(y_test, rounded_and_clipped_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=rounded_and_clipped_predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values (Clipped)')
plt.title('True Values vs Predicted Values (Clipped)')
plt.savefig('out/scatter_plot.png')

residuals = y_test - rounded_and_clipped_predictions
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('out/Residual_Plot.png')

plt.figure(figsize=(8, 8))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.savefig('out/Distribution_of_Residuals.png')

plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.savefig('out/importance.png')




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