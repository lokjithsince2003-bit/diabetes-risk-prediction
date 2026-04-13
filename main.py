import pandas as pd

# load dataset
df = pd.read_csv("data/diabetes.csv")

print("Minimum values BEFORE cleaning:")
print(df.min())

# columns where 0 is invalid
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# replace 0 with median
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

print("\nMinimum values AFTER cleaning:")
print(df.min())

from sklearn.model_selection import train_test_split

# separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# create model
model = LogisticRegression(max_iter=1000)

# train model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:", accuracy)

from sklearn.tree import DecisionTreeClassifier

# create decision tree model
dt_model = DecisionTreeClassifier()

# train model
dt_model.fit(X_train, y_train)

# predict
y_pred_dt = dt_model.predict(X_test)

# accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", dt_accuracy)

from sklearn.ensemble import RandomForestClassifier

# create random forest model
rf_model = RandomForestClassifier()

# train model
rf_model.fit(X_train, y_train)

# predict
y_pred_rf = rf_model.predict(X_test)

# accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_accuracy)

import joblib

# save best model (Logistic Regression)
joblib.dump(model, "model/diabetes_model.pkl")

print("Model saved successfully!")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.show()

from sklearn.metrics import roc_curve, auc

# get probability scores from Logistic Regression
y_prob = model.predict_proba(X_test)[:, 1]

# compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.savefig("outputs/roc_curve.png")
plt.show()

import shap

# SHAP explainer for Logistic Regression
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# summary plot
shap.summary_plot(shap_values, X_test, show=False)

plt.savefig("outputs/shap_summary.png")
plt.show()