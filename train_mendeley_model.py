import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

csv_path = r"data\Ransomware PE Header Feature Dataset\Ransomware_headers.csv"
df = pd.read_csv(csv_path)

df_ransom = df[df['GR'] == 1]
df_benign = df[df['GR'] == 0]

min_len = min(len(df_ransom), len(df_benign))

df_ransom_sampled = df_ransom.sample(n=min_len, random_state=42)
df_benign_sampled = df_benign.sample(n=min_len, random_state=42)

df_balanced = pd.concat([df_ransom_sampled, df_benign_sampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle


print("🔍 Balanced Class Distribution:\n", df_balanced['GR'].value_counts())

X = df_balanced.loc[:, [str(i) for i in range(1024)]]
y = df_balanced['GR']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb)
], voting='hard')


y_pred = ensemble_model.predict(X_test)

print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("🔹 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("🔹 Precision:", round(precision_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
print("🔹 Recall:", round(recall_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
print("🔹 F1 Score:", round(f1_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

os.makedirs("models", exist_ok=True)
joblib.dump(ensemble_model, "models/ensemble_model.pkl")
joblib.dump(scaler, "models/ensemble_scaler.pkl")

print("\n Trained & saved balanced ensemble model + scaler.")
