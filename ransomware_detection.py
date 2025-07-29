import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("models/ensemble_model.pkl")
scaler = joblib.load("models/ensemble_scaler.pkl")

st.set_page_config(page_title="🔐 Ransomware Detection (PE Headers)", layout="wide")
st.title("Ransomware Detection using Ensemble Learning (Mendeley Dataset)")

st.markdown("Upload a CSV file with **1024 numeric PE header features (columns named '0' to '1023')**. "
            "If the file contains the optional `'GR'` column, evaluation will be shown.")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully.")

    expected_columns = [str(i) for i in range(1024)]
    if not all(col in df.columns for col in expected_columns):
        st.error("❌ Uploaded file must contain 1024 features named from '0' to '1023'.")
    else:
        X = df[expected_columns]
        y = df['GR'] if 'GR' in df.columns else None

        # Scale input features
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        pred_labels = ["Ransomware" if p == 1 else "Benign" for p in preds]

        pred_counts = pd.Series(pred_labels).value_counts().to_dict()
        st.markdown("### 🔍 Prediction Distribution")
        st.write(pred_counts)

        if 1 in preds:
            st.markdown("""
            <div style='padding:10px; border-radius:5px; background-color:#ffe6e6; color:#990000;'>
                ⚠️ <strong>Alert:</strong> Ransomware activity detected in the uploaded file!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No ransomware detected in this file.")

        result_df = df.copy()
        result_df['Prediction'] = pred_labels

        def highlight_ransomware(val):
            return 'background-color: #ffcccc' if val == 'Ransomware' else 'background-color: #ccffcc'

        st.markdown("### 🧾 Prediction Results")
        st.dataframe(result_df.style.map(highlight_ransomware, subset=['Prediction']))

        if y is not None:
            st.markdown("### 📊 Evaluation Metrics")
            report = classification_report(y, preds, output_dict=True, zero_division=0)
            st.write(pd.DataFrame(report).transpose())

            cm = confusion_matrix(y, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Benign", "Ransomware"],
                        yticklabels=["Benign", "Ransomware"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
