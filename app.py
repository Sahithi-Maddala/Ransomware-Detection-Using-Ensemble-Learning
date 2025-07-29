import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("models/ensemble_model.pkl")
scaler = joblib.load("models/ensemble_scaler.pkl")

st.set_page_config(page_title="Ransomware Detection - Mendeley", layout="wide")
st.title("💻 Ransomware Detection using PE Header Features (Mendeley Dataset)")
st.markdown("Upload a CSV file with **1024 numeric features named 0–1023**, optionally with a `GR` label column.")


uploaded_file = st.file_uploader("📁 Upload Mendeley CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully.")

    expected_features = [str(i) for i in range(1024)]
    has_all_features = all(col in df.columns for col in expected_features)

    if not has_all_features:
        st.error("❌ Uploaded file must contain columns named from '0' to '1023'.")
    else:
        X = df[expected_features]
        y = df['GR'] if 'GR' in df.columns else None

        try:
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            pred_labels = ["Ransomware" if p == 1 else "Benign" for p in preds]

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

            st.markdown("### 🧾 Prediction Results:")
            st.dataframe(result_df.style.map(highlight_ransomware, subset='Prediction'))

            # Class prediction stats
            unique, counts = np.unique(preds, return_counts=True)
            pred_stats = dict(zip(["Benign" if u == 0 else "Ransomware" for u in unique], counts))
            st.markdown("### 🔍 Prediction Distribution:")
            st.write(pred_stats)

            if y is not None:
                st.markdown("### 📊 Evaluation Metrics:")
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

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
