# 🛡️ Ransomware Detection using Ensemble Learning

This project presents an intelligent ransomware detection system that leverages static analysis of PE (Portable Executable) header features and an ensemble of machine learning models (Random Forest and XGBoost). By applying feature engineering, preprocessing, and model fusion techniques, the system effectively identifies ransomware with high accuracy.

## ✨ Features

📁 Static analysis using PE header features
🧠 Ensemble model with Random Forest & XGBoost
📊 High-performance metrics (97%+ accuracy)
📈 Real-time classification via Streamlit UI
📦 Saved model & scaler for deployment

## ⚙️ Tech Stack

👩‍💻 Python
📚 Libraries: Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
🧪 Streamlit (for UI)
📂 Mendeley PE file dataset

## 📈 Performance Metrics

✅ Accuracy: 97.56%

🎯 Precision: 98.03%

🔁 Recall: 97.07%

📊 F1 Score: 97.55%

## 🧠 Feature Analysis

🔍 Used correlation heatmap to remove redundant features (multicollinearity)

✂️ Feature selection boosted model efficiency and generalization

## 🔬 Model Pipeline

Load and preprocess PE header dataset

Feature scaling using StandardScaler

Train ensemble model using VotingClassifier (Random Forest + XGBoost)

Evaluate performance on test set

Save model & scaler using joblib

Deploy with Streamlit UI for predictions

## 🚀 Streamlit App

To run the Streamlit UI locally:

streamlit run ransomware_detector.py

## 📂 Project Setup

Clone the repository and install dependencies:

git clone https://github.com/Sahithi-Maddala/ransomware-detection-ensemble.git
cd ransomware-detection-ensemble
pip install -r requirements.txt
streamlit run ransomware_detector.py

## 🧪 Sample Prediction

Upload a .csv or input PE header values to predict:

Label: 0 → Benign

Label: 1 → Ransomware

## 📌 How to Use

Input PE header features via Streamlit form

The system will return “Benign” or “Ransomware”

Based on trained ensemble model on static features

## 🔐 Why Use Feature Scaling?

StandardScaler transforms feature values to a standard normal distribution (mean = 0, std = 1). This is essential because:

Algorithms like XGBoost and Random Forest are sensitive to feature magnitude.

Scaling ensures convergence and balance in feature importance.

## 🛠️ Future Improvements

🧬 Include dynamic features (API calls, system logs)

📶 Incorporate real-time monitoring via hybrid analysis

📦 Expand dataset for broader ransomware families

🔗 Integrate with antivirus and SIEM systems

## 👩‍💻 Author

M. Jhansi Sahithi Maddala
Cybersecurity Student – Vignan Institute of Engineering for Women
GitHub: github.com/Sahithi-Maddala

## 📄 License

This project is licensed under the MIT License.

Let me know if you'd like me to tailor this for a GitHub markdown file or include your actual GitHub repo URL.
