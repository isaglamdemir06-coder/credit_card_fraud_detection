# 💳 Credit Card Fraud Detection with Machine Learning & SMOTE

This repository contains an end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions on highly imbalanced data.

---

## 📌 Project Overview
Credit card fraud detection is a classic **imbalanced classification** problem. In real-world financial datasets, legitimate transactions heavily outnumber fraudulent ones (~0.17%). Standard accuracy-driven models often fail by missing fraud cases while scoring high accuracy.

This project focuses on optimizing **Recall** to minimize financial losses (False Negatives) while maintaining a balanced **Precision** to avoid overwhelming legitimate customers with false alarms.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Handling & Processing:** Pandas, NumPy, Scikit-Learn
* **Imbalanced Data Handling:** `imbalanced-learn` (SMOTE)
* **ML Algorithms:** Logistic Regression, Random Forest, XGBoost
* **Visualization:** Seaborn, Matplotlib

---

## ⚙️ Methodology & Optimization Pipeline
1. **Data Preprocessing & Scaling:** Standardized `Amount` and `Time` features while maintaining clean dataset splits.
2. **Handling Class Imbalance:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly to the training split to prevent data leakage.
3. **Model Benchmarking:** Compared baseline models against SMOTE-enhanced models.

---

## 📊 Model Performance Comparison

| Model | Technique | Precision | Recall | F1 Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | SMOTE | 0.1172 | 0.8526 | 0.2061 | High False Positives |
| **XGBoost** | SMOTE | 0.5166 | 0.8211 | 0.6341 | Moderate Balance |
| **Random Forest** | **SMOTE** | **0.6124** | **0.8316** | **0.7054** | **🏆 Winner Model** |

---

## 📈 Key Results & Business Impact
* **Winner Model:** **Random Forest Classifier + SMOTE**
* **Fraud Detection (Recall):** Successfully caught **79 out of 95** fraud test cases (~83.2% Recall).
* **False Alarms:** Maintained an extremely low False Positive rate across **56,600+** normal transactions (only 50 false alarms).
* **Outcome:** Significantly reduced missed fraud cases compared to baseline models without impacting legitimate user experience.
