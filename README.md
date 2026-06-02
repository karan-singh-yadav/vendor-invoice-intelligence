# 📊 Vendor Invoice Intelligence System

An end-to-end Machine Learning and MLOps pipeline designed to automate financial auditing, prevent capital leakage, and optimize supply chain accounts payable. The system handles large-scale relational data processing, feature engineering, and hyperparameter tuning to replace manual invoice auditing with AI-driven workflows.

---

## 📌 Business Problem & Impact

In enterprise logistics and retail operations, businesses process thousands of supply chain vendor invoices weekly. Manual verification of freight costs, billing amounts, and delivery delays leads to:
* **Financial Leakage:** Overpaying vendors due to hidden mismatches.
* **Operational Inefficiency:** Accounting teams wasting hours auditing line items.

**The Solution:** This intelligent dual-module portal completely automates the pipeline. It uses machine learning to forecast transport expenses and flags anomalous or high-risk invoices, allowing human auditors to focus **only** on high-risk discrepancies.

---

## 🚀 Key Modules

### 1. 🚚 Freight Cost Prediction (Regression)
* **Goal:** Forecast expected logistics/freight costs for an invoice before payment approval.
* **Model:** **Linear Regression** (selected during EDA due to a strong linear relationship between shipment volume, invoice values, and transport fees).
* **Impact:** Provides an accurate baseline cost. If a vendor charges significantly higher than predicted, the system isolates it.

### 2. 🚩 Invoice Risk Flagging (Classification)
* **Goal:** Classify invoices automatically as "Safe to Auto-Approve" or "Requires Manual Review".
* **Model:** Hyperparameter-tuned **Random Forest Classifier** optimized using `GridSearchCV`.
* **Feature Engineering:** Features were engineered using custom SQL relational joins (calculating rolling metrics like receiving delays). Insignificant features were eliminated via independent **Hypothesis Testing (T-Tests)**.
* **Data Processing:** Robust feature scaling executed using `StandardScaler` to balance numeric distributions between large monetary fields and volume metrics.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.9+
* **Data Layer:** SQLite3, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, GridSearchCV, StandardScaler)
* **Model Deployment:** Joblib (Serialization)
* **Frontend UI:** Streamlit (Interactive Multi-Module Web Portal)

---
SCREENSHOTS 
<img width="1351" height="361" alt="Screenshot_2-6-2026_20187_localhost" src="https://github.com/user-attachments/assets/82cbfc11-7bac-4ad7-bc0a-ccb8f7b826ba" />

<img width="1338" height="482" alt="Screenshot_2-6-2026_201748_localhost" src="https://github.com/user-attachments/assets/756430d0-8ccc-41d3-aab3-131ae9f247d4" />


## 📂 Project Structure

The project strictly follows modular MLOps software engineering principles, isolating data preprocessing, training, evaluation, and user interface layers.

```text
vendor_invoice_intelligence/
│
├── data/
│   └── inventory.db               # Local Relational Database (Ignored by Git due to size)
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py      # SQL extraction & feature preparation for regression
│   ├── model_evaluation.py        # Regression model architecture & validation
│   └── train.py                   # Orchestration script for freight training
│
├── invoice_flagging/
│   ├── data_preprocessing.py      # SQL joins, labeling logic & StandardScaler implementation
│   ├── model_evaluation.py        # Random Forest GridSearch configuration & metrics
│   └── train.py                   # Orchestration script for classification training
│
├── models/                        # Serialized production-ready artifacts
│   ├── predict_freight_model.pkl  
│   ├── predict_flag_invoice.pkl   
│   └── scaler.pkl                 
│
├── .gitignore                     # Prevents tracking .venv and heavy binary database files
├── app.py                         # Streamlit multi-page production web application
└── README.md                      # Comprehensive project documentation
