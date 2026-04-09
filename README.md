# 📉 Customer Churn Predictor

A machine learning project to predict whether a telecom customer will churn (leave the service), using classification algorithms and explainable AI techniques.

---

## 🎯 Problem Statement

Customer churn costs businesses billions annually. Retaining an existing customer is **5x cheaper** than acquiring a new one. This project builds a predictive model that identifies at-risk customers early, enabling proactive retention strategies.

---

## 📁 Project Structure

```
customer-churn-predictor/
│
├── data/                   # Raw and processed datasets
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering & cleaning
│   └── 03_modelling.ipynb  # Model training & evaluation
│
├── src/
│   ├── preprocess.py       # Data preprocessing functions
│   ├── train.py            # Model training script
│   └── predict.py          # Prediction on new data
│
├── models/                 # Saved model files (.pkl)
├── reports/                # Figures, plots, results
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

- **7,043 rows**, 21 features
- Target: `Churn` (Yes / No)
- Features include: tenure, contract type, monthly charges, internet service, payment method, etc.

---

## 🔬 Approach

1. **EDA** — Understand churn distribution, correlations, and feature patterns
2. **Preprocessing** — Handle missing values, encode categoricals, scale numerics
3. **Modelling** — Train and compare:
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
4. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
5. **Explainability** — SHAP values to interpret model decisions

---

## 📈 Results

| Model               | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression | ~80%    | ~0.58    | ~0.84   |
| Random Forest       | ~82%    | ~0.61    | ~0.87   |
| XGBoost             | ~83%    | ~0.63    | ~0.88   |

> Results will be updated after training.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/rathan-raj/customer-churn-predictor.git
cd customer-churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in data/
#    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 4. Run notebooks in order
jupyter notebook notebooks/
```

---

## 📌 Key Learnings

- Handling class imbalance with SMOTE
- Comparing multiple classifiers and tuning hyperparameters
- Using SHAP to explain model predictions to non-technical stakeholders
- End-to-end ML project lifecycle

---

## 🙋 Author

**Rathan Raj** · [GitHub](https://github.com/rathan-raj) · [Email](mailto:rathanraj.dasari@gmail.com)
