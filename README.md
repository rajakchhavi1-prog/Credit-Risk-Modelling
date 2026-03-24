# Credit Card Default Risk Prediction

Machine learning project to predict whether a credit card customer will default on their payment in the next month. The project compares multiple models and incorporates cost-sensitive evaluation and explainability techniques.

---

# Project Overview

Credit risk modeling is a critical task for financial institutions. Predicting whether a customer will default allows banks to manage risk, reduce losses, and make informed lending decisions.

This project builds machine learning models to predict default probability using the **Default of Credit Card Clients Dataset**.

The workflow includes:

- Data preprocessing
- Model training and comparison
- Evaluation using ROC-AUC and PR-AUC
- Cost-sensitive threshold optimization
- Model explainability using SHAP

---

# Dataset

Source: UCI Machine Learning Repository

The dataset contains information about credit card clients in Taiwan and includes features such as:

- Credit limit
- Gender
- Education
- Marital status
- Age
- Payment history
- Bill statements
- Previous payment amounts

Target variable:

`default.payment.next.month`

- **0 → No Default**
- **1 → Default**

Dataset size:

- 30,000 customers
- 23 features

---

# Project Workflow

## 1. Data Preprocessing

- Removed ID column
- Renamed target variable to `default`
- Train-test split (80 / 20)
- Standard scaling applied for Logistic Regression

---

## 2. Models Implemented

Three models were trained and compared.

### Logistic Regression
Baseline interpretable model commonly used in credit risk scoring.

### Random Forest
Ensemble tree-based model capable of capturing nonlinear relationships.

### XGBoost
Gradient boosting model optimized for structured/tabular data.

---

# Model Evaluation

Because credit default prediction is an **imbalanced classification problem**, accuracy is not an appropriate metric. Instead, the following metrics were used:

- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

### Performance Comparison

| Model | ROC-AUC | PR-AUC |
|------|--------|--------|
| Logistic Regression | 0.7076 | 0.4934 |
| Random Forest | 0.7731 | 0.5513 |
| XGBoost | **0.7784** | **0.5579** |

XGBoost achieved the best overall performance.

---

# Cost-Sensitive Threshold Optimization

In real-world credit risk modeling, the cost of misclassification is asymmetric.

Example cost structure:

| Prediction Error | Business Cost |
|-----------------|--------------|
| False Negative (missed default) | $5000 |
| False Positive (reject good customer) | $50 |

Instead of using the default threshold of **0.5**, an optimal threshold was determined by minimizing expected financial loss.

Optimal threshold:

```
0.01
```

This reflects the high cost associated with failing to detect a potential default.

---

# Model Explainability

Model interpretability is critical in financial risk modeling.

This project uses **SHAP (SHapley Additive exPlanations)** to explain the contribution of each feature to model predictions.

SHAP analysis showed that the most influential features include:

- Recent payment delays (PAY_0 – PAY_6)
- Credit limit
- Payment amounts
- Billing history

These findings align with real-world credit risk factors.

---

# Key Insights

- Payment history is the strongest predictor of credit default.
- Tree-based models outperform linear models for this dataset.
- Cost-sensitive optimization significantly changes classification thresholds.
- Explainability tools like SHAP help interpret complex models.

---

# Technologies Used

**Programming Language**

- Python

**Libraries**

- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn

---



