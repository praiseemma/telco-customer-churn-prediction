# Telco Customer Churn Prediction

A data-driven analysis of customer churn for a telecommunications provider identifying why customers leave, predicting which current customers are at risk, and translating both into concrete retention recommendations.

**[Try the live dashboard →**](#)



## Project Summary

Using IBM's Telco Customer Churn dataset (7,043 customers), this project answers three questions for the business:

1. **Diagnose** — which customer attributes are most strongly associated with churn?

2. **Predict** — can we build a model that scores existing customers by churn risk?

3. **Act** — what specific, prioritized retention actions follow from the findings?

The project covers the full data science workflow: data cleaning, exploratory analysis, feature engineering, predictive modeling, individual-level model interpretability, and a deployed interactive tool packaged into a client-ready report, slide deck, and live dashboard.

## Key Findings

- **Contract type is the strongest single lever.** Month-to-month customers churn at **42.7%**, compared to just **2.8%** for two-year contract holders over 15x higher. Most of the retention benefit comes from moving customers into *any* fixed-term contract, not maximizing contract length.

- **Three risk factors compound independently.** Contract type, internet service (Fiber optic), and payment method (Electronic check) each raise churn risk on their own and stack when combined. Customers with Fiber optic on a month-to-month contract churn at **54.6%**, the highest-risk segment identified.

- **Churn is front-loaded.** Nearly half of all churn happens within a customer's first 12 months, and this effect is concentrated among month-to-month customers specifically.

- **Add-on bundling is a strong, actionable lever, but not evenly.** Customers with zero add-ons churn at over 50%; those with all six churn at just 5%. Online Security and Tech Support carry far more protective value than Streaming TV/Movies, despite streaming being more popular.

- **The model doesn't just score risk, it explains it.** For any customer, the model can break down *why* they're flagged as high-risk, enabling retention offers tailored to the actual cause rather than a generic response.

Full findings, methodology, and caveats are in the [written report](file:///C:/Users/Praiseemma/Documents/GitHub/telco-customer-churn-prediction/report/).

## Tech Stack

| Purpose | Tools |
| - | - |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn (Logistic Regression, Random Forest) |
| Model interpretability | SHAP |
| Dashboard | Streamlit |


## Model Performance

| Metric | Logistic Regression | Random Forest (tuned threshold) |
| - | - | - |
| Recall (Churn) | 0.79 | 0.78 |
| Precision (Churn) | 0.51 | 0.50 |
| ROC-AUC | 0.84 | 0.83 |


Both models achieve similar discriminative power (ROC-AUC ≈ 0.83–0.84); the Random Forest's classification threshold was tuned from the default 0.5 to 0.25 to prioritize catching at-risk customers, aligning its recall with the logistic regression baseline. Performance was validated with 5-fold cross-validation (mean ROC-AUC 0.822).

## Repository Structure

```
├── README.md  
├── notebooks/  
│   ├── 01\_eda.ipynb                       \# Cleaning + univariate/bivariate/multivariate EDA  
│   ├── 02\_logistic\_regression.ipynb       \# Baseline model, VIF, coefficients  
│   └── 03\_random\_forest.ipynb             \# Stronger model, threshold tuning, SHAP  
├── report/  
│   └── Telco\_Churn\_Consolidated\_Report.docx  
├── slides/  
│   └── Telco\_Churn\_Slide\_Deck.pptx  
├── dashboard/  
│   ├── app.py                             \# Streamlit app  
│   ├── requirements.txt  
│   ├── churn\_model.pkl                    \# Trained Random Forest model  
│   └── scaler.pkl                         \# Fitted StandardScaler  
└── docs/  
    └── dashboard\_preview.png              \# Screenshot for this README
```

## Running the Dashboard Locally

```
cd dashboard  
pip install -r requirements.txt  
streamlit run app.py
```

The dashboard lets you input a customer's details (tenure, contract, internet service, payment method, add-ons, household structure) and returns:

- A live churn risk score, using a business-tuned classification threshold

- A breakdown of the specific factors driving that score up or down

## Dataset

The dataset used was [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and it contains 7,043 customers, 21 original features, publicly available on Kaggle for educational and portfolio use.

## Limitations & Next Steps

- This analysis reflects a single historical snapshot; churn drivers may shift as pricing, competition, or service quality change over time.

- Relationships identified are statistical associations, not confirmed causal mechanisms — recommended interventions (e.g., contract-conversion incentives) would ideally be validated through small-scale testing before full rollout.

- Planned improvements: an XGBoost comparison, and additional input validation safeguards in the dashboard.

## Author

Built by Praise as a portfolio project demonstrating an end-to-end data science workflow — from raw data to a deployed, interpretable decision-support tool.

