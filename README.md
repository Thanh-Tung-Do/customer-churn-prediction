# Customer Churn Prediction — Kaggle PS S6E3

End-to-end ML solution for the [Kaggle Playground Series Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3/overview) customer churn competition. 594K training rows, binary classification, evaluated on **ROC-AUC**.

## Business Framing

Customer acquisition costs 5–25x more than retention. This model identifies at-risk customers **before** they leave, enabling targeted retention offers. Beyond the Kaggle metric, the cost analysis section quantifies what each model is worth in real dollar terms — false negatives (missed churners) and false positives (wasted retention spend) have very different costs.

## Project Structure

```
├── data/
│   ├── raw/                  # Original competition CSVs (not committed)
│   └── processed/            # Cleaned data, submission files
├── notebooks/
│   ├── 01_eda.ipynb          # EDA & business insights
│   └── 02_modeling.ipynb     # Preprocessing, modeling, evaluation, submission
├── src/
│   ├── preprocessing.py      # Reusable pipeline components
│   ├── model_utils.py        # Evaluation helpers, cost matrix, submission export
│   └── models/               # Saved .pkl model files
├── app/                      # Flask API (coming soon)
├── requirements.txt
└── README.md
```

## Models & Results

| Model | CV ROC-AUC | Notes |
|---|---|---|
| Logistic Regression (baseline) | TBD | Interpretable, fast |
| Logistic Regression (cost-sensitive) | TBD | `class_weight='balanced'` |
| XGBoost | TBD | Best performer |

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab
# Open notebooks/01_eda.ipynb, then 02_modeling.ipynb
```

## Competition

- **Platform**: Kaggle Playground Series S6E3
- **Task**: Binary classification — predict customer churn
- **Metric**: ROC-AUC
- **Data**: ~594K train / ~255K test rows, 20 features
- **Dataset origin**: Synthetically generated from the IBM Telco Customer Churn dataset

## Live Demo

_Link coming soon._
