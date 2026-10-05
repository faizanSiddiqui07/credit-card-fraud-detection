# Credit Card Fraud Detection

Machine learning pipeline for detecting fraudulent credit card transactions on a highly imbalanced dataset (0.17% fraud). Four models are trained and compared: XGBoost, Random Forest, LightGBM and KNN. The final model is XGBoost.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | **99.95%** | 94.67% | **74.74%** | **83.53%** | 0.9771 | **0.8233** |
| Random Forest | 99.95% | **95.89%** | 73.68% | 83.33% | 0.9354 | 0.8141 |
| LightGBM | 99.95% | 93.24% | 72.63% | 81.66% | **0.9785** | 0.8133 |
| KNN (K=15) | 99.94% | 94.20% | 68.42% | 79.27% | 0.9102 | 0.7888 |

Decision thresholds were selected on out-of-fold validation predictions, not on the test set: XGBoost 0.35, Random Forest 0.40, LightGBM 0.42, KNN 0.55. Full numbers, including confusion matrix counts, are in [`results/model_comparison.csv`](results/model_comparison.csv).

Accuracy is high for every model because 99.8% of transactions are legitimate. PR-AUC, recall and precision are the metrics that separate the models.

## Dataset

Public credit card fraud dataset of European cardholder transactions (Kaggle: *Credit Card Fraud Detection*, ULB / Worldline).

- Columns: `Time`, `V1`-`V28` (anonymized PCA components), `Amount`, `Class` (1 = fraud)
- The dataset file is not included in this repository. Download `creditcard.csv` and place it in the project root.

| Stage | Rows | Fraud |
|---|---|---|
| Raw | 284,807 | 492 |
| After removing duplicates and an incomplete row | 283,726 | 473 |
| Train (80%) | 226,980 | 378 |
| Test (20%) | 56,746 | 95 |

## Approach

1. **Cleaning.** Dropped the incomplete row and duplicate rows before splitting, so identical transactions cannot appear in both train and test.
2. **Split.** Stratified 80/20 train/test split. The test set was used only for the final evaluation.
3. **Feature engineering.** Original `V1`-`V28`, plus `log(Amount)`, hour of day encoded as sine/cosine, the Euclidean norm of the V vector and its maximum absolute value. None of the features use the target.
4. **Validation.** With 378 training frauds, a single hold-out set is too noisy. Models were compared using 5-fold stratified out-of-fold predictions on the training data.
5. **Threshold selection.** Chosen on out-of-fold predictions: best F1 subject to precision of at least 0.90.
6. **Models.** XGBoost, Random Forest (balanced subsample), LightGBM and KNN (distance weighted, K selected by validation PR-AUC).
7. **Hyperparameter search.** Time-boxed random search for XGBoost and LightGBM, scored by out-of-fold PR-AUC. LightGBM improved from 0.8319 to 0.8464 on validation. XGBoost improved from 0.8495 to 0.8515. The default XGBoost configuration was kept as the final model because the tuned version did not give a meaningful difference on the test set.

## Repository structure

```
credit-card-fraud-detection/
├── README.md
├── requirements.txt
├── features.py                 # feature engineering shared by training and inference
├── predict.py                  # score a CSV with a saved model
├── models/
│   ├── fraud_xgboost.pkl       # final model
│   ├── fraud_random_forest.pkl
│   ├── fraud_lightgbm.pkl
│   ├── fraud_knn.pkl
│   ├── fraud_xgboost_tuned.pkl
│   └── fraud_lightgbm_tuned.pkl
├── notebooks/
│   └── fraud_detection.ipynb   # full training and evaluation pipeline
└── results/
    └── model_comparison.csv
```

Each `.pkl` file is a joblib bundle containing the fitted model, the decision threshold and the feature names.

## Installation

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

The saved models were trained with scikit-learn 1.6.1, XGBoost 3.4.1 and LightGBM 4.6.0. Loading them with different versions may fail or give different results, so `requirements.txt` pins these three.

## Usage

Command line:

```bash
python predict.py transactions.csv --model models/fraud_xgboost.pkl --out predictions.csv
```

Python:

```python
import joblib
import pandas as pd
from features import make_features

bundle = joblib.load("models/fraud_xgboost.pkl")
df = pd.read_csv("transactions.csv")        # requires Time, V1..V28, Amount
X = make_features(df)[bundle["features"]]

proba = bundle["model"].predict_proba(X)[:, 1]
is_fraud = (proba >= bundle["threshold"]).astype(int)
```

## Limitations

- The test set contains 95 fraud cases, so a single transaction changes recall by about 1 percentage point. Differences of that size between models are not conclusive.
- Recall is around 75% at the chosen operating point. Raising it substantially on this data costs a large drop in precision.
- The data covers a short period of European card transactions and may not transfer to other issuers without retraining.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Matplotlib, Google Colab.
