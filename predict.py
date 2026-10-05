import argparse
import joblib
import pandas as pd
from features import make_features


def main():
    ap = argparse.ArgumentParser(description="Score transactions with a saved fraud model")
    ap.add_argument("csv", help="CSV with Time, V1..V28, Amount")
    ap.add_argument("--model", default="models/fraud_xgboost.pkl")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    df = pd.read_csv(a.csv).dropna(subset=["Time", "Amount"])
    X = make_features(df)[bundle["features"]]
    prob = bundle["model"].predict_proba(X)[:, 1]

    out = df.copy()
    out["fraud_probability"] = prob
    out["fraud_prediction"] = (prob >= bundle["threshold"]).astype(int)
    out.to_csv(a.out, index=False)
    print(f"Model: {bundle['model_name']} | threshold: {bundle['threshold']}")
    print(f"Flagged {int(out['fraud_prediction'].sum())} of {len(out)} transactions. Saved to {a.out}")


if __name__ == "__main__":
    main()
