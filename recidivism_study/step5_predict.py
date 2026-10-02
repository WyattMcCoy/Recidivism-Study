"""
Step 5: Predictive model with scikit-learn.

Question: at intake, before a program is chosen, how well can we predict
which youth will re-offend within 24 months?

Design decision (Travis's call): no cutoff. The tool reports each youth's
probability of re-offending and leaves the decision to a person. So instead
of counting flag errors, we check two things:
  1. Ranking (AUC): do re-offenders get higher scores?
  2. Calibration: when the tool says 30%, do about 30% re-offend?
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv("synthetic_juvenile_cohort.csv")

# Intake information only. Intervention is left out because the tool
# would be used before a program is assigned.
FEATURES = ["age", "prior_record", "school_attendance", "family_support"]
X = df[FEATURES]
y = df["recid_24m"]

# ---- 1. Train/test split ---------------------------------------------------
# stratify=y keeps the same re-offense rate in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Training on {len(X_train)} youth, testing on {len(X_test)}\n")

# ---- 2. Fit the model ------------------------------------------------------
# StandardScaler puts every feature on the same scale before fitting
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

test = X_test.copy()
test["actual"] = y_test.values
test["predicted_%"] = (model.predict_proba(X_test)[:, 1] * 100).round(1)

# ---- 3. What the tool reports ----------------------------------------------
print("Sample output: what a probation officer would see")
print(test[FEATURES + ["predicted_%"]].head(5).to_string(), "\n")

# ---- 4. Ranking ability (AUC) ----------------------------------------------
auc = roc_auc_score(test["actual"], test["predicted_%"])
print(f"AUC on test set: {auc:.3f}  (0.5 = coin flip, 1.0 = perfect ranking)\n")

# ---- 5. Calibration --------------------------------------------------------
# Sort kids into five equal-sized groups by predicted risk, then compare
# the average prediction to the share who actually re-offended.
test["risk_group"] = pd.qcut(test["predicted_%"], 5,
                             labels=["1 (lowest)", "2", "3", "4", "5 (highest)"])
calib = test.groupby("risk_group", observed=True).agg(
    youth=("actual", "size"),
    avg_predicted_pct=("predicted_%", "mean"),
    actual_pct=("actual", lambda s: 100 * s.mean()),
).round(1)
print("Calibration: predicted vs. actual re-offense rate")
print(calib, "\n")

# ---- 6. Calibration by group -----------------------------------------------
# A fair probability should mean the same thing for every group.
test["offender_status"] = test["prior_record"].map({0: "First-Time",
                                                    1: "Prior Record"})
by_group = test.groupby("offender_status").agg(
    youth=("actual", "size"),
    avg_predicted_pct=("predicted_%", "mean"),
    actual_pct=("actual", lambda s: 100 * s.mean()),
).round(1)
print("Calibration by offender status")
print(by_group)
