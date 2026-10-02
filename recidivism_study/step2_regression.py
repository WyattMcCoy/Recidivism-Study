"""
Step 2: Logistic regression on the synthetic cohort.

Fits two models on 24-month recidivism:
  Model A (naive):    intervention only
  Model B (adjusted): intervention + prior record + risk score + age
Comparing them shows how confounding distorts the naive comparison.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("synthetic_juvenile_cohort.csv")

# Standardize risk score so its coefficient means "per 1 SD increase"
df["risk_z"] = (df["risk_score"] - df["risk_score"].mean()) / df["risk_score"].std()

# Standard Supervision is the reference group every program is compared to
arm = "C(intervention, Treatment('Standard Supervision'))"

model_a = smf.logit(f"recid_24m ~ {arm}", data=df).fit(disp=False)
model_b = smf.logit(
    f"recid_24m ~ {arm} + prior_record + risk_z + age", data=df
).fit(disp=False)


def odds_ratio_table(model):
    ci = model.conf_int()
    table = pd.DataFrame({
        "odds_ratio": np.exp(model.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": model.pvalues,
    }).round(3)
    table.index = (table.index
                   .str.replace(arm, "", regex=False)
                   .str.replace("[T.", "", regex=False)
                   .str.replace("]", "", regex=False))
    return table.drop(index="Intercept")


print("MODEL A: intervention only (naive)")
print(odds_ratio_table(model_a), "\n")
print("MODEL B: adjusted for prior record, risk, and age")
print(odds_ratio_table(model_b), "\n")
print(model_b.summary())
