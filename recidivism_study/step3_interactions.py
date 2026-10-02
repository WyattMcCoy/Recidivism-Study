"""
Step 3: Interaction terms.

Model C: does each program's effect differ by offender status?
Model D: does each program's effect differ by risk tier?

Each is compared to the matching no-interaction model with a
likelihood-ratio test, then turned into predicted recidivism rates.
"""
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_csv("synthetic_juvenile_cohort.csv")
df["risk_z"] = (df["risk_score"] - df["risk_score"].mean()) / df["risk_score"].std()

arm = "C(intervention, Treatment('Standard Supervision'))"
tier = "C(risk_tier, Treatment('Low'))"

# ---- 1. Cell sizes: how many kids land in each group? ----------------------
print("Kids per program x offender status")
print(pd.crosstab(df["intervention"], df["offender_status"]), "\n")
print("Kids per program x risk tier")
print(pd.crosstab(df["intervention"], df["risk_tier"])
      [["Low", "Moderate", "High"]], "\n")


# ---- 2. Fit models with and without the interaction ------------------------
def lr_test(small, big, label):
    """Does adding the interaction terms improve the model beyond chance?"""
    stat = 2 * (big.llf - small.llf)
    df_diff = big.df_model - small.df_model
    p = stats.chi2.sf(stat, df_diff)
    print(f"{label}: chi2 = {stat:.2f}, df = {df_diff:.0f}, p = {p:.4f}")


# Offender status
c_main = smf.logit(f"recid_24m ~ {arm} + prior_record + risk_z + age",
                   data=df).fit(disp=False)
c_int = smf.logit(f"recid_24m ~ {arm} * prior_record + risk_z + age",
                  data=df).fit(disp=False)

# Risk tier (risk_z is left out: the tier is built from the same score)
d_main = smf.logit(f"recid_24m ~ {arm} + {tier} + prior_record + age",
                   data=df).fit(disp=False)
d_int = smf.logit(f"recid_24m ~ {arm} * {tier} + prior_record + age",
                  data=df).fit(disp=False)

print("Likelihood-ratio tests")
lr_test(c_main, c_int, "Program x offender status")
lr_test(d_main, d_int, "Program x risk tier       ")
print()


# ---- 3. Predicted recidivism for each program in each group ----------------
def predicted_rates(model, group_col, group_values):
    """Predict for every kid as if they were in each program and group,
    then average. Keeps everyone's other traits as they really are."""
    rows = []
    for program in df["intervention"].unique():
        for g in group_values:
            scenario = df.copy()
            scenario["intervention"] = program
            scenario[group_col] = g
            rows.append({"program": program, group_col: g,
                         "pred_rate": model.predict(scenario).mean()})
    table = pd.DataFrame(rows).pivot(index="program", columns=group_col,
                                     values="pred_rate")
    return (table * 100).round(1)


print("Predicted 24-month recidivism (%), by offender status")
status = predicted_rates(c_int, "prior_record", [0, 1])
status.columns = ["First-Time", "Prior Record"]
print(status, "\n")

print("Predicted 24-month recidivism (%), by risk tier")
print(predicted_rates(d_int, "risk_tier", ["Low", "Moderate", "High"])
      [["Low", "Moderate", "High"]])
