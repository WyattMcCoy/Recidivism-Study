"""
Generate a synthetic, anonymized dataset of justice-involved male adolescents.

Every record is simulated. The effect sizes below are ASSUMPTIONS built into the
simulation (loosely informed by juvenile-justice meta-analyses: CBT tends to
reduce re-offending, deterrence-style "tough love" programs tend to do nothing
or make it worse). The analysis will recover these assumptions; it cannot
discover anything real. Swap in real data before drawing conclusions.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 3000

# ---- Tunable effect sizes (log-odds scale) ---------------------------------
BASE = -1.10
EFFECTS = {
    "prior_record": 0.85,
    "risk_score": 0.45,          # per 1 SD of the composite risk score
    "age_per_year": -0.10,       # older youth age out slightly
    "CBT": -0.55,
    "Vocational": -0.30,
    "Tough Love": 0.20,
    # Risk principle: intensive services matter most for higher-risk youth
    "CBT_x_high": -0.35,
    "Vocational_x_age16plus": -0.30,
    "ToughLove_x_first_time": 0.25,
}
SHARE_BY_12M = 0.62  # share of 24-month re-offenses that happen in year one


def main() -> pd.DataFrame:
    age = RNG.integers(13, 18, N)
    prior_record = RNG.binomial(1, 0.45, N)

    # Attendance (proportion of school days) and family support (1-10 scale)
    school_attendance = np.clip(RNG.beta(5, 2, N) - 0.12 * prior_record, 0, 1)
    family_support = np.clip(
        np.round(RNG.normal(6.0 - 0.8 * prior_record, 2.0, N)), 1, 10
    ).astype(int)

    # Composite risk score (0-100): higher = riskier
    raw = (
        1.2 * prior_record
        - 2.0 * (school_attendance - 0.7)
        - 0.35 * (family_support - 6)
        + RNG.normal(0, 0.6, N)
    )
    risk_score = np.round(100 / (1 + np.exp(-raw)), 1)
    risk_z = (risk_score - risk_score.mean()) / risk_score.std()
    risk_tier = pd.cut(
        risk_score, bins=[-1, 40, 65, 101], labels=["Low", "Moderate", "High"]
    )

    # Intervention assignment with mild confounding: higher-risk youth are
    # steered toward tough-love programs, older youth toward vocational.
    arms = np.array(["Standard Supervision", "CBT", "Vocational", "Tough Love"])
    logits = np.column_stack([
        np.zeros(N),
        np.full(N, 0.1),
        0.35 * (age - 15),
        0.5 * risk_z,
    ])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    intervention = np.array([RNG.choice(arms, p=p) for p in probs])

    high = (risk_tier == "High").astype(int)
    lin = (
        BASE
        + EFFECTS["prior_record"] * prior_record
        + EFFECTS["risk_score"] * risk_z
        + EFFECTS["age_per_year"] * (age - 15)
        + np.where(intervention == "CBT",
                   EFFECTS["CBT"] + EFFECTS["CBT_x_high"] * high, 0)
        + np.where(intervention == "Vocational",
                   EFFECTS["Vocational"]
                   + EFFECTS["Vocational_x_age16plus"] * (age >= 16), 0)
        + np.where(intervention == "Tough Love",
                   EFFECTS["Tough Love"]
                   + EFFECTS["ToughLove_x_first_time"] * (1 - prior_record), 0)
    )
    p24 = 1 / (1 + np.exp(-lin))
    recid_24m = RNG.binomial(1, p24)
    recid_12m = recid_24m * RNG.binomial(1, SHARE_BY_12M, N)

    df = pd.DataFrame({
        "youth_id": [f"Y{i:05d}" for i in range(N)],
        "age": age,
        "offender_status": np.where(prior_record == 1, "Prior Record", "First-Time"),
        "prior_record": prior_record,
        "school_attendance": school_attendance.round(3),
        "family_support": family_support,
        "risk_score": risk_score,
        "risk_tier": risk_tier.astype(str),
        "intervention": intervention,
        "recid_12m": recid_12m,
        "recid_24m": recid_24m,
    })
    return df


if __name__ == "__main__":
    df = main()
    df.to_csv("synthetic_juvenile_cohort.csv", index=False)
    print(df.head())
    print(f"\n{len(df)} records written to synthetic_juvenile_cohort.csv")
    print(df.groupby("intervention")[["recid_12m", "recid_24m"]].mean().round(3))
