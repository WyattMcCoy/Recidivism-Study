"""
Step 4: Multi-panel figure comparing intervention efficacy.

  A  Raw recidivism by program, 12 vs 24 months
  B  Adjusted odds ratios (forest plot) from Model B
  C  Interaction plot: predicted rates by risk tier
  D  Interaction plot: predicted rates by offender status

Panels C and D get 95% confidence intervals from a bootstrap.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

df = pd.read_csv("synthetic_juvenile_cohort.csv")
df["risk_z"] = (df["risk_score"] - df["risk_score"].mean()) / df["risk_score"].std()

PROGRAMS = ["Standard Supervision", "CBT", "Vocational", "Tough Love"]
COLORS = dict(zip(PROGRAMS, sns.color_palette("colorblind", 4)))
TIERS = ["Low", "Moderate", "High"]
arm = "C(intervention, Treatment('Standard Supervision'))"
tier = "C(risk_tier, Treatment('Low'))"

FORMULA_B = f"recid_24m ~ {arm} + prior_record + risk_z + age"
FORMULA_STATUS = f"recid_24m ~ {arm} * prior_record + risk_z + age"
FORMULA_TIER = f"recid_24m ~ {arm} * {tier} + prior_record + age"


# ---- Helpers ---------------------------------------------------------------
def predicted_rates(model, data, group_col, group_values):
    """Average predicted recidivism if everyone had program X and group G."""
    out = {}
    for program in PROGRAMS:
        for g in group_values:
            scenario = data.copy()
            scenario["intervention"] = program
            scenario[group_col] = g
            out[(program, g)] = model.predict(scenario).mean()
    return pd.Series(out)


def bootstrap_rates(formula, group_col, group_values, n_boot=200, seed=1):
    """Resample kids with replacement, refit, re-predict. The spread of
    the results across resamples is the uncertainty in each estimate."""
    rng = np.random.default_rng(seed)
    point = predicted_rates(smf.logit(formula, df).fit(disp=False),
                            df, group_col, group_values)
    draws = []
    for _ in range(n_boot):
        sample = df.sample(len(df), replace=True,
                           random_state=rng.integers(1e9))
        fit = smf.logit(formula, sample).fit(disp=False)
        draws.append(predicted_rates(fit, sample, group_col, group_values))
    draws = pd.concat(draws, axis=1)
    return pd.DataFrame({
        "rate": point,
        "low": draws.quantile(0.025, axis=1),
        "high": draws.quantile(0.975, axis=1),
    }) * 100


def interaction_panel(ax, table, group_values, labels, title, xlabel):
    """One line per program; small horizontal offsets stop error bars
    from stacking on top of each other."""
    x = np.arange(len(group_values))
    offsets = np.linspace(-0.12, 0.12, len(PROGRAMS))
    for program, off in zip(PROGRAMS, offsets):
        rows = table.loc[program].loc[group_values]
        ax.errorbar(x + off, rows["rate"],
                    yerr=[rows["rate"] - rows["low"], rows["high"] - rows["rate"]],
                    marker="o", capsize=3, lw=2, color=COLORS[program],
                    label=program)
    ax.set_xticks(x, labels)
    ax.set(title=title, xlabel=xlabel, ylabel="Predicted 24-month recidivism (%)")
    ax.set_ylim(0, None)


# ---- Build the figure ------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
(ax_a, ax_b), (ax_c, ax_d) = axes

# Panel A: raw rates
raw = df.melt(id_vars="intervention", value_vars=["recid_12m", "recid_24m"],
              var_name="window", value_name="reoffended")
raw["window"] = raw["window"].map({"recid_12m": "12 months",
                                   "recid_24m": "24 months"})
raw["reoffended"] *= 100
sns.barplot(data=raw, x="intervention", y="reoffended", hue="window",
            order=PROGRAMS, palette="Greys", errorbar=("ci", 95),
            capsize=0.1, ax=ax_a)
ax_a.set(title="A. Raw recidivism by program", xlabel="",
         ylabel="Re-offended (%)")
ax_a.tick_params(axis="x", rotation=15)
ax_a.legend(title="Follow-up")

# Panel B: forest plot of adjusted odds ratios
model_b = smf.logit(FORMULA_B, df).fit(disp=False)
ci = model_b.conf_int()
rows = [p for p in PROGRAMS[1:]]
names = [f"{arm}[T.{p}]" for p in rows]
or_est = np.exp(model_b.params[names]).values
or_lo = np.exp(ci.loc[names, 0]).values
or_hi = np.exp(ci.loc[names, 1]).values
y = np.arange(len(rows))[::-1]
for yi, p, est, lo, hi in zip(y, rows, or_est, or_lo, or_hi):
    ax_b.errorbar(est, yi, xerr=[[est - lo], [hi - est]], fmt="o",
                  capsize=4, ms=8, lw=2, color=COLORS[p])
    ax_b.text(hi * 1.04, yi, f"{est:.2f}", va="center")
ax_b.axvline(1, color="black", ls="--", lw=1)
ax_b.set_xscale("log")
ax_b.set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
ax_b.xaxis.set_minor_formatter(plt.NullFormatter())  # hide cluttered minor labels
ax_b.set_yticks(y, rows)
ax_b.set(title="B. Adjusted odds ratios vs. Standard Supervision",
         xlabel="Odds ratio (log scale)  <-- fewer re-offenses | more -->")

# Panels C and D: interaction plots with bootstrap CIs
print("Bootstrapping panel C (risk tier)...")
tier_table = bootstrap_rates(FORMULA_TIER, "risk_tier", TIERS)
print("Bootstrapping panel D (offender status)...")
status_table = bootstrap_rates(FORMULA_STATUS, "prior_record", [0, 1])

interaction_panel(ax_c, tier_table, TIERS, TIERS,
                  "C. Program effect by risk tier", "Risk tier")
interaction_panel(ax_d, status_table, [0, 1], ["First-Time", "Prior Record"],
                  "D. Program effect by offender status", "Offender status")

handles, labels = ax_c.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           title="Program (panels B-D)")
fig.suptitle("Juvenile Recidivism: Intervention Efficacy Across Cohorts "
             "(synthetic data, n = {:,})".format(len(df)),
             fontsize=15, weight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
fig.savefig("intervention_efficacy.png", dpi=200)
print("Saved intervention_efficacy.png")
