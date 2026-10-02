# Step 6: Replicate the Step 2 and Step 3 regressions in R.
# Uses base R only, so no packages need to be installed.
# If R gives the same answers as Python, the results don't depend on the tool.

df <- read.csv("synthetic_juvenile_cohort.csv")

# Set the reference groups, same as in Python
df$intervention <- relevel(factor(df$intervention), ref = "Standard Supervision")
df$risk_tier <- factor(df$risk_tier, levels = c("Low", "Moderate", "High"))

# Standardize risk score (mean 0, SD 1)
df$risk_z <- as.numeric(scale(df$risk_score))

# Helper: turn a model into an odds ratio table
odds_ratio_table <- function(model) {
  ci <- confint.default(model)   # Wald intervals, same method as statsmodels
  out <- data.frame(
    odds_ratio = exp(coef(model)),
    ci_low     = exp(ci[, 1]),
    ci_high    = exp(ci[, 2]),
    p_value    = summary(model)$coefficients[, 4]
  )
  rownames(out) <- sub("intervention", "", rownames(out))
  round(out[-1, ], 3)            # drop the intercept row
}

# ---- Step 2: naive vs. adjusted ----------------------------------------------
model_a <- glm(recid_24m ~ intervention,
               family = binomial, data = df)
model_b <- glm(recid_24m ~ intervention + prior_record + risk_z + age,
               family = binomial, data = df)

cat("MODEL A: intervention only (naive)\n")
print(odds_ratio_table(model_a))
cat("\nMODEL B: adjusted for prior record, risk, and age\n")
print(odds_ratio_table(model_b))

# ---- Step 3: interaction likelihood-ratio tests ------------------------------
c_int  <- glm(recid_24m ~ intervention * prior_record + risk_z + age,
              family = binomial, data = df)
d_main <- glm(recid_24m ~ intervention + risk_tier + prior_record + age,
              family = binomial, data = df)
d_int  <- glm(recid_24m ~ intervention * risk_tier + prior_record + age,
              family = binomial, data = df)

cat("\nLikelihood-ratio test: program x offender status\n")
print(anova(model_b, c_int, test = "Chisq"))
cat("\nLikelihood-ratio test: program x risk tier\n")
print(anova(d_main, d_int, test = "Chisq"))
