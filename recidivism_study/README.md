# Juvenile Recidivism Predictor Study

A data science project modeling risk factors and intervention outcomes for justice-involved male adolescents. I built a synthetic dataset, tested which interventions are associated with lower re-offending across risk profiles, and built an intake risk model that reports probabilities.

All data in this project is simulated. No real youth are represented.

## Why synthetic data

Real juvenile justice records require a data use agreement and IRB approval. I used synthetic data to learn and test the full analysis pipeline first. Because I set the true effects when generating the data, I could check whether each method recovered them. The dataset design draws on published juvenile justice research: cognitive behavioral therapy tends to reduce re-offending, and deterrence-style programs tend to show no benefit or cause harm.

## Tools

- **Python:** pandas, NumPy, statsmodels, scikit-learn, Seaborn, Matplotlib
- **R:** base R (`glm`) to replicate the main regressions

## Files

| File | What it does |
|---|---|
| `generate_data.py` | Builds 3,000 synthetic youth records |
| `step2_regression.py` | Naive and adjusted logistic regression |
| `step3_interactions.py` | Tests whether program effects differ by offender status and risk tier |
| `step4_visualize.py` | Four-panel figure comparing intervention efficacy |
| `step5_predict.py` | Intake risk model with train/test evaluation |
| `step6_analysis.R` | R replication of steps 2 and 3 |
| `intervention_efficacy.png` | Output figure |

Run the Python scripts in order from the project folder, starting with `python generate_data.py`.

## Data

Each record includes age (13 to 17), offender status (first-time or prior record), school attendance, family support (1 to 10), a composite risk score and tier, one of four programs, and re-offense at 12 and 24 months.

Programs: Standard Supervision (the comparison group), Cognitive Behavioral Therapy, Vocational Training, and Tough Love.

Program assignment is not random. Higher-risk youth were more likely to be placed in Tough Love, and older youth in Vocational Training. I built this in on purpose to mirror how placements happen in practice.

## Methods and findings

**1. Naive vs. adjusted regression.** Tough Love looked harmful in raw comparisons (odds ratio 1.38, p = .002). After controlling for prior record, risk score, and age, the effect dropped to 1.12 and was no longer significant (p = .33). Much of the apparent harm came from which youth were placed there. CBT (OR 0.35) and Vocational Training (OR 0.59) were associated with lower re-offending after adjustment. The model recovered the effects I built in for prior record, risk, and age.

**2. Interactions.** I tested program × offender status and program × risk tier, chosen in advance based on the risk principle and research on deterrence programs. I fit them as separate models because offender status is part of the risk score. With 3,000 youth, neither was significant, although both effects exist in the data. When I increased the sample to 9,000, the risk tier interaction became significant (p = .02) and offender status did not (p = .24). The risk tier effect is larger, so it needed less data to detect. The non-significant results at 3,000 came from low statistical power.

**3. Visualization.** A four-panel figure shows raw rates, a forest plot of adjusted odds ratios on a log scale, and two interaction plots with bootstrap confidence intervals. CBT's line is the flattest across risk tiers: its benefit is largest for high-risk youth.

**4. Risk prediction.** A scikit-learn logistic regression using intake information only (age, prior record, school attendance, family support) reached an AUC of 0.69 on held-out test data. I chose not to set a cutoff. The tool reports each youth's probability and leaves the decision to a person. I evaluated it on calibration instead: predictions matched actual rates within a few points for most youth and for both offender groups, but the highest-risk group was overestimated by about 7 points (57% predicted, 50% actual).

**5. Replication.** R produced identical odds ratios and test results.

## Limitations

### Data bias
- The effects were set by me. The analysis can show that the methods work. It cannot show that CBT or any other program works for real youth.
- The confounding in program assignment is simple and fully measured. Real placement decisions depend on judges, counties, available beds, and family advocacy, much of which is never recorded.
- The dataset has no race, ethnicity, neighborhood, or income variables. Real risk tools have been found to carry uneven error rates across racial groups. This project checks fairness only across offender status and cannot speak to that problem.
- All youth are male, and the findings would not transfer to girls in the system.

### Reporting gaps
- Recidivism here is measured without error. In real data it is usually measured by re-arrest or re-adjudication, which reflects how heavily an area is policed as well as how youth behave.
- There is no missing data and no loss to follow-up. Real studies lose youth who move, age out, or transfer to adult court.
- Programs are treated as one thing each. The data does not capture program quality, staff training, dosage, or whether a youth completed the program.
- "Tough Love" has no standard definition. Real programs under that label vary widely.

### What this analysis can't tell you
- It cannot establish cause. Even with controls, an observational design can't rule out factors that weren't measured.
- It cannot explain why a program works or which parts of it matter.
- The risk model ranks youth only moderately well. A 0.69 AUC means many individual predictions will be wrong, and the tool overstates risk for the youth most likely to be acted on.
- Reporting probabilities without a cutoff does not remove the tradeoff between wrongly flagging youth and missing those who re-offend. It moves that decision to the person reading the score, where it goes unmeasured.

## Next steps

Apply the same pipeline to real data, such as the Pathways to Desistance study (ICPSR), which followed 1,354 adjudicated youth for seven years. This would require restricted-data access through a faculty sponsor and IRB approval.
