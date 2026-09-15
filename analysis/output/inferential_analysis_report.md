# Inferential analysis -- Maharashtra KCC dataset

## 1. Crop x Season independence (chi-square test)
- Contingency table: 19 crops x 3 seasons, n = 2,596,002 calls
- chi2 = 562,278.7, dof = 36, p < 1e-300
- Cramer's V (effect size) = 0.329
- **Verdict:** the association between crop and season is statistically significant (p < 1e-300) with a **large** effect size. This formally confirms what the seasonal charts show visually: which crop a call is about is not independent of when in the year it happens.

## 2. District x Concern independence (chi-square test)
- Contingency table: 36 districts x 11 concern categories, n = 5,157,249 calls (restricted to Maharashtra's 36 official districts)
- chi2 = 174,112.6, dof = 350, p < 1e-300
- Cramer's V (effect size) = 0.058
- **Verdict:** concern mix differs significantly by district (p < 1e-300), but the effect size is **small** (V = 0.058) -- with 36 districts and ~5.17M calls, even small, policy-irrelevant deviations from the state average will read as 'significant'. The chi-square test tells us districts differ; it does not by itself tell us the differences are large enough to matter, which is exactly why the regression models look at effect sizes (relative risk / share differences), not just p-values.

## 3. Seasonal difference in monthly call volume (Kruskal-Wallis test)
- Comparing statewide monthly call totals across ['Kharif', 'Rabi', 'Zaid'] months, 2012-2024 (153 months)
- H = 6.39, p = 0.0411

| season   |   mean |   median |    std |   count |
|:---------|-------:|---------:|-------:|--------:|
| Kharif   | 36,786 |   32,848 | 22,584 |      64 |
| Rabi     | 29,864 |   26,763 | 17,318 |      65 |
| Zaid     | 25,354 |   20,506 | 20,148 |      24 |

- **Verdict:** monthly call volume differs significantly by season (p = 0.0411). Kharif months average the highest volume, consistent with the June-September sowing/pest-pressure window driving the bulk of calls.

