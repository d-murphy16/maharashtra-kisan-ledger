# Regression analysis -- Maharashtra KCC dataset

## M1. Poisson GLM -- call volume ~ crop x season (+ year), 2012-2024

- n = 710 crop x season x year cells
- Full model (with crop:season interaction, controlling for year) deviance = 435,869 on 641 df
- Reduced model (no interaction, controlling for year) deviance = 968,740 on 677 df
- Likelihood-ratio test for the crop:season interaction: LR = 532,870.1, df = 36, p < 1e-300
- **Verdict:** even after controlling for year (to absorb the statewide trend/COVID shock modeled separately in M2), the crop:season interaction is highly significant -- crops do not just have different overall call volumes, they have genuinely different *seasonal shapes* (e.g. cotton's Kharif spike is proportionally far sharper than sugarcane's, which is closer to flat year-round), and this holds year after year rather than being an artifact of any single year's data.

## M2. Negative Binomial GLM -- statewide monthly volume over time

- n = 160 months, 2012-01 to 2025-07
- Pseudo R2 (McFadden) = 0.012 (McFadden's pseudo-R2 runs much lower than an OLS R2 even for a well-fitting count model -- values of 0.01-0.05 are typical; judge fit from the observed-vs-fitted plot below, not this number in isolation)

|           |    coef |   std_err |   p_value |
|:----------|--------:|----------:|----------:|
| Intercept | 10.8806 |    0.1312 |    0.0000 |
| t         | -0.0070 |    0.0015 |    0.0000 |
| sin1      | -0.2170 |    0.0858 |    0.0114 |
| cos1      | -0.0623 |    0.0850 |    0.4639 |
| covid     | -3.7766 |    0.5462 |    0.0000 |
| alpha     |  0.5718 |    0.0593 |    0.0000 |

- **Trend:** each additional month is associated with a decline of 0.70% in expected call volume (p = 1.67e-06) -- compounded over ~150 months this reproduces the large fall from the 2016 peak seen in the dashboard.
- **COVID lockdown (Apr-Jul 2020):** associated with a 97.7% drop in expected calls versus the trend/season-adjusted baseline (p = 4.72e-12).

## M3. Poisson GLM with exposure offset -- Water Management/Irrigation share ~ crop x season (+ year)

Models the Water Management/Irrigation *share* of calls (not raw volume) within every crop x season x year cell, using each cell's total call count as an offset (the Poisson-for-binomial trick -- statistically equivalent to logistic regression on the underlying calls, fit on grouped counts). n = 710 crop x season x year cells, 2012-2024.

- Likelihood-ratio test for the crop:season interaction on water-management share: LR = 2,062.3, df = 36, p < 1e-300

Model-fitted Water Management/Irrigation share (%) by crop x season:

| crop                                      |   Kharif |   Rabi |   Zaid |
|:------------------------------------------|---------:|-------:|-------:|
| Other Named Crops                         |     0.61 |   1.15 |   1.41 |
| Cotton (Kapas)                            |     0.78 |   0.44 |   0.25 |
| Onion                                     |     1.09 |   0.73 |   0.64 |
| Soybean (bhat)                            |     0.66 |   0.19 |   0.10 |
| Sugarcane (Noble Cane)                    |     0.88 |   1.17 |   2.00 |
| Bengal Gram (Gram/Chick Pea/Kabuli/Chana) |     0.55 |   4.07 |   0.09 |
| Pigeon pea (red gram/arhar/tur)           |     0.79 |   1.51 |   0.12 |
| Chillies                                  |     0.17 |   0.16 |   0.48 |

- **Verdict:** the crop:season interaction is highly significant, so water-management share is not uniform across crop/season -- but every fitted cell stays within roughly 0.09% to 4.07% (highest for **Bengal Gram (Gram/Chick Pea/Kabuli/Chana) / Rabi**), against an overall state average of 0.75%. Statistically real, substantively small: even the peak crop/season combination is nowhere near the double-digit share you would expect if farmers' drought/irrigation anxiety were routinely being logged under this label. The regression's own fitted values corroborate the dashboard's reading -- most water-scarcity concern is most likely being asked (and logged) as a Weather query, not a Water Management one.

## M4. OLS (n=36 districts) -- pest-concern share ~ cotton crop-share

- n = 36 districts
- R-squared = 0.082
|              |   coef |   std_err |   p_value |
|:-------------|-------:|----------:|----------:|
| const        | 0.1864 |    0.0081 |    0.0000 |
| cotton_share | 0.0724 |    0.0414 |    0.0896 |

- **Verdict:** a 10-percentage-point increase in a district's cotton share of named-crop calls is associated with a 0.72-percentage-point increase in that district's pest/disease/weed-concern share (p = 0.0896, R2 = 0.08). This relationship is not statistically significant at n=36 -- cotton dependence alone does not reliably predict a district's pest-concern share; other factors (which pest, local agronomy, extension coverage) likely dominate.

