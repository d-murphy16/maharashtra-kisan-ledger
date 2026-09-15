# Analysis pipeline

Python code behind [The Maharashtra Kisan Ledger](../README.md) dashboard, plus
the inferential and regression analysis that goes beyond what the dashboard
shows.

## Pipeline

```bash
pip install -r requirements.txt

python fetch_kcc_data.py          # ~5-10 min: full census, 5.17M rows -> data/*.csv
python inferential_analysis.py    # chi-square / Kruskal-Wallis tests -> output/inferential_analysis_report.md
python regression_analysis.py     # regression models + plots        -> output/regression_analysis_report.md
```

`fetch_kcc_data.py` is the only script that talks to the network. It reads
every Maharashtra record from the live [Kisan Call Centre API](https://api.data.gov.in)
in 100,000-row pages (a full census, not a sample) and immediately folds each
page into a handful of small count tables — it never holds the raw 5.17M rows
in memory or on disk. Those aggregate tables (`data/*.csv`, a few hundred KB
total) are what the two analysis scripts actually run on, and are checked
into this repo so the regressions can be re-run offline without hitting the
API again.

## What the two analysis scripts test

**`inferential_analysis.py`** — hypothesis tests behind the dashboard's
descriptive charts:

| Test | Question |
|---|---|
| Chi-square, Crop × Season | Is a call's season independent of its crop? |
| Chi-square, District × Concern | Does concern mix differ by district? |
| Kruskal-Wallis | Does monthly call volume differ across Kharif/Rabi/Zaid? |

**`regression_analysis.py`** — four models, fit on grouped counts via Poisson
/ Negative Binomial / OLS regression (see the module docstring for why
grouped-count Poisson regression is statistically equivalent to row-level
logistic/multinomial regression for categorical predictors, and far cheaper
to fit on 5M+ rows):

| Model | Question |
|---|---|
| M1 — Poisson, crop × season interaction | Does each crop have its *own* seasonal shape, or just a shared one? |
| M2 — Negative Binomial, monthly time series | How large is the post-2016 decline, and the 2020 lockdown shock? |
| M3 — Poisson w/ exposure offset | Does the Water Management/Irrigation *share* of calls vary by crop/season, or is it uniformly rare? |
| M4 — OLS, n=36 districts | Does a district's cotton-dependence predict its pest-concern share? |

Full results (coefficients, p-values, effect sizes) land in
`output/inferential_analysis_report.md` and `output/regression_analysis_report.md`,
with supporting plots as PNGs in `output/`.

## Notes

- All concern-category and crop-bucketing logic mirrors the mapping used to
  build the dashboard (see `fetch_kcc_data.py` docstrings), so results here
  are directly comparable to what's shown there.
- These models describe association within the KCC dataset itself; there is
  no causal claim being made, and no external covariates (rainfall,
  population, credit access, etc.) are included since none are present in
  the source dataset. Read the regression coefficients as "controlling for
  the other terms in this model," not as causal effects.
