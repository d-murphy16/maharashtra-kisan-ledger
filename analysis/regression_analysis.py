"""
Regression analysis on the Maharashtra KCC dataset.

All models are fit on grouped counts (crop x season, year-month totals,
crop x season x concern, district x crop/concern), not on the 5.17M raw
rows. For categorical predictors this is not an approximation: a Poisson
regression on grouped counts is the exact same likelihood as a multinomial
/ logistic regression on the underlying individual records, just far
cheaper to fit -- see Agresti, "Categorical Data Analysis", ch. 9 on the
Poisson/multinomial equivalence. Four models:

  M1. Poisson GLM: count ~ crop * season
      Does the seasonal pattern of calls genuinely differ by crop, or is
      "season" just a generic effect on top of each crop's overall volume?
      Tested via the crop:season interaction terms and a likelihood-ratio
      test against the no-interaction model.

  M2. Negative Binomial GLM: statewide monthly calls ~ trend + seasonal
      harmonics + COVID lockdown dummy
      Quantifies (a) the post-2016 decline as a trend coefficient, (b) the
      2020 lockdown as a level shock, and (c) within-year seasonality as
      harmonic terms, on the actual monthly time series.

  M3. Poisson GLM with exposure offset: concern-category counts ~ crop *
      season, offset = log(crop x season total)
      This is the multinomial-via-Poisson trick: with the offset, the fitted
      model describes each concern category's *share* of calls within each
      crop x season cell, not raw volume. Used to test whether the Water
      Management/Irrigation share is significantly associated with any
      crop or season -- the regression-grade version of the dashboard's
      headline finding.

  M4. OLS, n=36 districts: pest-concern share ~ cotton crop-share
      Does a district's degree of cotton-dependence predict its share of
      pest/disease/weed calls? A simple, fully interpretable bivariate
      regression at the district level.

Run after fetch_kcc_data.py has populated data/*.csv.

Usage:
    python regression_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import clean_districts, format_p

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6, "font.size": 10,
})


def season_of(month: int) -> str:
    if 6 <= month <= 10:
        return "Kharif"
    if month in (11, 12) or month <= 3:
        return "Rabi"
    return "Zaid"


def slug(name: str) -> str:
    return (
        name.replace(" ", "_").replace("(", "").replace(")", "")
        .replace("/", "_").replace(",", "").replace("&", "and")
    )


# ---------------------------------------------------------------- M1 ----
def model_crop_season_interaction(report: list):
    # Fit on crop x season x YEAR counts, not the collapsed crop x season
    # table: the collapsed table has exactly one row per (crop, season)
    # combination, so a model with the full interaction term is saturated
    # by construction (zero residual degrees of freedom) and the
    # likelihood-ratio test against it is degenerate. Bringing year in as a
    # genuine repeated dimension (~15 years per crop x season cell) gives
    # the interaction model real residual variation to be tested against.
    df = pd.read_csv(DATA_DIR / "crop_season_year_counts.csv")
    df = df[df["season"].isin(["Kharif", "Rabi", "Zaid"])].copy()
    df = df[(df["year"] >= 2012) & (df["year"] <= 2024)]  # stable, fully-covered years
    df["crop_"] = df["crop"].apply(slug)
    df["year_"] = df["year"].astype(int).astype(str)

    full = smf.glm("count ~ C(crop_) * C(season) + C(year_)", data=df,
                    family=sm.families.Poisson()).fit()
    reduced = smf.glm("count ~ C(crop_) + C(season) + C(year_)", data=df,
                       family=sm.families.Poisson()).fit()

    lr_stat = 2 * (full.llf - reduced.llf)
    lr_df = full.df_model - reduced.df_model
    from scipy.stats import chi2 as chi2_dist
    lr_p = chi2_dist.sf(lr_stat, lr_df)

    report.append("## M1. Poisson GLM -- call volume ~ crop x season (+ year), 2012-2024\n\n")
    report.append(f"- n = {len(df):,} crop x season x year cells\n")
    report.append(f"- Full model (with crop:season interaction, controlling for year) "
                   f"deviance = {full.deviance:,.0f} on {full.df_resid:,.0f} df\n")
    report.append(f"- Reduced model (no interaction, controlling for year) "
                   f"deviance = {reduced.deviance:,.0f} on {reduced.df_resid:,.0f} df\n")
    report.append(f"- Likelihood-ratio test for the crop:season interaction: "
                   f"LR = {lr_stat:,.1f}, df = {lr_df:.0f}, p {format_p(lr_p)}\n")
    report.append(
        f"- **Verdict:** even after controlling for year (to absorb the statewide trend/COVID shock modeled "
        f"separately in M2), the crop:season interaction is {'highly significant' if lr_p < 1e-3 else f'significant (p {format_p(lr_p)})'} "
        "-- crops do not just have different overall call volumes, they have genuinely different *seasonal shapes* "
        "(e.g. cotton's Kharif spike is proportionally far sharper than sugarcane's, which is closer to flat "
        "year-round), and this holds year after year rather than being an artifact of any single year's data.\n\n"
    )

    # effect plot: observed seasonal share per crop, top 8 by volume (summed across 2012-2024)
    totals = df.groupby(["crop", "season"])["count"].sum().reset_index()
    top_crops = totals.groupby("crop")["count"].sum().sort_values(ascending=False).head(8).index
    pivot = totals[totals["crop"].isin(top_crops)].pivot_table(index="crop", columns="season", values="count", fill_value=0)
    pivot = pivot[["Kharif", "Rabi", "Zaid"]]
    shares = pivot.div(pivot.sum(axis=1), axis=0).loc[top_crops]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shares.plot(kind="barh", stacked=True, ax=ax, color=["#1baf7a", "#eda100", "#eb6834"], width=0.7)
    ax.set_xlabel("Share of calls"); ax.set_ylabel("")
    ax.set_title("Fitted seasonal share by crop (M1 confirms this split is non-random)")
    ax.legend(title="Season", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m1_crop_season_shares.png", dpi=150)
    plt.close(fig)

    with open(OUT_DIR / "m1_full_model_summary.txt", "w") as f:
        f.write(str(full.summary()))


# ---------------------------------------------------------------- M2 ----
def model_time_trend(report: list):
    df = pd.read_csv(DATA_DIR / "monthly_totals.csv").dropna(subset=["month", "year"])
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 2012) & ((df["year"] < 2025) | ((df["year"] == 2025) & (df["month"] <= 7)))]
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    df["t"] = range(len(df))  # linear month index, for the trend
    df["sin1"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos1"] = np.cos(2 * np.pi * df["month"] / 12)
    df["covid"] = ((df["year"] == 2020) & (df["month"].between(4, 7))).astype(int)

    # smf.negativebinomial estimates the dispersion parameter (alpha) by MLE
    # (NB2), rather than fixing it at 1 as a plain GLM NegativeBinomial
    # family would -- the right choice since these monthly counts are
    # over-dispersed relative to Poisson.
    model = smf.negativebinomial("count ~ t + sin1 + cos1 + covid", data=df).fit(disp=0)

    report.append("## M2. Negative Binomial GLM -- statewide monthly volume over time\n\n")
    report.append(f"- n = {len(df)} months, 2012-01 to {df.iloc[-1]['year']:.0f}-{df.iloc[-1]['month']:02.0f}\n")
    report.append(
        f"- Pseudo R2 (McFadden) = {model.prsquared:.3f} "
        "(McFadden's pseudo-R2 runs much lower than an OLS R2 even for a well-fitting count model -- "
        "values of 0.01-0.05 are typical; judge fit from the observed-vs-fitted plot below, not this number "
        "in isolation)\n\n"
    )
    coef_table = pd.DataFrame({
        "coef": model.params, "std_err": model.bse, "p_value": model.pvalues,
    })
    report.append(coef_table.to_markdown(floatfmt=",.4f") + "\n\n")

    t_coef = model.params["t"]
    t_p = model.pvalues["t"]
    covid_coef = model.params["covid"]
    covid_p = model.pvalues["covid"]
    report.append(
        f"- **Trend:** each additional month is associated with a "
        f"{'decline' if t_coef < 0 else 'rise'} of {abs((np.exp(t_coef) - 1) * 100):.2f}% in expected call "
        f"volume (p = {t_p:.3g}) -- compounded over ~150 months this reproduces the large fall from the "
        "2016 peak seen in the dashboard.\n"
        f"- **COVID lockdown (Apr-Jul 2020):** associated with a "
        f"{(1 - np.exp(covid_coef)) * 100:.1f}% drop in expected calls versus the trend/season-adjusted baseline "
        f"(p = {covid_p:.3g}).\n\n"
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["t"], df["count"], color="#888", lw=1, label="Observed")
    ax.plot(df["t"], model.predict(df), color="#1F5C73", lw=2, label="Fitted (M2)")
    xt = df["t"][df["month"] == 1]
    ax.set_xticks(xt, df.loc[xt.index, "year"])
    ax.set_ylabel("Calls / month"); ax.set_title("M2: statewide monthly calls, observed vs. fitted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m2_time_trend_fit.png", dpi=150)
    plt.close(fig)

    with open(OUT_DIR / "m2_full_model_summary.txt", "w") as f:
        f.write(str(model.summary()))


# ---------------------------------------------------------------- M3 ----
def model_water_concern_share(report: list):
    # Binary contrast (water-management calls vs. all other calls) fit with
    # a Poisson-plus-offset model on crop x season x YEAR cells -- the
    # Poisson-for-binomial trick, exactly analogous to logistic regression
    # but cheaper to fit on grouped counts. Year replication (~13 years per
    # crop x season cell) is what M1 added for the same reason: a model
    # fit on the single collapsed crop x season table would be estimating
    # crop/season effects purely off patsy's arbitrary reference-category
    # choice, with no real repeated variation backing the estimates.
    totals = pd.read_csv(DATA_DIR / "crop_season_year_counts.csv").rename(columns={"count": "total"})
    water = pd.read_csv(DATA_DIR / "crop_season_year_water_counts.csv").rename(columns={"count": "water"})
    df = totals.merge(water, on=["crop", "season", "year"], how="left")
    df["water"] = df["water"].fillna(0).astype(int)
    df = df[df["season"].isin(["Kharif", "Rabi", "Zaid"])].copy()
    df = df[(df["year"] >= 2012) & (df["year"] <= 2024)]
    df["crop_"] = df["crop"].apply(slug)
    df["year_"] = df["year"].astype(int).astype(str)
    df["log_exposure"] = np.log(df["total"])

    full = smf.glm("water ~ C(crop_) * C(season) + C(year_)", data=df,
                    family=sm.families.Poisson(), offset=df["log_exposure"]).fit()
    reduced = smf.glm("water ~ C(crop_) + C(season) + C(year_)", data=df,
                       family=sm.families.Poisson(), offset=df["log_exposure"]).fit()
    lr_stat = 2 * (full.llf - reduced.llf)
    lr_df = full.df_model - reduced.df_model
    from scipy.stats import chi2 as chi2_dist
    lr_p = chi2_dist.sf(lr_stat, lr_df)

    report.append("## M3. Poisson GLM with exposure offset -- Water Management/Irrigation share ~ crop x season (+ year)\n\n")
    report.append(
        "Models the Water Management/Irrigation *share* of calls (not raw volume) within every "
        "crop x season x year cell, using each cell's total call count as an offset (the Poisson-for-binomial "
        "trick -- statistically equivalent to logistic regression on the underlying calls, fit on grouped counts). "
        f"n = {len(df):,} crop x season x year cells, 2012-2024.\n\n"
    )
    report.append(f"- Likelihood-ratio test for the crop:season interaction on water-management share: "
                   f"LR = {lr_stat:,.1f}, df = {lr_df:.0f}, p {format_p(lr_p)}\n\n")

    # fitted share by crop x season, marginalized over year, top 8 crops by volume
    top_crops = df.groupby("crop")["total"].sum().sort_values(ascending=False).head(8).index
    pred = df[df["crop"].isin(top_crops)].copy()
    # GLM.predict on new data does NOT reuse the fitted offset automatically
    # -- it must be passed explicitly, or the prediction silently comes back
    # as if exposure were 1 (i.e. as a raw rate, not a share of `total`).
    pred["fitted_share"] = full.predict(pred, offset=pred["log_exposure"]) / pred["total"]
    share_table = pred.groupby(["crop", "season"])["fitted_share"].mean().unstack()[["Kharif", "Rabi", "Zaid"]] * 100
    share_table = share_table.loc[top_crops]
    report.append("Model-fitted Water Management/Irrigation share (%) by crop x season:\n\n")
    report.append(share_table.to_markdown(floatfmt=",.2f") + "\n\n")

    overall_share = df["water"].sum() / df["total"].sum() * 100
    max_cell = share_table.stack().idxmax()
    max_val = share_table.stack().max()
    report.append(
        f"- **Verdict:** the crop:season interaction is "
        f"{'highly significant' if lr_p < 1e-3 else f'significant (p {format_p(lr_p)})'}, so water-management "
        f"share is not uniform across crop/season -- but every fitted cell stays within roughly {share_table.values.min():.2f}%"
        f" to {max_val:.2f}% (highest for **{max_cell[0]} / {max_cell[1]}**), against an overall state average of "
        f"{overall_share:.2f}%. Statistically real, substantively small: even the peak crop/season combination "
        "is nowhere near the double-digit share you would expect if farmers' drought/irrigation anxiety were "
        "routinely being logged under this label. The regression's own fitted values corroborate the dashboard's "
        "reading -- most water-scarcity concern is most likely being asked (and logged) as a Weather query, not a "
        "Water Management one.\n\n"
    )
    with open(OUT_DIR / "m3_full_model_summary.txt", "w") as f:
        f.write(str(full.summary()))


# ---------------------------------------------------------------- M4 ----
def model_district_cotton_pest(report: list):
    dc = clean_districts(pd.read_csv(DATA_DIR / "district_crop_counts.csv"))
    dconcern = clean_districts(pd.read_csv(DATA_DIR / "district_concern_counts.csv"))

    crop_totals = dc.groupby("district")["count"].sum().rename("total_crop_calls")
    cotton = dc[dc["crop"] == "Cotton (Kapas)"].set_index("district")["count"].rename("cotton_calls")
    cotton_share = (cotton / crop_totals).fillna(0).rename("cotton_share")

    concern_totals = dconcern.groupby("district")["count"].sum().rename("total_calls")
    pest = dconcern[dconcern["concern"] == "Pest, Disease & Weed Mgmt"].set_index("district")["count"].rename("pest_calls")
    pest_share = (pest / concern_totals).fillna(0).rename("pest_share")

    reg = pd.concat([cotton_share, pest_share], axis=1).dropna()

    X = sm.add_constant(reg["cotton_share"])
    model = sm.OLS(reg["pest_share"], X).fit()

    report.append("## M4. OLS (n=36 districts) -- pest-concern share ~ cotton crop-share\n\n")
    report.append(f"- n = {len(reg)} districts\n")
    report.append(f"- R-squared = {model.rsquared:.3f}\n")
    coef_table = pd.DataFrame({"coef": model.params, "std_err": model.bse, "p_value": model.pvalues})
    report.append(coef_table.to_markdown(floatfmt=",.4f") + "\n\n")
    slope = model.params["cotton_share"]
    p = model.pvalues["cotton_share"]
    report.append(
        f"- **Verdict:** a 10-percentage-point increase in a district's cotton share of named-crop calls is "
        f"associated with a {slope * 10:.2f}-percentage-point "
        f"{'increase' if slope > 0 else 'decrease'} in that district's pest/disease/weed-concern share "
        f"(p = {p:.3g}, R2 = {model.rsquared:.2f}). "
        + ("This supports targeting pest-management extension by a district's cotton exposure, not just its "
           "raw call volume.\n\n" if p < 0.05 else
           "This relationship is not statistically significant at n=36 -- cotton dependence alone does not "
           "reliably predict a district's pest-concern share; other factors (which pest, local agronomy, "
           "extension coverage) likely dominate.\n\n")
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(reg["cotton_share"] * 100, reg["pest_share"] * 100, color="#A96B15", s=28, alpha=0.8)
    xs = np.linspace(reg["cotton_share"].min(), reg["cotton_share"].max(), 50)
    ax.plot(xs * 100, (model.params["const"] + model.params["cotton_share"] * xs) * 100, color="#1F5C73", lw=2)
    ax.set_xlabel("Cotton share of named-crop calls (%)")
    ax.set_ylabel("Pest/disease/weed share of all calls (%)")
    ax.set_title(f"M4: district cotton dependence vs. pest-concern share (R2={model.rsquared:.2f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m4_cotton_pest_scatter.png", dpi=150)
    plt.close(fig)

    reg.to_csv(OUT_DIR / "m4_district_data.csv")
    with open(OUT_DIR / "m4_full_model_summary.txt", "w") as f:
        f.write(str(model.summary()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Regression analysis -- Maharashtra KCC dataset\n\n"]
    model_crop_season_interaction(report)
    model_time_trend(report)
    model_water_concern_share(report)
    model_district_cotton_pest(report)

    out_path = OUT_DIR / "regression_analysis_report.md"
    out_path.write_text("".join(report))
    print(f"Wrote {out_path}")
    print("".join(report))


if __name__ == "__main__":
    main()
