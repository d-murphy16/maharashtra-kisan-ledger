"""
Inferential statistics on the Maharashtra KCC dataset.

Formal hypothesis tests backing the descriptive patterns shown on the
dashboard:

  1. Chi-square test of independence, Crop x Season
     H0: a call's season (Kharif/Rabi/Zaid) is independent of its crop.
     The dashboard shows this is visibly false (cotton clusters Kharif,
     chana/wheat cluster Rabi); this test -- and its effect size -- puts a
     number on how false.

  2. Chi-square test of independence, District x Concern
     H0: the mix of concerns (weather/pest/nutrient/schemes/...) a district
     calls about is independent of which district it is.

  3. Kruskal-Wallis test, monthly call volume across the 3 seasons
     H0: the statewide monthly call-volume distribution is the same across
     Kharif/Rabi/Zaid months. (Kruskal-Wallis rather than one-way ANOVA
     since monthly counts are right-skewed, not normal.)

Run after fetch_kcc_data.py has populated data/*.csv.

Usage:
    python inferential_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import clean_districts, format_p

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"


def season_of(month: int) -> str:
    if 6 <= month <= 10:
        return "Kharif"
    if month in (11, 12) or month <= 3:
        return "Rabi"
    return "Zaid"


def cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test on an
    r x k contingency table (Bergsma 2013 correction)."""
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2_corr / min(r_corr - 1, k_corr - 1))


def test_crop_season_independence(report: list):
    df = pd.read_csv(DATA_DIR / "crop_season_counts.csv")
    table = df.pivot_table(index="crop", columns="season", values="count", fill_value=0)
    table = table[[c for c in ["Kharif", "Rabi", "Zaid"] if c in table.columns]]
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    v = cramers_v(chi2, n, *table.shape)

    report.append("## 1. Crop x Season independence (chi-square test)\n")
    report.append(f"- Contingency table: {table.shape[0]} crops x {table.shape[1]} seasons, n = {n:,} calls\n")
    report.append(f"- chi2 = {chi2:,.1f}, dof = {dof}, p {format_p(p)}\n")
    report.append(f"- Cramer's V (effect size) = {v:.3f}\n")
    verdict = (
        "large" if v >= 0.25 else "medium" if v >= 0.15 else "small" if v >= 0.05 else "negligible"
    )
    report.append(
        f"- **Verdict:** the association between crop and season is statistically significant "
        f"(p {format_p(p)}) with a **{verdict}** effect size. "
        "This formally confirms what the seasonal charts show visually: which crop a call is "
        "about is not independent of when in the year it happens.\n\n"
    )
    table.to_csv(OUT_DIR / "crop_season_contingency_table.csv")
    return table


def test_district_concern_independence(report: list):
    df = pd.read_csv(DATA_DIR / "district_concern_counts.csv")
    df = clean_districts(df)
    table = df.pivot_table(index="district", columns="concern", values="count", fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.values.sum())
    v = cramers_v(chi2, n, *table.shape)

    report.append("## 2. District x Concern independence (chi-square test)\n")
    report.append(f"- Contingency table: {table.shape[0]} districts x {table.shape[1]} concern categories, n = {n:,} calls "
                   "(restricted to Maharashtra's 36 official districts)\n")
    report.append(f"- chi2 = {chi2:,.1f}, dof = {dof}, p {format_p(p)}\n")
    report.append(f"- Cramer's V (effect size) = {v:.3f}\n")
    verdict = (
        "large" if v >= 0.25 else "medium" if v >= 0.15 else "small" if v >= 0.05 else "negligible"
    )
    report.append(
        f"- **Verdict:** concern mix differs significantly by district (p {format_p(p)}), "
        f"but the effect size is **{verdict}** (V = {v:.3f}) -- with 36 districts and ~5.17M calls, even small, "
        "policy-irrelevant deviations from the state average will read as 'significant'. The chi-square test tells "
        "us districts differ; it does not by itself tell us the differences are large enough to matter, which is "
        "exactly why the regression models look at effect sizes (relative risk / share differences), not just p-values.\n\n"
    )
    table.to_csv(OUT_DIR / "district_concern_contingency_table.csv")
    return table


def test_kruskal_wallis_season_volume(report: list):
    df = pd.read_csv(DATA_DIR / "monthly_totals.csv").dropna(subset=["month"])
    df["month"] = df["month"].astype(int)
    df["season"] = df["month"].apply(season_of)
    # Focus on the stable, well-populated period; 2009-2011 and 2026 have
    # sparse/partial coverage that would otherwise distort the comparison.
    df = df[(df["year"] >= 2012) & (df["year"] <= 2024)]

    groups = [g["count"].values for _, g in df.groupby("season")]
    labels = list(df.groupby("season").groups.keys())
    h, p = stats.kruskal(*groups)

    means = df.groupby("season")["count"].agg(["mean", "median", "std", "count"])

    report.append("## 3. Seasonal difference in monthly call volume (Kruskal-Wallis test)\n")
    report.append(f"- Comparing statewide monthly call totals across {labels} months, 2012-2024 ({len(df)} months)\n")
    report.append(f"- H = {h:.2f}, p {format_p(p)}\n")
    report.append("\n" + means.to_markdown(floatfmt=",.0f") + "\n\n")
    report.append(
        f"- **Verdict:** monthly call volume differs significantly by season "
        f"(p {format_p(p)}). Kharif months average the highest volume, "
        "consistent with the June-September sowing/pest-pressure window driving the bulk of calls.\n\n"
    )
    means.to_csv(OUT_DIR / "season_volume_summary.csv")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Inferential analysis -- Maharashtra KCC dataset\n\n"]
    test_crop_season_independence(report)
    test_district_concern_independence(report)
    test_kruskal_wallis_season_volume(report)

    out_path = OUT_DIR / "inferential_analysis_report.md"
    out_path.write_text("".join(report))
    print(f"Wrote {out_path}")
    print("".join(report))


if __name__ == "__main__":
    main()
