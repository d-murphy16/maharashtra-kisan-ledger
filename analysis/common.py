"""Shared helpers for inferential_analysis.py and regression_analysis.py."""
import pandas as pd

# The 36 official Maharashtra districts. Everything else appearing in the
# raw DistrictName field is a data-quality artifact: "0" and "9999" are the
# dataset's own missing-value placeholders, blank/NaN is an empty field, and
# a single record is mistakenly tagged "BHAVNAGAR" (a Gujarat district).
VALID_DISTRICTS = {
    "AHMADNAGAR", "AKOLA", "AMRAVATI", "AURANGABAD", "BEED", "BHANDARA",
    "BULDANA", "CHANDRAPUR", "DHULE", "GADCHIROLI", "GONDIYA", "HINGOLI",
    "JALGAON", "JALNA", "KOLHAPUR", "LATUR", "MUMBAI", "Mumbai Suburban",
    "NAGPUR", "NANDED", "NANDURBAR", "NASIK", "OSMANABAD", "PALGHAR",
    "PARBHANI", "PUNE", "RAIGARH", "RATNAGIRI", "SANGLI", "SATARA",
    "SINDHUDURG", "SOLAPUR", "THANE", "WARDHA", "WASHIM", "YEVATMAL",
}


def clean_districts(df: pd.DataFrame, col: str = "district") -> pd.DataFrame:
    """Drop rows whose district isn't one of the 36 official districts,
    and report how many records that excluded."""
    before = df["count"].sum()
    out = df[df[col].isin(VALID_DISTRICTS)].copy()
    after = out["count"].sum()
    dropped = before - after
    if dropped:
        print(f"  [clean_districts] dropped {dropped:,} of {before:,} records "
              f"({dropped / before:.3%}) with non-official district values")
    return out


def format_p(p: float) -> str:
    """scipy reports p=0.0 for anything that underflows float64 precision;
    report it as a bound, not a (misleading) literal zero."""
    if p == 0.0:
        return "< 1e-300"
    if p < 1e-3:
        return f"< 0.001 (exact: {p:.3g})"
    return f"= {p:.3g}"
