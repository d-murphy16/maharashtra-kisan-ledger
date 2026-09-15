"""
Fetch the full Maharashtra slice of the Kisan Call Centre (KCC) dataset from
the live data.gov.in API and reduce it to a set of small, analysis-ready
aggregate tables.

This performs a full census (no sampling): every one of the ~5.17M Maharashtra
records is read, in 100,000-row pages, and folded into running counters. Raw
rows are never held in memory or written to disk -- only the aggregates are
saved, to data/*.csv -- so this script is safe to re-run and the outputs stay
small enough to check into git.

Usage:
    python fetch_kcc_data.py

Takes several minutes (52 API pages of 100,000 rows each).
"""
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f"
API_KEY = "579b464db66ec23bdd000001cdc3b564546246a772a26393094f5645"
PAGE_SIZE = 100_000
STATE = "MAHARASHTRA"
FIELDS = "DistrictName,BlockName,Crop,QueryType,month,year"
DATA_DIR = Path(__file__).parent / "data"

# Top named crops tracked individually; every other named crop (i.e. not
# "Others"/"9999", the dataset's own non-crop-specific placeholders) folds
# into "Other Named Crops" so the crop dimension stays a manageable size for
# the contingency tables and regressions below.
TOP_CROPS = {
    "Cotton (Kapas)", "Onion", "Soybean (bhat)", "Sugarcane (Noble Cane)",
    "Bengal Gram (Gram/Chick Pea/Kabuli/Chana)", "Pigeon pea (red gram/arhar/tur)",
    "Chillies", "Pomegranate", "Brinjal", "Wheat", "Tomato",
    "Groundnut (pea nut/mung phalli)", "Maize (Makka)",
    "Bhindi(Okra/Ladysfinger)", "Turmeric", "Mango",
    "Sorghum (Jowar/Great Millet)", "Watermelon",
}


def map_concern(query_type: str) -> str:
    """Collapse the raw QueryType field (103 distinct values in the wild,
    including blank/numeric-code data-quality artifacts) into 8 concern
    buckets. Mirrors the mapping used to build the public dashboard, so the
    two stay directly comparable."""
    if not query_type:
        return "Unspecified"
    t = query_type.replace("\t", "").strip()
    if t == "" or t.isdigit() or t.lower() == "others":
        return "Unspecified"
    l = t.lower()
    if "weather" in l or "sowing time" in l:
        return "Weather & Climate Advisory"
    if "water management" in l or "irrigation" in l:
        return "Water Management/Irrigation"
    if "plant protection" in l or "weed management" in l or "bio-pesticide" in l:
        return "Pest, Disease & Weed Mgmt"
    if any(k in l for k in ("fertilizer", "nutrient", "soil testing", "soil health", "organic farming")):
        return "Nutrient & Soil Management"
    if any(k in l for k in ("government scheme", "crop insurance", "credit", "training")):
        return "Govt Schemes, Insurance & Policy"
    if "market information" in l:
        return "Market & Price Information"
    if any(k in l for k in ("varieties", "seed", "vegetative propagation", "tissue culture")):
        return "Seeds & Varieties"
    if any(k in l for k in ("cultural practices", "field preparation", "mechanization")):
        return "Cultural Practices & Mechanization"
    if "animal" in l:
        return "Animal Husbandry"
    return "Other Advisory"


def season_of(month: int) -> str:
    """Standard Indian cropping calendar, mapped from calendar month. The
    dataset's own Season field is populated for only ~35% of rows (mostly
    "NA"), so month-derived season is used throughout for full coverage."""
    if 6 <= month <= 10:
        return "Kharif"
    if month in (11, 12) or month <= 3:
        return "Rabi"
    if month in (4, 5):
        return "Zaid"
    return "Unknown"


def crop_bucket(crop: str):
    if not crop or crop in ("Others", "9999"):
        return None
    return crop if crop in TOP_CROPS else "Other Named Crops"


def fetch_page(offset: int, session: requests.Session, retries: int = 5):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": PAGE_SIZE,
        "offset": offset,
        "fields": FIELDS,
        "filters[StateName]": STATE,
    }
    for attempt in range(retries):
        try:
            r = session.get(API_BASE, params=params, timeout=120,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            wait = 3 * (attempt + 1)
            print(f"  retry {attempt + 1}/{retries} after error ({e}); waiting {wait}s", flush=True)
            time.sleep(wait)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Running counters for every aggregate table this analysis needs.
    crop_season = Counter()                 # (crop, season) -> count
    crop_season_concern = Counter()         # (crop, season, concern) -> count
    crop_season_year = Counter()            # (crop, season, year) -> count
    crop_season_year_water = Counter()      # (crop, season, year) -> water-concern count
    monthly_totals = Counter()              # (year, month) -> count
    district_crop = Counter()               # (district, crop) -> count
    district_concern = Counter()            # (district, concern) -> count
    district_month = Counter()              # (district, year, month) -> count
    district_block = Counter()              # (district, block) -> count

    offset = 0
    total_rows = None
    rows_read = 0
    t0 = time.time()

    while True:
        page = fetch_page(offset, session)
        if total_rows is None:
            total_rows = page.get("total", 0)
            print(f"Total Maharashtra records to read: {total_rows:,}", flush=True)
        records = page.get("records", [])
        if not records:
            break

        for rec in records:
            district = (rec.get("DistrictName") or "NA").strip()
            block = (rec.get("BlockName") or "NA").strip()
            crop_raw = rec.get("Crop")
            query_type = rec.get("QueryType")
            month = rec.get("month")
            year = rec.get("year")

            try:
                month_i = int(month)
            except (TypeError, ValueError):
                month_i = None

            concern = map_concern(query_type)
            season = season_of(month_i) if month_i else "Unknown"
            cbucket = crop_bucket(crop_raw)

            monthly_totals[(year, month_i)] += 1
            district_concern[(district, concern)] += 1
            district_month[(district, year, month_i)] += 1
            district_block[(district, block)] += 1

            if cbucket:
                crop_season[(cbucket, season)] += 1
                crop_season_concern[(cbucket, season, concern)] += 1
                district_crop[(district, cbucket)] += 1
                if year is not None:
                    crop_season_year[(cbucket, season, year)] += 1
                    if concern == "Water Management/Irrigation":
                        crop_season_year_water[(cbucket, season, year)] += 1

        rows_read += len(records)
        offset += PAGE_SIZE
        elapsed = time.time() - t0
        print(f"  page at offset {offset - PAGE_SIZE:>9,}: "
              f"{rows_read:,}/{total_rows:,} rows ({elapsed:5.0f}s elapsed)", flush=True)

        if rows_read >= total_rows:
            break

    print(f"Done. Read {rows_read:,} rows in {time.time() - t0:.0f}s.")

    def save(counter: Counter, cols, path):
        df = pd.DataFrame(
            [(*k, v) for k, v in counter.items()],
            columns=[*cols, "count"],
        )
        df.to_csv(DATA_DIR / path, index=False)
        print(f"  wrote {path} ({len(df):,} rows)")

    save(crop_season, ["crop", "season"], "crop_season_counts.csv")
    save(crop_season_concern, ["crop", "season", "concern"], "crop_season_concern_counts.csv")
    save(crop_season_year, ["crop", "season", "year"], "crop_season_year_counts.csv")
    save(crop_season_year_water, ["crop", "season", "year"], "crop_season_year_water_counts.csv")
    save(monthly_totals, ["year", "month"], "monthly_totals.csv")
    save(district_crop, ["district", "crop"], "district_crop_counts.csv")
    save(district_concern, ["district", "concern"], "district_concern_counts.csv")
    save(district_month, ["district", "year", "month"], "district_month_counts.csv")
    save(district_block, ["district", "block"], "district_block_counts.csv")


if __name__ == "__main__":
    main()
