# Kisan Call Centre (KCC) Data Analysis – Maharashtra

A full-census dashboard of Maharashtra's Kisan Call Centre (KCC) farmer helpline data — hotspot districts and blocks, crop and seasonal query patterns, concern breakdowns, and policy-targeting recommendations.

**[View the live dashboard →](https://d-murphy16.github.io/maharashtra-kisan-ledger/)**

## What this is

Every farmer call logged for Maharashtra in India's Kisan Call Centre dataset — 5,167,913 records spanning 2009–2025 — pulled in full (no sampling) from the live [data.gov.in API](https://www.data.gov.in/datasets_webservices/datasets/6622307) and aggregated client-side into:

- **A spatial hotspot map** of all 36 districts, with a layer toggle between total call volume and specific concern categories (weather, pest/disease, nutrient, government schemes)
- **A crop ledger** ranking all named crops by call volume
- **Seasonal analysis** showing how query volume tracks the Kharif/Rabi/Zaid cropping calendar, month-by-month and year-over-year
- **A concern breakdown** of what farmers are actually asking about, overall and by district
- **Policy-targeting recommendations** — the highest-volume district × block × crop combination for each actionable concern, with suggested interventions

## Data source

Kisan Call Centre (KCC) transcripts dataset, Ministry of Agriculture and Farmers Welfare, via `api.data.gov.in` (resource `cef25fe2-9231-4128-8aec-2c948fedd43f`). Full methodology and data-quality notes are documented in the dashboard's own "Methodology & data notes" section.

## Analysis pipeline

The [`analysis/`](analysis/) directory has the Python code behind this dashboard, plus inferential statistics (chi-square, Kruskal-Wallis) and regression models (Poisson / Negative Binomial / OLS) that go beyond what the dashboard shows — see [`analysis/README.md`](analysis/README.md).

## Running locally

This is a single self-contained HTML file with no build step or dependencies beyond Google Fonts (loaded via CDN). Just open `index.html` in a browser, or serve it:

```bash
python3 -m http.server 8000
```

## License

Code in this repository is [MIT licensed](LICENSE). The underlying data is published by the Government of India under the [National Data Sharing and Accessibility Policy](https://data.gov.in/government-open-data-license-india) and is not covered by that license.
