# Conversion Rate Challenge

[![CI](https://github.com/tryph0n/conversion-rate-challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/tryph0n/conversion-rate-challenge/actions/workflows/ci.yml)

> Binary classification to predict newsletter subscription conversions on highly imbalanced data (3.2% positive class). Includes a Jupyter notebook pipeline and a Streamlit web app.

## Key Results

- **Best model:** GradientBoosting (untuned baseline) — threshold optimization selected over class reweighting
- **Default threshold (0.5):** F1 = 0.7593
- **Optimized threshold (0.4459):** F1 = 0.7629, Precision = 0.8542, Recall = 0.6892
- **Break-even threshold (0.3025):** F1 = 0.7516, Precision ≈ Recall ≈ 0.75
- **Top features:** `total_pages_visited` (0.887), `is_china` (0.040), `new_user` (0.023)
- **Class imbalance:** 96.8% / 3.2% split (~30:1). `class_weight` 5x tested and rejected (degraded F1 by -0.002); threshold optimization retained instead
- **Models tested:** Logistic Regression, Random Forest, GradientBoosting (baseline)
- **Geography:** Germany has the highest conversion rate (~8-10%), China the lowest (~0.1%)
- **User type:** Returning users convert 5x more than new users

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.13 |
| Data | pandas, numpy |
| ML | scikit-learn |
| Visualization | Plotly |
| App | Streamlit |
| Serialization | joblib |
| Package manager | uv |

## Installation

```bash
git clone https://github.com/tryph0n/conversion-rate-challenge.git
cd conversion-rate-challenge
uv sync
```

## Usage

**Step 1 -- Launch the Streamlit app** (4 pages: Home, Methodology, Dashboard, Predictions):

```bash
uv run streamlit run app/Home.py
```

The trained model is included in models/ (3 files, about 150 KB), so the Home, Methodology and Predictions pages work right after a clone. The Dashboard page needs the training CSV (see Data).

**Step 2 -- Retrain (optional).** The notebook rewrites `models/` and needs the CSV files:

```bash
uv run jupyter notebook conversion_rate_final.ipynb
```

## Data

Data is not included in the repository (gitignored). Place the following CSV files in `data/input/`:

| File | Rows | Columns |
|------|------|---------|
| `conversion_data_train.csv` | 284,580 | 6 (5 features + target) |
| `conversion_data_test.csv` | 31,620 | 5 (features only) |

Source: course dataset of the Jedha data science training (newsletter conversion case study). The source states no licence; these files are not covered by this repository's licence.

Features: `country`, `age`, `new_user`, `source`, `total_pages_visited`. Target: `converted` (binary).

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## License

Code: MIT, see [LICENSE](LICENSE).
