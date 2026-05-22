# Customer Segmentation & CLV Prediction

End-to-end customer analytics on a UK gift retailer's transaction history: 
segmenting 5,248 retail customers, predicting 6-month lifetime value, and 
surfacing insights through an interactive Streamlit dashboard.

**Live app:** https://customer-segmentation-clv.streamlit.app

---

## Headline findings

- **26% of customers (Champions) are projected to drive 69% of next-period 
  revenue** — a sharper Pareto than the 77% historical concentration, 
  suggesting value concentration is intensifying.
- **£403k of future revenue is at risk** in a segment of 1,911 customers 
  showing 6-month churn signals — the largest single retention opportunity.
- **£2.29M total predicted 6-month revenue** — approximately 86% of the 
  historical 6-month average, reflecting expected churn the model 
  accounts for honestly.
- **103 wholesale customers (1.7% of base) drive 37% of historical revenue** 
  and were split out into a separate dataset to keep retail segmentation 
  actionable.

---

## What's in this project

**1. Rigorous exploratory analysis** (notebook 01) — uncovered the 77% 
revenue Pareto, the wholesale concentration, ~22% missing Customer IDs, 
and a host of data-quality issues that shaped downstream decisions.

**2. Production-grade cleaning** (notebook 02 + `src/preprocessing.py`) — 
ten cleaning rules applied via a composable pipeline; each step's row 
count logged for auditability; wholesale customers split out cleanly.

**3. Per-customer feature engineering** (notebook 03 + `src/features.py`) — 
8 features capturing RFM + behavioral + lifecycle dimensions, with 
log-transformation justified by inspecting raw distributions.

**4. Three clustering methods compared honestly** (notebook 04 + 
`src/clustering.py`) — K-Means (selected), GMM (enrichment), and HDBSCAN 
(outlier detection), with bootstrap stability validation.

**5. CLV prediction with validation** (notebook 05 + `src/clv.py`) — 
BG/NBD + Gamma-Gamma fitted and verified via calibration/holdout split 
and independence-assumption check.

**6. Interactive dashboard** (`app/`) — Streamlit app with five tabs: 
overview, segment explorer, customer lookup, method comparison, about.

---

## Methodology notes worth highlighting

### Honest method comparison

K-Means with K=5 was selected — but it wasn't the metric winner. 
HDBSCAN scored higher on every internal validation metric (silhouette 
0.397 vs K-Means 0.306) but produced only 2 clusters + 218 noise points. 
The metrics reward cluster compactness; the business needs interpretable, 
differentiated segments. The chosen method is the right tool for the 
*business* goal, not the metric-best result.

Similarly within K-Means: K=2 had the highest silhouette in the sweep, 
but two clusters aren't actionable. K=5 was chosen via elbow analysis 
plus business interpretability.

GMM scored lower than K-Means on silhouette by design — silhouette 
penalizes the overlapping boundaries GMM's soft assignments allow. 
GMM's value isn't winning on metrics; its probabilistic outputs surface 
**borderline customers** in the dashboard (e.g., "this customer is 60% 
Champion, 40% At Risk").

HDBSCAN's 218 "noise" customers are surfaced as a **borderline customer 
flag** in the lookup tab — customers whose lifecycle doesn't fit the 
dominant retail arc.

### Rigorous validation

- **Bootstrap stability** (K-Means): mean ARI = 0.97 over 30 resamples 
  (28/30 runs near-identical), confirming cluster boundaries are robust
- **Gamma-Gamma assumption check**: frequency × monetary correlation = 
  0.093 among repeat customers, comfortably below the 0.1 threshold
- **BG/NBD calibration/holdout split**: trained on 18 months, validated 
  against 183 days held out — predictions aligned along the diagonal 
  with no systematic bias

### Known limitations documented (not hidden)

- **CLV is £0 for ~29% of customers** — BG/NBD requires repeat purchase 
  history; single-order customers can't be forecast under this model. 
  New-customer CLV would require a separate approach (e.g., binary 
  classifier on first-month behavior) — flagged as future work.
- **No product-category features** — segmentation reflects behavior, not 
  product preferences. Would require text clustering on descriptions.
- **UK-only scope** — other countries (~8% of cleaned revenue) excluded 
  for analytical focus.
- **p_alive caveat** — BG/NBD's "probability alive" is mechanically near 1 
  for single-order customers (they haven't had any opportunities to drop 
  out under the model). Use predicted purchases or CLV directly for 
  individual decisions; surfaced via tooltip in the app.

---

## Final segmentation

| Segment | Customers | % | Median 6m CLV | Total CLV | % of Future Revenue |
|---|---|---|---|---|---|
| **Champions** | 1,376 | 26.2% | £883 | £1,573,197 | **68.7%** |
| **At Risk** | 1,911 | 36.4% | £170 | £402,873 | 17.6% |
| **New & Promising** | 473 | 9.0% | £463 | £311,917 | 13.6% |
| **Lost / One-Time Lapsed** | 1,117 | 21.3% | £0 | £595 | <1% |
| **New One-Time Buyers** | 371 | 7.1% | £0 | £0 | 0% |

**Business interpretation:**
- 35% active base (Champions + New & Promising) drives most current revenue
- 36% at risk of churning — largest single retention opportunity
- 28% one-time buyers — conversion challenge

---

## Project structure
customer-segmentation-clv/
├── app/                        # Streamlit dashboard
│   ├── streamlit_app.py        # main entrypoint
│   ├── utils.py                # shared data loading
│   └── tabs/                   # one file per tab
│       ├── overview.py
│       ├── segment_explorer.py
│       ├── customer_lookup.py
│       ├── method_comparison.py
│       └── about.py
├── data/
│   ├── raw/                    # original Excel (gitignored)
│   └── processed/              # cleaned parquets
├── models/                     # fitted sklearn + lifetimes models
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_clv.ipynb
├── src/                        # production code
│   ├── preprocessing.py        # cleaning pipeline
│   ├── features.py             # feature engineering
│   ├── clustering.py           # clustering + segment names
│   └── clv.py                  # BG/NBD + Gamma-Gamma wrappers
├── requirements.txt
└── README.md

The **`src/` ↔ notebook split** matters: cleaning, feature engineering, 
clustering, and CLV logic live as importable functions in `src/`. 
Notebooks orchestrate and document. The Streamlit app imports from the 
same `src/` modules. One source of truth — no drift between notebooks 
and the deployed app.

---

## Data pipeline

Raw Excel (1.07M rows)
│
▼
Cleaning (src/preprocessing.py)
│   • Drop missing Customer ID (-22%)
│   • Drop cancellations
│   • Drop non-product SKUs (POST, BANK CHARGES, AMAZONFEE, ...)
│   • Drop non-positive prices
│   • Filter to UK
│   • Split wholesale (>60 orders OR >£20k spend) → 86 customers
│   • Drop non-positive-spend retail customers
▼
retail_clean.parquet (624,590 transactions, 5,248 customers)
│
▼
Feature engineering (src/features.py)
│   • 8 features: recency, frequency, monetary, tenure,
│     days_since_first_purchase, avg_order_value,
│     avg_days_between_orders, n_unique_products
│   • Snapshot date: 2011-12-10
▼
customer_features.parquet (5,248 × 8)
│
├──► Clustering (src/clustering.py)
│       • log1p + StandardScaler
│       • K-Means K=5 + GMM K=5 + HDBSCAN
│       • Bootstrap stability (ARI = 0.97)
│       • Named segments
│
└──► CLV (src/clv.py)
• BG/NBD: purchase frequency + p_alive
• Gamma-Gamma: average transaction value
• 6-month prediction, validated via cal/holdout split
│
▼
customer_final.parquet
Per-customer: features + segment + GMM probs + HDBSCAN flag + CLV
│
▼
Streamlit app

---

## Tech stack

**Data:** pandas · numpy · pyarrow (parquet)  
**Modeling:** scikit-learn · hdbscan · lifetimes (BG/NBD, Gamma-Gamma)  
**Visualization:** matplotlib · seaborn (notebooks) · Plotly (app)  
**App:** Streamlit  
**Reproducibility:** pinned dependencies; clean code/notebook separation; 
modular `src/` package

---

## Running locally

```bash
git clone https://github.com/yourusername/customer-segmentation-clv
cd customer-segmentation-clv
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download raw data (~45 MB) from UCI into data/raw/
# https://archive.ics.uci.edu/dataset/502/online+retail+ii

# Run the analysis end-to-end via notebooks 01-05, then:
streamlit run app/streamlit_app.py
```

The pipeline is deterministic — same inputs produce same outputs. 
Bootstrap ARI confirms cluster reproducibility across resamples.

---

## Future work

Each item below is a real extension that would deliver meaningful 
additional value, not just polish:

- **New-customer CLV model** — a binary classifier on first-month 
  behavior predicting whether customers will become repeat buyers. 
  Would fill the gap BG/NBD leaves for one-time buyers.
- **Survival analysis** — `lifelines` library for time-to-next-purchase 
  and time-to-churn with proper survival curves; complements the 
  Champions / At Risk distinction.
- **Uplift modeling** — predicting *who would respond* to a retention 
  offer, not just *who's likely to churn*. The actual frontier of 
  marketing analytics.
- **Product-affinity features** — text clustering on product descriptions 
  to derive categories; segment by preference, not just behavior.
- **Wholesale account dashboard** — separate top-accounts analysis for 
  the 103 wholesale customers (37% of historical revenue).
- **Multi-country extension** — re-introduce the 8% of revenue outside 
  the UK, with country-level features and possibly per-country 
  sub-segmentation.

---

## Data source

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 
transactions from a UK-based online gift retailer, Dec 2009 – Dec 2011.

---

## Built by

**Usama Bin Rahat**  
[www.linkedin.com/in/usama-bin-rahat/] · [usama.bin.rahat@gmail.com]