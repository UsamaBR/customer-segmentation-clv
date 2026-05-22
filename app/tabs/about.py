"""About — methodology, decisions, limitations, future work."""

import streamlit as st


def render():
    st.title("About This Project")

    st.markdown(
        """
        ### The Problem
        
        A UK online gift retailer's two-year transaction history contains 5,248 customers 
        and £10.6M in revenue. Like most retail businesses, customer value is heavily 
        concentrated — but *which* customers, *how much* future revenue they represent, 
        and *which segments to invest retention budget in* are unclear from raw transactions.
        
        This project answers those questions through a complete analytical pipeline: 
        from raw transactions to deployable predictions, with rigorous validation at each step.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Approach
        
        **1. Data preparation** — Online Retail II dataset (UCI). Cleaning removed:
        - 22% of rows with missing Customer IDs
        - 19k cancellation invoices ("C"-prefixed)
        - 6,207 zero-price entries (administrative non-product rows)
        - Non-UK transactions (~8% of clean rows) for focus
        - 103 wholesale customers (>60 orders or >£20k spend) split into a separate dataset
        
        **2. Feature engineering** — 8 per-customer features computed:
        - **RFM:** recency, frequency, monetary
        - **Behavioral:** tenure, average order value, avg days between orders, distinct products
        - **Lifecycle:** days since first purchase
        
        **3. Segmentation** — three clustering methods compared:
        - **K-Means (K=5)** — selected. Chosen via elbow + silhouette; bootstrap ARI = 0.97
        - **GMM (K=5)** — used as enrichment for borderline customer probabilities
        - **HDBSCAN** — used as outlier detector, surfacing 218 atypical customers
        
        **4. CLV prediction** — BG/NBD + Gamma-Gamma:
        - BG/NBD predicts future purchase frequency (Beta-Geometric / Negative Binomial)
        - Gamma-Gamma predicts average transaction value
        - Combined for expected 6-month revenue per customer
        - Validated via calibration/holdout split
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Headline Findings
        
        - **26% of customers (Champions) drive 69% of next-period predicted revenue** — a sharper Pareto than the historical 77%, suggesting concentration is increasing
        - **Total predicted 6-month revenue: £2.29M** — ~86% of historical 6-month average, reflecting natural churn the model accounts for
        - **At Risk segment is the largest retention opportunity: £403k** in projected revenue from 1,911 customers showing churn signals
        - **The Lost segment is effectively dead: <£600** total predicted revenue from 1,117 customers — validates minimal win-back investment
        - **CLV modeling exposed a real product gap:** New customers receive ~£0 CLV because BG/NBD requires repeat purchase history; predicting new-customer value requires a different modeling approach
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Limitations Acknowledged
        
        These aren't excuses — they're known constraints worth understanding:
        
        - **CLV for new customers:** BG/NBD cannot predict customers with only one purchase. New & Promising and New One-Time Buyer segments would benefit from a separate conversion model trained on early-tenure behavior.
        - **No product-category features:** Stock codes weren't categorized; segmentation reflects purchasing *behavior* but not purchasing *preferences*. Text clustering on product descriptions would enrich this.
        - **UK-only:** Other countries excluded for cleanliness. A multi-country version would need country-level features and possibly per-country sub-segmentation.
        - **Two-year window:** Predictions assume customer behavior patterns continue. Major business changes (pricing, marketing strategy, product line) would invalidate predictions.
        - **Wholesale customers excluded:** The 103 wholesale accounts represent 37% of revenue but were removed from segmentation to avoid distorting the retail-focused analysis. A separate B2B analysis would be valuable.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Future Work
        
        - **New-customer CLV model** — binary classifier on first-month behavior predicting whether they'll become repeat buyers
        - **Survival analysis** — using `lifelines` to model time-to-next-purchase and time-to-churn with proper survival curves
        - **Uplift modeling** — predicting *who would respond* to a retention offer, not just *who's likely to churn*
        - **Product-affinity features** — text clustering on descriptions to derive product categories; segment by preference, not just behavior
        - **Wholesale account analysis** — a separate top-accounts dashboard for the 103 wholesale customers
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Tech Stack
        
        - **Data:** pandas, numpy, pyarrow (parquet)
        - **Modeling:** scikit-learn (clustering, preprocessing), hdbscan, lifetimes (BG/NBD, Gamma-Gamma)
        - **Visualization:** matplotlib, seaborn (notebooks); Plotly (interactive app)
        - **App:** Streamlit
        - **Reproducibility:** pinned dependencies in `requirements.txt`; clean code/notebook separation; modular `src/` package
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Links
        
        - **GitHub:** [github.com/yourusername/customer-segmentation-clv](https://github.com/yourusername/customer-segmentation-clv)  *(update with your URL)*
        - **Data source:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
        - **Built by:** [Your Name](#)  *(add your LinkedIn / portfolio link)*
        """
    )