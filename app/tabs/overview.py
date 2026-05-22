"""Overview tab — landing page with headline KPIs and segment distribution."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils import (
    load_customer_data,
    SEGMENT_COLORS,
    SEGMENT_ORDER,
    format_currency,
)


def render():
    """Render the Overview tab."""
    df = load_customer_data()

    st.title("Customer Segmentation & CLV Dashboard")
    st.markdown(
        "Analysis of **5,248 retail customers** from a UK online gift retailer "
        "(2009–2011). Segmentation via K-Means clustering on RFM + behavioral "
        "features; predicted 6-month CLV via BG/NBD + Gamma-Gamma models."
    )

    # ---- KPI row ----
    st.markdown("### Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    n_customers = len(df)
    total_clv = df["clv_6m"].sum()
    champions_count = (df["cluster_name"] == "Champions").sum()
    champions_share = df.loc[df["cluster_name"] == "Champions", "clv_6m"].sum() / total_clv
    at_risk_clv = df.loc[df["cluster_name"] == "At Risk", "clv_6m"].sum()

    c1.metric("Total Customers", f"{n_customers:,}")
    c2.metric("Predicted 6-mo Revenue", format_currency(total_clv))
    c3.metric(
        "Champions Share",
        f"{champions_share:.0%}",
        help=f"{champions_count:,} customers ({champions_count / n_customers:.0%} of base) "
             f"drive {champions_share:.0%} of predicted revenue",
    )
    c4.metric(
        "At Risk Revenue",
        format_currency(at_risk_clv),
        help="Predicted revenue from customers showing churn signals — the retention opportunity"
    )

    st.markdown("---")

    # ---- Headline finding ----
    st.markdown(
        f"""
        ### Headline Finding
        
        **{champions_count:,} Champions ({champions_count / n_customers:.0%} of customers) 
        are projected to generate {champions_share:.0%} of next-period revenue.**
        
        This is a sharper Pareto than the historical 77% — suggesting value concentration 
        is *increasing*, not stabilizing. Strategic implication: retention investment 
        should be heavily weighted toward Champions and the At Risk segment behind them.
        """
    )

    st.markdown("---")

    # ---- Segment distribution ----
    st.markdown("### Segment Distribution")

    seg_summary = (
        df.groupby("cluster_name")
        .agg(
            n_customers=("clv_6m", "size"),
            total_clv=("clv_6m", "sum"),
            median_clv=("clv_6m", "median"),
        )
        .reindex(SEGMENT_ORDER)
        .reset_index()
    )
    seg_summary["pct_customers"] = seg_summary["n_customers"] / seg_summary["n_customers"].sum() * 100
    seg_summary["pct_revenue"] = seg_summary["total_clv"] / seg_summary["total_clv"].sum() * 100

    col_l, col_r = st.columns(2)

    # Left: customer count by segment (donut chart)
    with col_l:
        fig = px.pie(
            seg_summary,
            values="n_customers",
            names="cluster_name",
            hole=0.5,
            color="cluster_name",
            color_discrete_map=SEGMENT_COLORS,
            title="Customers by Segment",
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Right: revenue share by segment (bar chart)
    with col_r:
        fig = px.bar(
            seg_summary,
            x="pct_revenue",
            y="cluster_name",
            orientation="h",
            color="cluster_name",
            color_discrete_map=SEGMENT_COLORS,
            title="Predicted Revenue Share by Segment",
            text=seg_summary["pct_revenue"].round(1).astype(str) + "%",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="% of 6-month predicted revenue",
            yaxis_title=None,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Segment-level table ----
    st.markdown("### Per-Segment Detail")
    display_table = seg_summary[
        ["cluster_name", "n_customers", "pct_customers", "median_clv", "total_clv", "pct_revenue"]
    ].copy()
    display_table.columns = [
        "Segment", "Customers", "% Customers", "Median CLV (£)", "Total CLV (£)", "% of Revenue"
    ]
    display_table["% Customers"] = display_table["% Customers"].round(1).astype(str) + "%"
    display_table["% of Revenue"] = display_table["% of Revenue"].round(1).astype(str) + "%"
    display_table["Median CLV (£)"] = display_table["Median CLV (£)"].round(0).astype(int).apply(lambda x: f"£{x:,}")
    display_table["Total CLV (£)"] = display_table["Total CLV (£)"].round(0).astype(int).apply(lambda x: f"£{x:,}")

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    # ---- Methodology note ----
    with st.expander("Methodology summary"):
        st.markdown(
            """
            - **Data:** ~655k UK retail transactions from Online Retail II dataset (UCI)
            - **Cleaning:** removed cancellations, non-product SKUs, missing IDs; split out 103 wholesale customers
            - **Features:** 8 per-customer features (RFM + tenure, AOV, breadth)
            - **Clustering:** K-Means K=5 selected via elbow + silhouette; validated with bootstrap stability (ARI = 0.97)
            - **CLV:** BG/NBD for purchase prediction, Gamma-Gamma for AOV; calibration/holdout validated
            - **Compared against:** GMM and HDBSCAN for honest method evaluation
            """
        )