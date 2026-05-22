"""Segment Explorer — drill into a single segment's behavior and value."""

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

# Which features to show in the segment profile
PROFILE_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "tenure",
    "avg_order_value",
    "n_unique_products",
]

FEATURE_LABELS = {
    "recency": "Recency (days since last purchase)",
    "frequency": "Frequency (number of orders)",
    "monetary": "Monetary (lifetime spend £)",
    "tenure": "Tenure (days from first to last purchase)",
    "avg_order_value": "Avg Order Value (£)",
    "n_unique_products": "Distinct products purchased",
    "days_since_first_purchase": "Days since first purchase",
    "avg_days_between_orders": "Avg days between orders",
}


def render():
    df = load_customer_data()

    st.title("Segment Explorer")
    st.markdown(
        "Select a segment to see its behavioral profile, value distribution, "
        "and top contributors."
    )

    # ---- Segment picker ----
    segment = st.selectbox(
        "Segment",
        options=SEGMENT_ORDER,
        index=0,
        help="Segments ordered roughly from highest to lowest value."
    )
    seg_color = SEGMENT_COLORS[segment]

    seg_df = df[df["cluster_name"] == segment]
    other_df = df[df["cluster_name"] != segment]

    # ---- Segment summary row ----
    st.markdown(f"### {segment}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(seg_df):,}")
    c2.metric("% of base", f"{len(seg_df) / len(df):.1%}")
    c3.metric("Median 6-mo CLV", format_currency(seg_df["clv_6m"].median()))
    c4.metric(
        "Total 6-mo CLV",
        format_currency(seg_df["clv_6m"].sum()),
        help=f"{seg_df['clv_6m'].sum() / df['clv_6m'].sum():.1%} of total predicted revenue"
    )

    # ---- Marketing implication callout ----
    marketing_notes = {
        "Champions": "**Retention priority #1.** This segment drives the majority of revenue. "
                     "Even a small churn rate is expensive. Consider VIP programs, "
                     "early access to new products, and personalized account management.",
        "At Risk": "**Largest retention opportunity.** These customers were once active "
                   "but are fading. Targeted re-engagement campaigns (personalized offers, "
                   "abandoned-cart reminders, win-back emails) have measurable ROI here.",
        "New & Promising": "**High-potential nurture segment.** Recently acquired with "
                           "early repeat behavior — invest in onboarding and second-purchase "
                           "incentives to convert them into Champions.",
        "Lost / One-Time Lapsed": "**Minimal investment justified.** Predicted CLV is near zero. "
                                  "A selective, low-cost win-back attempt (single email, "
                                  "discount code) is appropriate; broader campaigns aren't.",
        "New One-Time Buyers": "**Conversion-critical moment.** They just made their first "
                               "purchase. The goal is to drive a *second* purchase. Onboarding "
                               "sequences, product recommendations, and a time-sensitive "
                               "incentive in the next 30 days have highest impact here.",
    }
    st.info(marketing_notes.get(segment, "—"))

    st.markdown("---")

    # ---- Behavioral profile: segment vs rest ----
    st.markdown("#### Behavioral Profile vs Other Customers")
    st.caption("Median values. The segment's bar is colored; other segments combined are grey.")

    profile_data = []
    for feat in PROFILE_FEATURES:
        profile_data.append({
            "feature": FEATURE_LABELS[feat],
            "segment": seg_df[feat].median(),
            "others": other_df[feat].median(),
        })
    prof = pd.DataFrame(profile_data)

    # Two-column layout for the bar charts so they don't dominate
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=prof["feature"], y=prof["segment"], name=segment,
        marker_color=seg_color,
    ))
    fig.add_trace(go.Bar(
        x=prof["feature"], y=prof["others"], name="All other customers",
        marker_color="#BDC3C7",
    ))
    fig.update_layout(
        barmode="group",
        height=400,
        yaxis_title="Median value",
        xaxis_tickangle=-20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- CLV distribution ----
    st.markdown("#### CLV Distribution Within Segment")
    col_l, col_r = st.columns([2, 1])

    with col_l:
        # Only show non-zero CLVs for distribution clarity; mention zero count separately
        clv_nonzero = seg_df.loc[seg_df["clv_6m"] > 0, "clv_6m"]
        if len(clv_nonzero) > 0:
            # Clip at 99th pctl to keep visualization readable
            clip_val = clv_nonzero.quantile(0.99)
            fig = px.histogram(
                clv_nonzero.clip(upper=clip_val),
                nbins=40,
                color_discrete_sequence=[seg_color],
            )
            fig.update_layout(
                xaxis_title="6-month CLV (£)",
                yaxis_title="Customers",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "All customers in this segment have CLV = 0. "
                "BG/NBD requires repeat purchase history to predict; "
                "this segment doesn't have enough."
            )

    with col_r:
        st.markdown("**Distribution stats**")
        stats = {
            "Min": seg_df["clv_6m"].min(),
            "25th pct": seg_df["clv_6m"].quantile(0.25),
            "Median": seg_df["clv_6m"].median(),
            "75th pct": seg_df["clv_6m"].quantile(0.75),
            "Max": seg_df["clv_6m"].max(),
        }
        for label, value in stats.items():
            st.markdown(f"- **{label}:** {format_currency(value)}")

        zero_count = (seg_df["clv_6m"] == 0).sum()
        if zero_count > 0:
            st.caption(
                f"{zero_count:,} customers ({zero_count / len(seg_df):.0%}) "
                f"have CLV = 0 (insufficient repeat history)."
            )

    st.markdown("---")

    # ---- Top 10 by CLV ----
    st.markdown("#### Top 10 Customers by Predicted CLV")

    top10 = (
        seg_df.nlargest(10, "clv_6m")[
            ["Customer ID", "recency", "frequency", "monetary",
             "predicted_purchases_6m", "predicted_aov", "clv_6m"]
        ]
        .copy()
    )
    top10.columns = [
        "Customer ID", "Recency", "Frequency", "Past Spend (£)",
        "Predicted Purchases (6m)", "Predicted AOV (£)", "Predicted CLV (£)"
    ]
    top10["Customer ID"] = top10["Customer ID"].astype(int)
    top10["Past Spend (£)"] = top10["Past Spend (£)"].apply(lambda x: f"£{x:,.0f}")
    top10["Predicted AOV (£)"] = top10["Predicted AOV (£)"].apply(lambda x: f"£{x:,.0f}")
    top10["Predicted CLV (£)"] = top10["Predicted CLV (£)"].apply(lambda x: f"£{x:,.0f}")
    top10["Predicted Purchases (6m)"] = top10["Predicted Purchases (6m)"].round(2)

    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.caption(
        "💡 Click on a Customer ID in this list, then go to the 'Customer Lookup' "
        "tab to see their full profile."
    )