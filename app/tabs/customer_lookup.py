"""Customer Lookup — drill into one customer's data, model output, and certainty."""

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


def _segment_median_chart(customer_row: pd.Series, segment_medians: pd.Series, seg_color: str):
    """Bar chart comparing this customer's RFM to segment medians."""
    features = ["recency", "frequency", "monetary"]
    labels = {"recency": "Recency", "frequency": "Frequency", "monetary": "Monetary (£)"}

    customer_vals = [customer_row[f] for f in features]
    segment_vals = [segment_medians[f] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[labels[f] for f in features],
        y=customer_vals,
        name="This customer",
        marker_color=seg_color,
        text=[f"{v:,.0f}" for v in customer_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=[labels[f] for f in features],
        y=segment_vals,
        name="Segment median",
        marker_color="#BDC3C7",
        text=[f"{v:,.0f}" for v in segment_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        height=320,
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _gmm_probability_chart(customer_row: pd.Series) -> go.Figure | None:
    """Bar chart of GMM cluster probabilities for this customer."""
    prob_cols = [c for c in customer_row.index if c.startswith("gmm_prob_")]
    if not prob_cols:
        return None

    probs = [customer_row[c] for c in prob_cols]
    labels = [f"GMM cluster {i}" for i in range(len(prob_cols))]

    fig = go.Figure(go.Bar(
        x=probs,
        y=labels,
        orientation="h",
        marker_color="#3498DB",
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        xaxis_title="Probability",
        xaxis=dict(range=[0, 1]),
        yaxis_title=None,
        margin=dict(l=80, r=40, t=20, b=40),
    )
    return fig


def render():
    df = load_customer_data()

    st.title("Customer Lookup")
    st.markdown(
        "Enter a Customer ID to see their segment, predicted lifetime value, "
        "and how confident the model is about the assignment."
    )

    # ---- Input ----
    col_input, col_random = st.columns([3, 1])
    with col_input:
        customer_input = st.text_input(
            "Customer ID",
            value="",
            placeholder="e.g. 14911",
        )
    with col_random:
        st.markdown("&nbsp;")  # vertical spacing
        random_button = st.button("Pick a random customer", use_container_width=True)

    if random_button:
        customer_input = str(int(df.sample(1)["Customer ID"].iloc[0]))
        st.session_state["customer_input"] = customer_input

    # Use session state if random button was clicked
    if "customer_input" in st.session_state and not customer_input:
        customer_input = st.session_state["customer_input"]

    if not customer_input:
        st.info("Enter a Customer ID above, or click 'Pick a random customer'.")
        return

    # ---- Parse and find ----
    try:
        cid = int(customer_input)
    except ValueError:
        st.error(f"'{customer_input}' is not a valid Customer ID (must be an integer).")
        return

    match = df[df["Customer ID"] == cid]
    if len(match) == 0:
        st.error(
            f"Customer ID {cid} not found in the retail dataset. "
            f"This could mean: they don't exist, they were classified as wholesale, "
            f"or they were removed during data cleaning."
        )
        return

    row = match.iloc[0]
    segment = row["cluster_name"]
    seg_color = SEGMENT_COLORS[segment]
    seg_median = df[df["cluster_name"] == segment].median(numeric_only=True)

    # ---- Header ----
    st.markdown("---")
    st.markdown(f"### Customer {cid}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", segment)
    c2.metric("Predicted 6-mo CLV", format_currency(row["clv_6m"]))
    c3.metric(
        "P(still active)",
        f"{row['p_alive']:.0%}",
        help="BG/NBD's estimate. For single-order customers, this is mathematically near 1 — "
             "the model has no evidence they've dropped out yet. Use predicted purchases for "
             "single-order customers instead."
    )
    c4.metric(
        "Predicted Purchases (6m)",
        f"{row['predicted_purchases_6m']:.2f}",
        help="Expected number of purchases over the next 6 months"
    )

    st.markdown("---")

    # ---- Two columns: customer profile vs segment median, and GMM probs ----
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Customer vs Segment Median (RFM)**")
        fig = _segment_median_chart(row, seg_median, seg_color)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**GMM Soft Cluster Probabilities**")
        st.caption(
            "How confident is the model that this customer belongs to each GMM cluster? "
            "A spread across multiple clusters suggests a borderline case."
        )
        fig = _gmm_probability_chart(row)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GMM probabilities not available for this customer.")

    st.markdown("---")

    # ---- Borderline / atypical flags ----
    flags = []
    if "is_hdbscan_noise" in row and row["is_hdbscan_noise"]:
        flags.append(
            "**HDBSCAN flagged this customer as atypical.** Their behavior doesn't "
            "match the dense clusters HDBSCAN identified — they may not fit the "
            "standard segment playbook cleanly."
        )

    # Borderline GMM: max probability under 0.7
    prob_cols = [c for c in row.index if c.startswith("gmm_prob_")]
    if prob_cols:
        max_prob = max(row[c] for c in prob_cols)
        if max_prob < 0.7:
            flags.append(
                f"**Borderline GMM assignment.** Maximum cluster probability is "
                f"only {max_prob:.0%} — this customer's behavior straddles "
                f"multiple segments. Treat segment-specific recommendations with caution."
            )

    if row["frequency"] == 1:
        flags.append(
            "**Single-order customer.** CLV predictions for single-order customers "
            "are conservative — BG/NBD requires repeat purchase history to forecast. "
            "Interpret p(alive) carefully (see tooltip on metric card)."
        )

    if flags:
        st.markdown("#### Notes on this customer")
        for flag in flags:
            st.warning(flag)

    # ---- Raw feature table ----
    with st.expander("Full customer record (raw features)"):
        display = row.drop(["Customer ID"]).to_frame("Value")
        display.index.name = "Feature"
        # Round numeric values for readability
        display["Value"] = display["Value"].apply(
            lambda v: f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)
        )
        st.dataframe(display, use_container_width=True)