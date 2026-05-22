"""Method Comparison — three clustering algorithms compared honestly."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from app.utils import (
    load_customer_data,
    load_models,
    SEGMENT_COLORS,
)


def _project_to_2d(features_clean: pd.DataFrame, preprocessing) -> np.ndarray:
    """Apply the project's preprocessing pipeline + PCA to get 2D coords."""
    X = preprocessing.transform(features_clean)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca


def render():
    df = load_customer_data()
    models = load_models()

    st.title("Method Comparison")
    st.markdown(
        "Three clustering algorithms were applied to identical preprocessed features. "
        "Each views the data differently, and the comparison is the point — different "
        "methods reveal different facets, and the *right* method depends on what you "
        "need from the segmentation."
    )

    st.markdown("---")

    # ---- The metric comparison table ----
    st.markdown("### Internal Validation Metrics")

    comparison = pd.DataFrame([
        {
            "Method": "K-Means (K=5)",
            "Clusters": 5,
            "Silhouette ↑": 0.306,
            "Davies-Bouldin ↓": 1.222,
            "Calinski-Harabasz ↑": 2134.68,
            "Noise points": 0,
            "Bootstrap ARI ↑": 0.967,
        },
        {
            "Method": "GMM (K=5)",
            "Clusters": 5,
            "Silhouette ↑": 0.170,
            "Davies-Bouldin ↓": 2.275,
            "Calinski-Harabasz ↑": 1386.70,
            "Noise points": 0,
            "Bootstrap ARI ↑": None,
        },
        {
            "Method": "HDBSCAN",
            "Clusters": 2,
            "Silhouette ↑": 0.397,
            "Davies-Bouldin ↓": 0.983,
            "Calinski-Harabasz ↑": 3359.57,
            "Noise points": 218,
            "Bootstrap ARI ↑": None,
        },
    ])

    # Mark the chosen method with a styled column
    comparison_display = comparison.copy()
    comparison_display["Selected"] = ["✓", "Auxiliary", "Reference"]
    cols_order = ["Method", "Selected", "Clusters", "Silhouette ↑", "Davies-Bouldin ↓",
                  "Calinski-Harabasz ↑", "Noise points", "Bootstrap ARI ↑"]
    st.dataframe(
        comparison_display[cols_order],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "↑ higher is better, ↓ lower is better. "
        "Bootstrap ARI = adjusted Rand index across 30 bootstrap resamples — "
        "a measure of cluster stability."
    )

    st.markdown("---")

    # ---- The interpretation ----
    st.markdown("### Why K-Means Was Chosen")

    st.markdown(
        """
        **HDBSCAN scored higher on every internal metric — but produced only 2 clusters 
        plus 218 noise points.** This is a textbook example of why metric-best is not 
        always business-best:
        
        - Fewer clusters = larger gaps between them = better silhouette and Davies-Bouldin scores
        - But "2 clusters" isn't actionable segmentation; you can't run differentiated campaigns 
          when everyone is either "Group A" or "Group B"
        - The metrics reward what they measure (compactness, separation), not what the business 
          needs (interpretable, differentiated segments)
        
        **GMM scored lower than K-Means on silhouette by design.** Silhouette penalizes overlapping 
        cluster boundaries, which GMM's soft assignments allow. GMM isn't worse — it's optimizing 
        something different (likelihood, with allowed overlap). We use GMM's *probabilistic outputs* 
        as an enrichment layer on the customer lookup page, surfacing borderline customers that 
        K-Means' hard boundaries hide.
        
        **K-Means with K=5 wins on what matters here:** 5 interpretable segments with strong 
        bootstrap stability (ARI = 0.97 — clusters reproduce reliably across resamples). 
        This is the right tool for the segmentation job; the other methods earn their keep elsewhere.
        """
    )

    # The 218 noise points story
    if "is_hdbscan_noise" in df.columns:
        n_noise = int(df["is_hdbscan_noise"].sum())
        st.info(
            f"**The 218 HDBSCAN noise points aren't wasted output.** They're surfaced as a "
            f"'borderline customer' flag in the Customer Lookup tab — customers whose behavior "
            f"doesn't fit cleanly into the dense majority. {n_noise} customers carry this flag "
            f"in the current dataset."
        )

    st.markdown("---")

    # ---- 2D projection of clusterings ----
    st.markdown("### Visualizing the Three Clusterings")
    st.caption(
        "All three algorithms operate in the same 8-dimensional feature space. "
        "PCA projects this to 2D so we can compare how each method partitions customers. "
        "Each dot is one customer; color = cluster assignment."
    )

    # Pull the exact column order from the saved pipeline
    feature_cols = models["clustering"]["feature_columns"]
    features_clean = df[feature_cols].copy()

    # customer_final.parquet already has avg_days_between_orders imputed
    # (the clustering step worked on imputed data and saved it), so no
    # imputation needed here.

    preprocessing = models["clustering"]["preprocessing"]
    coords, pca = _project_to_2d(features_clean, preprocessing)

    var_explained = pca.explained_variance_ratio_.sum()
    st.caption(f"PCA explains {var_explained:.0%} of variance in the 2-component projection.")

    df_plot = df.copy()
    df_plot["pca_1"] = coords[:, 0]
    df_plot["pca_2"] = coords[:, 1]

    col_l, col_m, col_r = st.columns(3)

    with col_l:
        st.markdown("**K-Means**")
        fig = px.scatter(
            df_plot, x="pca_1", y="pca_2",
            color="cluster_name",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.5,
            height=400,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            showlegend=False,
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_m:
        st.markdown("**GMM**")
        # GMM uses integer labels — give them a categorical color
        df_plot["gmm_label"] = df_plot["cluster_gmm"].astype(str)
        fig = px.scatter(
            df_plot, x="pca_1", y="pca_2",
            color="gmm_label",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.5,
            height=400,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            showlegend=False,
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**HDBSCAN**")
        if "is_hdbscan_noise" in df_plot.columns:
            df_plot["hdb_label"] = np.where(
                df_plot["is_hdbscan_noise"], "Noise",
                "Cluster " + (df_plot["cluster_km"].astype(str))  # placeholder — see note below
            )
        else:
            df_plot["hdb_label"] = "Cluster 0"

        fig = px.scatter(
            df_plot, x="pca_1", y="pca_2",
            color="is_hdbscan_noise" if "is_hdbscan_noise" in df_plot.columns else None,
            color_discrete_map={True: "#E74C3C", False: "#3498DB"},
            opacity=0.5,
            height=400,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            showlegend=False,
            xaxis_title="PC1",
            yaxis_title="PC2",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = noise, blue = clustered")

    st.markdown(
        """
        **What to notice in these plots:**
        - K-Means partitions space into 5 contiguous regions — clean boundaries
        - GMM produces similar groupings but with some overlap (the boundaries are softer)
        - HDBSCAN sees mostly one dense region with scattered noise — confirming that 
          customer behavior in RFM space is a smooth gradient, not naturally clumpy
        """
    )

    st.markdown("---")

    # ---- Stability detail ----
    st.markdown("### Cluster Stability (K-Means)")
    st.markdown(
        """
        To verify K-Means clusters aren't artifacts of the particular sample, the segmentation 
        was re-run on 30 bootstrap resamples (80% of customers each time). The Adjusted Rand 
        Index between each bootstrap clustering and the full-data clustering was recorded.
        
        - **Mean ARI: 0.967** — extremely high agreement, indicating very stable clusters
        - **Range: [0.583, 0.994]** — two bootstrap runs converged to alternate local optima, 
          but 28 of 30 produced essentially identical clusterings
        - **Interpretation:** core cluster boundaries are robust; a small number of borderline 
          customers shift assignment between runs, which is expected and surfaced via the 
          GMM probabilities in the Customer Lookup tab.
        """
    )