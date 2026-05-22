"""Main Streamlit app entrypoint.

Run locally: `streamlit run app/streamlit_app.py` from the project root.
"""

import sys
from pathlib import Path

# Make project root importable so `from app.utils import ...` works
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.tabs import (
    overview,
    segment_explorer,
    customer_lookup,
    method_comparison,
    about,
)

st.set_page_config(
    page_title="Customer Segmentation & CLV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Sidebar — minimal, for now
with st.sidebar:
    st.markdown("### Customer Segmentation & CLV")
    st.markdown(
        "A portfolio project demonstrating end-to-end customer analytics: "
        "EDA, clustering (K-Means / GMM / HDBSCAN), and CLV prediction "
        "(BG/NBD + Gamma-Gamma)."
    )
    st.markdown("---")
    st.markdown(
        "**Built with:** pandas, scikit-learn, hdbscan, lifetimes, Streamlit, Plotly  \n"
        "[GitHub](https://github.com/yourusername/customer-segmentation-clv)"
    )

# Main tabs
tab_overview, tab_segments, tab_lookup, tab_comparison, tab_about = st.tabs(
    ["Overview", "Segment Explorer", "Customer Lookup", "Method Comparison", "About"]
)

with tab_overview:
    overview.render()

with tab_segments:
    segment_explorer.render()

with tab_lookup:
    customer_lookup.render()

with tab_comparison:
    method_comparison.render()

with tab_about:
    about.render()