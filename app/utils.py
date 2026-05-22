"""Shared utilities for the Streamlit app."""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Friendly colors per segment — used consistently across all tabs
SEGMENT_COLORS = {
    "Champions": "#2E86AB",
    "At Risk": "#E67E22",
    "New & Promising": "#27AE60",
    "Lost / One-Time Lapsed": "#95A5A6",
    "New One-Time Buyers": "#9B59B6",
}

# Segment display order (most-to-least valuable)
SEGMENT_ORDER = [
    "Champions",
    "New & Promising",
    "At Risk",
    "New One-Time Buyers",
    "Lost / One-Time Lapsed",
]


@st.cache_data
def load_customer_data() -> pd.DataFrame:
    """Load the master per-customer DataFrame.

    Cached so it only loads once per session. The @st.cache_data
    decorator handles the lifecycle automatically.
    """
    return pd.read_parquet(DATA_DIR / "customer_final.parquet")


@st.cache_resource
def load_models() -> dict:
    """Load fitted clustering + CLV models.

    @st.cache_resource is for non-data objects (models, connections).
    """
    return {
        "clustering": joblib.load(MODELS_DIR / "clustering_pipeline.joblib"),
        "clv": joblib.load(MODELS_DIR / "clv_models.joblib"),
    }


def format_currency(value: float) -> str:
    """Format £ amounts nicely with comma separators."""
    return f"£{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.1%}"