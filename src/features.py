"""
Feature engineering for customer segmentation.

Aggregates transaction-level data into one row per customer, computing
RFM and behavioral features used for clustering and CLV modeling.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def compute_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Snapshot date is one day after the last transaction in the data."""
    return df['InvoiceDate'].max().normalize() + timedelta(days=1)


def build_customer_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate transaction-level data into per-customer features.

    Expects the cleaned retail DataFrame (output of clean_transactions).
    Returns a DataFrame indexed by Customer ID with engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions. Must contain columns:
        Customer ID, Invoice, InvoiceDate, StockCode, Quantity, Price, Revenue.
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculations. Defaults to one day
        after the max InvoiceDate.

    Returns
    -------
    pd.DataFrame
        One row per customer. Index is Customer ID. Columns:
        recency, frequency, monetary, tenure, avg_order_value,
        avg_days_between_orders, n_unique_products, weekend_purchase_ratio,
        days_since_first_purchase.
    """
    if snapshot_date is None:
        snapshot_date = compute_snapshot_date(df)

    # Mark weekend purchases at the transaction level
    df = df.copy()

    # Aggregate to one row per customer
    grouped = df.groupby('Customer ID')

    features = pd.DataFrame(index=grouped.size().index)

    # First and last purchase dates per customer
    first_purchase = grouped['InvoiceDate'].min()
    last_purchase = grouped['InvoiceDate'].max()

    # RFM core
    features['recency'] = (snapshot_date - last_purchase).dt.days
    features['frequency'] = grouped['Invoice'].nunique()
    features['monetary'] = grouped['Revenue'].sum()

    # Tenure: span between first and last purchase
    features['tenure'] = (last_purchase - first_purchase).dt.days
    features['days_since_first_purchase'] = (snapshot_date - first_purchase).dt.days

    # Derived ratios — guard against division by zero
    features['avg_order_value'] = features['monetary'] / features['frequency']

    # avg_days_between_orders: only meaningful for customers with >1 order
    # For single-order customers, set to NaN; we'll handle below.
    multi_order = features['frequency'] > 1
    features['avg_days_between_orders'] = np.where(
        multi_order,
        features['tenure'] / (features['frequency'] - 1),
        np.nan,
    )

    # Product breadth
    features['n_unique_products'] = grouped['StockCode'].nunique()

    
    return features