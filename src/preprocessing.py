"""
Data cleaning module for the Online Retail II dataset.

The main entrypoint is `clean_transactions()`. Individual cleaning steps
are exposed as separate functions so they can be tested or composed
differently if needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


NON_PRODUCT_STOCKCODES = {
    'POST', 'D', 'C2', 'M', 'BANK CHARGES', 'PADS', 'DOT',
    'CRUK', 'AMAZONFEE', 'B', 'TEST001', 'TEST002', 'gift_0001',
    'ADJUST', 'ADJUST2', 'SP1002', 'AMAZON'
}


# ---------------- Loading ----------------

def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load both sheets of the Online Retail II Excel file."""
    filepath = Path(filepath)
    a = pd.read_excel(filepath, sheet_name="Year 2009-2010")
    b = pd.read_excel(filepath, sheet_name="Year 2010-2011")
    return pd.concat([a, b], ignore_index=True)


# ---------------- Atomic cleaning steps ----------------

def drop_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['Customer ID']).copy()


def drop_cancellation_invoices(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['Invoice'].astype(str).str.startswith('C')].copy()


def drop_non_positive_quantity(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Quantity'] > 0].copy()


def drop_non_positive_price(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Price'] > 0].copy()


def drop_non_product_stockcodes(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['StockCode'].astype(str).str.upper().isin(NON_PRODUCT_STOCKCODES)
    return df[~mask].copy()


def filter_country(df: pd.DataFrame, country: str = "United Kingdom") -> pd.DataFrame:
    return df[df['Country'] == country].copy()


def cast_customer_id_to_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Customer ID'] = df['Customer ID'].astype(int)
    return df


def add_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Revenue'] = df['Quantity'] * df['Price']
    return df


def split_wholesale_retail(
    df: pd.DataFrame,
    order_threshold: int = 60,
    spend_threshold: float = 20000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into retail and wholesale based on customer behavior."""
    summary = df.groupby('Customer ID').agg(
        n_orders=('Invoice', 'nunique'),
        total_spend=('Revenue', 'sum'),
    )
    wholesale_ids = summary[
        (summary['n_orders'] > order_threshold) |
        (summary['total_spend'] > spend_threshold)
    ].index
    wholesale = df[df['Customer ID'].isin(wholesale_ids)].copy()
    retail = df[~df['Customer ID'].isin(wholesale_ids)].copy()
    return retail, wholesale


def drop_non_positive_spend_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customers whose lifetime Revenue is <= 0."""
    spend = df.groupby('Customer ID')['Revenue'].sum()
    keep = spend[spend > 0].index
    return df[df['Customer ID'].isin(keep)].copy()

def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type ID columns to string for storage consistency."""
    df = df.copy()
    df['StockCode'] = df['StockCode'].astype(str)
    df['Invoice'] = df['Invoice'].astype(str)
    return df


# ---------------- Orchestrator ----------------

def clean_transactions(
    df: pd.DataFrame,
    country: str = "United Kingdom",
    wholesale_order_threshold: int = 60,
    wholesale_spend_threshold: float = 20000.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Returns (retail, wholesale) — both as cleaned DataFrames.
    """
    def _log(step_name: str, before: int, after: int):
        if verbose:
            print(f"  {step_name}: {before:,} -> {after:,} (dropped {before - after:,})")

    if verbose:
        print(f"Starting cleaning: {len(df):,} rows")

    # Single-DataFrame steps, applied in order
    steps = [
        ("Drop missing Customer ID", drop_missing_customer_id),
        ("Drop cancellation invoices", drop_cancellation_invoices),
        ("Drop non-positive Quantity", drop_non_positive_quantity),
        ("Drop non-positive Price", drop_non_positive_price),
        ("Drop non-product stock codes", drop_non_product_stockcodes),
        ("Filter to country", lambda d: filter_country(d, country)),
        ("Cast Customer ID to int", cast_customer_id_to_int),
        ("Add Revenue column", add_revenue_column),
        ("Normalize string columns", normalize_string_columns),   # ← new
    ]
    for name, fn in steps:
        before = len(df)
        df = fn(df)
        _log(name, before, len(df))

    # Split
    retail, wholesale = split_wholesale_retail(
        df, wholesale_order_threshold, wholesale_spend_threshold
    )
    if verbose:
        n_retail = retail['Customer ID'].nunique()
        n_wholesale = wholesale['Customer ID'].nunique()
        print(f"  Split: {n_wholesale} wholesale customers ({len(wholesale):,} rows), "
              f"{n_retail} retail customers ({len(retail):,} rows)")

    # Final retail safety filter
    before = retail['Customer ID'].nunique()
    retail = drop_non_positive_spend_customers(retail)
    after = retail['Customer ID'].nunique()
    if verbose:
        print(f"  Drop non-positive-spend retail customers: {before} -> {after} customers")
        print(f"Final retail: {len(retail):,} rows, {after:,} customers")
        print(f"Final wholesale: {len(wholesale):,} rows, {wholesale['Customer ID'].nunique():,} customers")

    return retail, wholesale