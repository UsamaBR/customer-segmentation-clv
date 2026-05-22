"""
CLV modeling using BG/NBD and Gamma-Gamma.

Fits two models:
  - BG/NBD predicts future purchase counts
  - Gamma-Gamma predicts average transaction value

Their product gives expected revenue over a future horizon.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data


def build_clv_summary(
    transactions: pd.DataFrame,
    observation_period_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate transactions into the format lifetimes expects.

    Output columns:
      - frequency: number of REPEAT purchases (first purchase doesn't count)
      - recency: age of customer at last purchase, in days
      - T: age of customer at observation end, in days
      - monetary_value: average value of REPEAT transactions

    Parameters
    ----------
    transactions : pd.DataFrame
        Must have columns: Customer ID, InvoiceDate, Invoice, Revenue.
    observation_period_end : pd.Timestamp, optional
        End of the observation window. Defaults to last InvoiceDate.

    Returns
    -------
    pd.DataFrame indexed by Customer ID with frequency/recency/T/monetary_value.
    """
    if observation_period_end is None:
        observation_period_end = transactions['InvoiceDate'].max()

    summary = summary_data_from_transaction_data(
        transactions,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='Revenue',
        observation_period_end=observation_period_end,
        freq='D',
    )
    return summary


def fit_bgnbd(summary: pd.DataFrame, penalizer: float = 0.001) -> BetaGeoFitter:
    """Fit BG/NBD on frequency, recency, T."""
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    return bgf


def fit_gamma_gamma(summary: pd.DataFrame, penalizer: float = 0.001) -> GammaGammaFitter:
    """Fit Gamma-Gamma on repeat customers only.

    Gamma-Gamma requires frequency > 0 and monetary_value > 0,
    so we filter to repeat customers.
    """
    repeat = summary[(summary['frequency'] > 0) & (summary['monetary_value'] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(repeat['frequency'], repeat['monetary_value'])
    return ggf


def predict_clv(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    months: int = 6,
    discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Combine fitted models to predict CLV per customer.

    Returns a DataFrame indexed by Customer ID with:
      - predicted_purchases_<months>m: BG/NBD's forecast
      - predicted_aov: Gamma-Gamma's per-customer average transaction value
      - p_alive: probability customer is still active
      - clv_<months>m: combined expected revenue over horizon

    For non-repeat customers (frequency=0), Gamma-Gamma can't estimate AOV;
    we fall back to their observed monetary_value (which is 0 by lifetimes
    convention) — they get CLV = 0.
    """
    result = summary.copy()

    # BG/NBD predictions
    result[f'predicted_purchases_{months}m'] = bgf.predict(
        t=months * 30,
        frequency=summary['frequency'],
        recency=summary['recency'],
        T=summary['T'],
    )
    result['p_alive'] = bgf.conditional_probability_alive(
        frequency=summary['frequency'],
        recency=summary['recency'],
        T=summary['T'],
    )

    # Gamma-Gamma: only valid for repeat customers
    repeat_mask = (summary['frequency'] > 0) & (summary['monetary_value'] > 0)
    result['predicted_aov'] = 0.0
    result.loc[repeat_mask, 'predicted_aov'] = ggf.conditional_expected_average_profit(
        summary.loc[repeat_mask, 'frequency'],
        summary.loc[repeat_mask, 'monetary_value'],
    )

    # Combined CLV using lifetimes' utility (handles discounting)
    result[f'clv_{months}m'] = 0.0
    result.loc[repeat_mask, f'clv_{months}m'] = ggf.customer_lifetime_value(
        bgf,
        summary.loc[repeat_mask, 'frequency'],
        summary.loc[repeat_mask, 'recency'],
        summary.loc[repeat_mask, 'T'],
        summary.loc[repeat_mask, 'monetary_value'],
        time=months,
        discount_rate=discount_rate,
        freq='D',
    )

    return result


def validate_bgnbd_holdout(
    transactions: pd.DataFrame,
    calibration_end: pd.Timestamp,
    observation_end: pd.Timestamp,
    penalizer: float = 0.001,
) -> Tuple[pd.DataFrame, BetaGeoFitter]:
    """Calibration/holdout validation for BG/NBD.

    Fit on transactions through `calibration_end`, predict purchases in
    `(calibration_end, observation_end]`, compare to actual.
    """
    from lifetimes.utils import calibration_and_holdout_data

    cal_holdout = calibration_and_holdout_data(
        transactions,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        calibration_period_end=calibration_end,
        observation_period_end=observation_end,
        freq='D',
        monetary_value_col='Revenue',
    )

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(
        cal_holdout['frequency_cal'],
        cal_holdout['recency_cal'],
        cal_holdout['T_cal'],
    )

    horizon_days = (observation_end - calibration_end).days
    cal_holdout['predicted_purchases_holdout'] = bgf.predict(
        t=horizon_days,
        frequency=cal_holdout['frequency_cal'],
        recency=cal_holdout['recency_cal'],
        T=cal_holdout['T_cal'],
    )

    return cal_holdout, bgf