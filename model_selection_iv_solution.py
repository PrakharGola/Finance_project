from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SEPARATOR = "||"
RANDOM_SEEDS = [7, 11, 21, 42, 77, 101, 2026]
EPS = 1.0e-8
EXPIRY_CLOSE_HOUR = 15
EXPIRY_CLOSE_MINUTE = 30

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

BASE_METHODS = [
    "row_type_mean_baseline",
    "cross_sectional_interpolation",
    "past_time_series_model",
    "ml_combined_model",
    "ml_separate_ce_pe_model",
    "expiry_day_special_model",
]
BLEND_COMPONENTS = [
    "cross_sectional_interpolation",
    "past_time_series_model",
    "ml_separate_ce_pe_model",
    "row_type_mean_baseline",
]


@dataclass(frozen=True)
class OptionMeta:
    column: str
    expiry_token: str
    expiry: pd.Timestamp
    strike: int
    option_type: str


@dataclass
class ExperimentBundle:
    seed: int
    strategy: str
    train_df: pd.DataFrame
    hidden_mask: pd.DataFrame
    public_mask: pd.DataFrame
    private_mask: pd.DataFrame
    base_predictions: Dict[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust NIFTY IV model selection with pseudo public/private validation."
    )
    parser.add_argument("--input", default="data/dataset.csv", help="Input dataset CSV.")

    parser.add_argument("--output", default="outputs/submission.csv", help="Kaggle submission CSV.")

    parser.add_argument(
        "--filled-output",
        default="outputs/filled_dataset.csv",
        help="Filled dataset CSV.",
    )

    parser.add_argument(
        "--leaderboard-output",
        default="outputs/validation_leaderboard.csv",
        help="Validation leaderboard CSV.",
    )

    parser.add_argument(
        "--config-output",
        default="outputs/best_model_config.json",
        help="Selected model configuration JSON.",
    )
    parser.add_argument(
        "--leaderboard-output",
        default="validation_leaderboard.csv",
        help="Validation leaderboard CSV.",
    )
    parser.add_argument(
        "--config-output",
        default="best_model_config.json",
        help="Selected model configuration JSON.",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.15,
        help="Fraction of observed cells hidden in random-style validation masks.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.10,
        help="Blend grid step. Use 0.05 for a larger search.",
    )
    return parser.parse_args()


def parse_expiry_token(token: str) -> pd.Timestamp:
    match = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", token)
    if not match:
        raise ValueError(f"Could not parse expiry token: {token}")
    return pd.Timestamp(
        year=2000 + int(match.group(3)),
        month=MONTHS[match.group(2)],
        day=int(match.group(1)),
    )


def parse_option_ticker(column: str) -> OptionMeta:
    match = re.fullmatch(r"NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)", column)
    if not match:
        raise ValueError(f"Unsupported option column: {column}")
    expiry_token, strike, option_type = match.groups()
    return OptionMeta(
        column=column,
        expiry_token=expiry_token,
        expiry=parse_expiry_token(expiry_token),
        strike=int(strike),
        option_type=option_type,
    )


def load_data(path: str | Path) -> Tuple[pd.DataFrame, List[str], Dict[str, OptionMeta]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    if "datetime" not in df.columns or "underlying_price" not in df.columns:
        raise ValueError("dataset.csv must contain datetime and underlying_price columns.")
    option_cols = [col for col in df.columns if col.startswith("NIFTY")]
    if not option_cols:
        raise ValueError("No NIFTY option columns found.")
    meta = {col: parse_option_ticker(col) for col in option_cols}
    df = df.copy()
    df["_original_index"] = np.arange(len(df))
    df["_parsed_datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")
    df = df.sort_values("_parsed_datetime").reset_index(drop=True)
    return df, option_cols, meta


def type_columns(option_cols: Iterable[str], meta: Dict[str, OptionMeta]) -> Dict[str, List[str]]:
    return {
        option_type: sorted(
            [col for col in option_cols if meta[col].option_type == option_type],
            key=lambda col: meta[col].strike,
        )
        for option_type in ["CE", "PE"]
    }


def expiry_close(meta: Dict[str, OptionMeta]) -> pd.Timestamp:
    expiries = sorted({item.expiry for item in meta.values()})
    if len(expiries) != 1:
        warnings.warn("Multiple expiries found; using first expiry for expiry features.")
    return expiries[0] + pd.Timedelta(hours=EXPIRY_CLOSE_HOUR, minutes=EXPIRY_CLOSE_MINUTE)


def datetime_features(df: pd.DataFrame, meta: Dict[str, OptionMeta]) -> pd.DataFrame:
    dt = df["_parsed_datetime"]
    expiry_ts = expiry_close(meta)
    out = pd.DataFrame(index=df.index)
    out["date"] = dt.dt.date
    out["day_index"] = (dt.dt.normalize() - dt.min().normalize()).dt.days.astype(float)
    out["hour"] = dt.dt.hour.astype(float)
    out["minute"] = dt.dt.minute.astype(float)
    out["minute_of_day"] = (dt.dt.hour * 60 + dt.dt.minute).astype(float)
    out["minutes_to_expiry"] = np.maximum(
        (expiry_ts - dt).dt.total_seconds().to_numpy(dtype=float) / 60.0,
        0.0,
    )
    out["days_to_expiry"] = out["minutes_to_expiry"] / 1440.0
    out["is_expiry_day"] = (dt.dt.normalize() == expiry_ts.normalize()).astype(bool)
    out["time_to_expiry_fraction"] = out["minutes_to_expiry"] / (365.0 * 1440.0)
    return out


def finite_mean(values: np.ndarray, default: float = np.nan) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def finite_median(values: np.ndarray, default: float = np.nan) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


def past_market_median(df: pd.DataFrame, option_cols: List[str], default: float = 0.20) -> np.ndarray:
    values = df[option_cols].to_numpy(dtype=float)
    output = np.full(len(df), default, dtype=float)
    seen: List[float] = []
    for row_idx, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            seen.extend(finite.tolist())
        if seen:
            output[row_idx] = float(np.median(seen))
    return output


def fill_remaining(
    pred: pd.DataFrame,
    train_df: pd.DataFrame,
    option_cols: List[str],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    out = pred.copy()
    row_all = train_df[option_cols].median(axis=1).ffill().fillna(0.20)
    for option_type, cols in type_cols.items():
        fallback = train_df[cols].median(axis=1).fillna(row_all).ffill().fillna(0.20)
        fallback_matrix = np.repeat(fallback.to_numpy(dtype=float)[:, None], len(cols), axis=1)
        current = out[cols].to_numpy(dtype=float)
        out.loc[:, cols] = np.where(np.isfinite(current), current, fallback_matrix)
    return out[option_cols].clip(lower=0.0001)


def interp_extrap(x_known: np.ndarray, y_known: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    order = np.argsort(x_known)
    x = x_known[order].astype(float)
    y = y_known[order].astype(float)
    target = x_target.astype(float)
    if len(x) == 1:
        return np.full(len(target), y[0], dtype=float)
    pred = np.interp(target, x, y)
    left = target < x[0]
    if left.any():
        slope = (y[1] - y[0]) / max(x[1] - x[0], EPS)
        pred[left] = y[0] + slope * (target[left] - x[0])
    right = target > x[-1]
    if right.any():
        slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], EPS)
        pred[right] = y[-1] + slope * (target[right] - x[-1])
    return pred


def poly_smile(
    strikes: np.ndarray,
    values: np.ndarray,
    observed: np.ndarray,
    spot: float,
    expiry_day: bool,
) -> np.ndarray | None:
    if int(observed.sum()) < 4:
        return None
    try:
        x_obs = np.log(strikes[observed] / spot)
        y_obs = values[observed]
        degree = min(3, len(y_obs) - 1)
        coef = np.polyfit(x_obs, y_obs, degree)
        pred = np.polyval(coef, np.log(strikes / spot))
        lo = float(np.nanmin(y_obs))
        hi = float(np.nanmax(y_obs))
        upper = hi * (3.6 if expiry_day else 2.2)
        return np.clip(pred, max(0.0001, lo * 0.20), max(upper, lo + 0.001))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def row_type_mean_baseline(
    train_df: pd.DataFrame,
    option_cols: List[str],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    pred = pd.DataFrame(index=train_df.index, columns=option_cols, dtype=float)
    all_median = train_df[option_cols].median(axis=1).ffill().fillna(0.20)
    for option_type, cols in type_cols.items():
        type_values = train_df[cols]
        row_mean = type_values.mean(axis=1).fillna(all_median)
        row_median = type_values.median(axis=1).fillna(row_mean)
        fallback = 0.55 * row_median + 0.45 * row_mean
        pred.loc[:, cols] = np.repeat(fallback.to_numpy()[:, None], len(cols), axis=1)
        pred.loc[:, cols] = pred.loc[:, cols].where(type_values.isna(), type_values)
    return pred[option_cols].clip(lower=0.0001)


def cross_sectional_interpolation(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    dtf = datetime_features(train_df, meta)
    pred = pd.DataFrame(np.nan, index=train_df.index, columns=option_cols, dtype=float)
    for row_idx, row in train_df.iterrows():
        spot = float(row["underlying_price"])
        expiry_day = bool(dtf.at[row_idx, "is_expiry_day"])
        for option_type, cols in type_cols.items():
            strikes = np.array([meta[col].strike for col in cols], dtype=float)
            values = row[cols].to_numpy(dtype=float)
            observed = np.isfinite(values)
            if observed.sum() >= 2:
                linear = interp_extrap(strikes[observed], values[observed], strikes)
                poly = poly_smile(strikes, values, observed, spot, expiry_day)
                estimates = linear if poly is None else 0.70 * linear + 0.30 * poly
                estimates[observed] = values[observed]
            elif observed.sum() == 1:
                estimates = np.full(len(cols), float(values[observed][0]), dtype=float)
            else:
                estimates = np.full(len(cols), np.nan, dtype=float)
            pred.loc[row_idx, cols] = estimates
    return fill_remaining(pred, train_df, option_cols, type_cols)


def past_time_series_model(
    train_df: pd.DataFrame,
    option_cols: List[str],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    pred = pd.DataFrame(index=train_df.index, columns=option_cols, dtype=float)
    
    for col in option_cols:
        series = train_df[col].astype(float)
        ffill = series.ffill()
        rolling_mean = ffill.rolling(window=10, min_periods=1).mean()
        rolling_median = ffill.rolling(window=14, min_periods=1).median()
        ewma = ffill.ewm(span=8, adjust=False, min_periods=1).mean()
        pred[col] = 0.40 * ffill + 0.20 * rolling_mean + 0.20 * rolling_median + 0.20 * ewma
        pred.loc[series.notna(), col] = series[series.notna()]
    return fill_remaining(pred, train_df, option_cols, type_cols)


def surface_context(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    past = past_market_median(train_df, option_cols)
    spot = train_df["underlying_price"].to_numpy(dtype=float)
    context: Dict[str, Dict[str, np.ndarray]] = {}
    for option_type, cols in type_cols.items():
        values = train_df[cols].to_numpy(dtype=float)
        strikes = np.array([meta[col].strike for col in cols], dtype=float)
        n_rows, n_cols = values.shape
        data = {
            "mean": np.zeros(n_rows),
            "median": np.zeros(n_rows),
            "std": np.zeros(n_rows),
            "min": np.zeros(n_rows),
            "max": np.zeros(n_rows),
            "count": np.zeros(n_rows),
            "lower": np.zeros((n_rows, n_cols)),
            "upper": np.zeros((n_rows, n_cols)),
            "spread": np.zeros((n_rows, n_cols)),
            "atm_rank": np.zeros((n_rows, n_cols)),
        }
        for row_idx in range(n_rows):
            row = values[row_idx]
            finite = row[np.isfinite(row)]
            all_fallback = finite_median(train_df.loc[row_idx, option_cols].to_numpy(dtype=float), past[row_idx])
            fallback = finite_median(finite, all_fallback)
            data["count"][row_idx] = float(finite.size)
            data["mean"][row_idx] = finite_mean(finite, fallback)
            data["median"][row_idx] = finite_median(finite, fallback)
            data["std"][row_idx] = float(np.std(finite)) if finite.size > 1 else 0.0
            data["min"][row_idx] = float(np.min(finite)) if finite.size else fallback
            data["max"][row_idx] = float(np.max(finite)) if finite.size else fallback

            lower = np.full(n_cols, fallback)
            last = np.nan
            for col_idx in range(n_cols):
                lower[col_idx] = fallback if not np.isfinite(last) else last
                if np.isfinite(row[col_idx]):
                    last = row[col_idx]
            upper = np.full(n_cols, fallback)
            last = np.nan
            for col_idx in range(n_cols - 1, -1, -1):
                upper[col_idx] = fallback if not np.isfinite(last) else last
                if np.isfinite(row[col_idx]):
                    last = row[col_idx]
            data["lower"][row_idx] = lower
            data["upper"][row_idx] = upper
            data["spread"][row_idx] = upper - lower

            distances = np.abs(strikes - spot[row_idx])
            order = np.argsort(distances)
            ranks = np.empty(n_cols, dtype=float)
            ranks[order] = np.arange(1, n_cols + 1, dtype=float)
            data["atm_rank"][row_idx] = ranks
        context[option_type] = data
    return context


def previous_contract_features(train_df: pd.DataFrame, option_cols: List[str]) -> Dict[str, pd.DataFrame]:
    previous = pd.DataFrame(index=train_df.index, columns=option_cols, dtype=float)
    rolling = pd.DataFrame(index=train_df.index, columns=option_cols, dtype=float)
    for col in option_cols:
        shifted = train_df[col].astype(float).ffill().shift(1)
        previous[col] = shifted
        rolling[col] = shifted.rolling(window=12, min_periods=1).median()
    return {"previous_iv": previous, "rolling_past_iv": rolling}


def long_feature_frame(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    dtf = datetime_features(train_df, meta)
    context = surface_context(train_df, option_cols, meta, type_cols)
    past_feats = previous_contract_features(train_df, option_cols)
    type_pos = {col: pos for cols in type_cols.values() for pos, col in enumerate(cols)}
    records: List[Dict[str, float | int | str]] = []
    for row_idx in range(len(train_df)):
        spot = float(train_df.at[row_idx, "underlying_price"])
        ce_ctx = context["CE"]
        pe_ctx = context["PE"]
        for col_idx, col in enumerate(option_cols):
            item = meta[col]
            option_type = item.option_type
            pos = type_pos[col]
            type_ctx = context[option_type]
            moneyness = item.strike / spot
            log_m = math.log(max(moneyness, EPS))
            distance = item.strike - spot
            previous_iv = past_feats["previous_iv"].at[row_idx, col]
            rolling_past_iv = past_feats["rolling_past_iv"].at[row_idx, col]
            fallback = float(type_ctx["median"][row_idx])
            records.append(
                {
                    "row_index": row_idx,
                    "column": col,
                    "col_index": col_idx,
                    "option_type": option_type,
                    "iv": train_df.at[row_idx, col],
                    "strike_scaled": (item.strike - 25200.0) / 1000.0,
                    "option_type_code": 1.0 if option_type == "CE" else 0.0,
                    "underlying_scaled": (spot - 25200.0) / 1000.0,
                    "moneyness": moneyness,
                    "log_moneyness_scaled": log_m * 100.0,
                    "log_moneyness_sq": (log_m * 100.0) ** 2,
                    "distance_scaled": distance / 1000.0,
                    "abs_distance_scaled": abs(distance) / 1000.0,
                    "minute_of_day_scaled": float(dtf.at[row_idx, "minute_of_day"]) / 1440.0,
                    "days_to_expiry_scaled": float(dtf.at[row_idx, "days_to_expiry"]) / 30.0,
                    "minutes_to_expiry_scaled": float(dtf.at[row_idx, "minutes_to_expiry"]) / 30000.0,
                    "is_expiry_day": float(dtf.at[row_idx, "is_expiry_day"]),
                    "ce_mean": float(ce_ctx["mean"][row_idx]),
                    "ce_median": float(ce_ctx["median"][row_idx]),
                    "ce_std": float(ce_ctx["std"][row_idx]),
                    "pe_mean": float(pe_ctx["mean"][row_idx]),
                    "pe_median": float(pe_ctx["median"][row_idx]),
                    "pe_std": float(pe_ctx["std"][row_idx]),
                    "row_type_mean": float(type_ctx["mean"][row_idx]),
                    "row_type_median": fallback,
                    "row_type_std": float(type_ctx["std"][row_idx]),
                    "nearest_lower_iv": float(type_ctx["lower"][row_idx, pos]),
                    "nearest_upper_iv": float(type_ctx["upper"][row_idx, pos]),
                    "lower_upper_spread": float(type_ctx["spread"][row_idx, pos]),
                    "observed_count_scaled": float(type_ctx["count"][row_idx]) / max(len(type_cols[option_type]), 1),
                    "atm_rank_scaled": float(type_ctx["atm_rank"][row_idx, pos]) / max(len(type_cols[option_type]), 1),
                    "previous_iv": fallback if not np.isfinite(previous_iv) else float(previous_iv),
                    "rolling_past_iv": fallback if not np.isfinite(rolling_past_iv) else float(rolling_past_iv),
                }
            )
    frame = pd.DataFrame.from_records(records)
    feature_cols = ml_feature_columns()
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def ml_feature_columns() -> List[str]:
    return [
        "strike_scaled",
        "option_type_code",
        "underlying_scaled",
        "moneyness",
        "log_moneyness_scaled",
        "log_moneyness_sq",
        "distance_scaled",
        "abs_distance_scaled",
        "minute_of_day_scaled",
        "days_to_expiry_scaled",
        "minutes_to_expiry_scaled",
        "is_expiry_day",
        "ce_mean",
        "ce_median",
        "ce_std",
        "pe_mean",
        "pe_median",
        "pe_std",
        "row_type_mean",
        "row_type_median",
        "row_type_std",
        "nearest_lower_iv",
        "nearest_upper_iv",
        "lower_upper_spread",
        "observed_count_scaled",
        "atm_rank_scaled",
        "previous_iv",
        "rolling_past_iv",
    ]


def fit_ridge_predict(
    feature_df: pd.DataFrame,
    fit_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.30,
) -> np.ndarray:
    features = ml_feature_columns()
    x_all = feature_df[features].to_numpy(dtype=float)
    y_all = feature_df["iv"].to_numpy(dtype=float)
    fallback = np.maximum(feature_df["row_type_median"].to_numpy(dtype=float), 0.0001)
    if int(fit_mask.sum()) < len(features) + 5:
        return fallback[pred_mask]
    x_fit = x_all[fit_mask]
    x_pred = x_all[pred_mask]
    y_fit = np.log(np.clip(y_all[fit_mask], 0.0001, None))
    mu = x_fit.mean(axis=0)
    sigma = x_fit.std(axis=0)
    sigma[sigma < EPS] = 1.0
    x_fit = (x_fit - mu) / sigma
    x_pred = (x_pred - mu) / sigma
    x_fit = np.column_stack([np.ones(len(x_fit)), x_fit])
    x_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
    penalty = np.eye(x_fit.shape[1]) * alpha
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(x_fit.T @ x_fit + penalty, x_fit.T @ y_fit)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(x_fit.T @ x_fit + penalty, x_fit.T @ y_fit, rcond=None)[0]
    pred = np.exp(x_pred @ beta)
    return np.clip(pred, 0.0001, None)


def ml_predictions(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
    separate: bool,
) -> pd.DataFrame:
    features = long_feature_frame(train_df, option_cols, meta, type_cols)
    row_index = features["row_index"].to_numpy(dtype=int)
    columns = features["column"].to_numpy(dtype=str)
    observed = np.isfinite(features["iv"].to_numpy(dtype=float))
    pred_all = np.maximum(features["row_type_median"].to_numpy(dtype=float), 0.0001)

    if separate:
        for option_type in ["CE", "PE"]:
            type_mask = features["option_type"].to_numpy(dtype=str) == option_type
            pred_mask = type_mask
            fit_mask = observed & type_mask
            pred_all[pred_mask] = fit_ridge_predict(features, fit_mask, pred_mask)
    else:
        pred_all[:] = fit_ridge_predict(features, observed, np.ones(len(features), dtype=bool))

    pred = pd.DataFrame(index=train_df.index, columns=option_cols, dtype=float)
    for row_idx, col, value in zip(row_index, columns, pred_all):
        pred.at[row_idx, col] = float(value)
    for col in option_cols:
        observed_col = train_df[col].notna()
        pred.loc[observed_col, col] = train_df.loc[observed_col, col]
    return fill_remaining(pred, train_df, option_cols, type_cols)


def fast_ml_prediction_frames(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = fast_ml_feature_frame(train_df, option_cols, meta, type_cols)
    observed = np.isfinite(feature_df["iv"].to_numpy(dtype=float))
    combined_values = fit_ridge_predict(feature_df, observed, np.ones(len(feature_df), dtype=bool), alpha=0.30)

    separate_values = np.maximum(feature_df["row_type_median"].to_numpy(dtype=float), 0.0001)
    option_types = feature_df["option_type"].to_numpy(dtype=str)
    for option_type in ["CE", "PE"]:
        pred_mask = option_types == option_type
        fit_mask = observed & pred_mask
        separate_values[pred_mask] = fit_ridge_predict(feature_df, fit_mask, pred_mask, alpha=0.35)

    combined = reshape_long_predictions(feature_df, combined_values, train_df, option_cols, type_cols)
    separate = reshape_long_predictions(feature_df, separate_values, train_df, option_cols, type_cols)
    return combined, separate


def fast_ml_feature_frame(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    n_rows = len(train_df)
    n_cols = len(option_cols)
    dtf = datetime_features(train_df, meta)
    spot = train_df["underlying_price"].to_numpy(dtype=float)
    strikes = np.array([meta[col].strike for col in option_cols], dtype=float)
    option_type = np.array([meta[col].option_type for col in option_cols], dtype=object)
    is_call = (option_type == "CE").astype(float)
    row_index = np.repeat(np.arange(n_rows), n_cols)
    columns = np.tile(np.array(option_cols, dtype=object), n_rows)
    tiled_strikes = np.tile(strikes, n_rows)
    tiled_spot = np.repeat(spot, n_cols)
    tiled_type = np.tile(option_type, n_rows)

    values = train_df[option_cols].to_numpy(dtype=float)
    iv = values.reshape(-1)
    moneyness = tiled_strikes / tiled_spot
    log_m = np.log(np.maximum(moneyness, EPS))
    distance = tiled_strikes - tiled_spot

    stat_map: Dict[str, np.ndarray] = {}
    lower_map: Dict[str, np.ndarray] = {}
    upper_map: Dict[str, np.ndarray] = {}
    count_map: Dict[str, np.ndarray] = {}
    for opt_type, cols in type_cols.items():
        type_values = train_df[cols].astype(float)
        row_median = type_values.median(axis=1).ffill().fillna(train_df[option_cols].median(axis=1)).fillna(0.20)
        stat_map[f"{opt_type}_mean"] = type_values.mean(axis=1).fillna(row_median).to_numpy(dtype=float)
        stat_map[f"{opt_type}_median"] = row_median.to_numpy(dtype=float)
        stat_map[f"{opt_type}_std"] = type_values.std(axis=1).fillna(0.0).to_numpy(dtype=float)
        count_map[opt_type] = type_values.notna().sum(axis=1).to_numpy(dtype=float) / max(len(cols), 1)

        fallback = np.repeat(row_median.to_numpy(dtype=float)[:, None], len(cols), axis=1)
        lower = type_values.shift(axis=1).ffill(axis=1).to_numpy(dtype=float)
        upper = type_values.shift(axis=1, periods=-1).bfill(axis=1).to_numpy(dtype=float)
        lower = np.where(np.isfinite(lower), lower, fallback)
        upper = np.where(np.isfinite(upper), upper, fallback)
        for pos, col in enumerate(cols):
            lower_map[col] = lower[:, pos]
            upper_map[col] = upper[:, pos]

    previous = train_df[option_cols].ffill().shift(1)
    rolling = previous.rolling(window=12, min_periods=1).median()
    prev_values = previous.to_numpy(dtype=float).reshape(-1)
    rolling_values = rolling.to_numpy(dtype=float).reshape(-1)

    row_type_mean = np.zeros(n_rows * n_cols, dtype=float)
    row_type_median = np.zeros(n_rows * n_cols, dtype=float)
    row_type_std = np.zeros(n_rows * n_cols, dtype=float)
    observed_count = np.zeros(n_rows * n_cols, dtype=float)
    lower_iv = np.zeros(n_rows * n_cols, dtype=float)
    upper_iv = np.zeros(n_rows * n_cols, dtype=float)
    for opt_type in ["CE", "PE"]:
        mask = tiled_type == opt_type
        row_type_mean[mask] = np.repeat(stat_map[f"{opt_type}_mean"], n_cols)[mask]
        row_type_median[mask] = np.repeat(stat_map[f"{opt_type}_median"], n_cols)[mask]
        row_type_std[mask] = np.repeat(stat_map[f"{opt_type}_std"], n_cols)[mask]
        observed_count[mask] = np.repeat(count_map[opt_type], n_cols)[mask]
    for col_idx, col in enumerate(option_cols):
        positions = np.arange(col_idx, n_rows * n_cols, n_cols)
        lower_iv[positions] = lower_map[col]
        upper_iv[positions] = upper_map[col]

    fallback = np.maximum(row_type_median, 0.0001)
    prev_values = np.where(np.isfinite(prev_values), prev_values, fallback)
    rolling_values = np.where(np.isfinite(rolling_values), rolling_values, fallback)

    frame = pd.DataFrame(
        {
            "row_index": row_index,
            "column": columns,
            "option_type": tiled_type,
            "iv": iv,
            "strike_scaled": (tiled_strikes - 25200.0) / 1000.0,
            "option_type_code": np.tile(is_call, n_rows),
            "underlying_scaled": (tiled_spot - 25200.0) / 1000.0,
            "moneyness": moneyness,
            "log_moneyness_scaled": log_m * 100.0,
            "log_moneyness_sq": (log_m * 100.0) ** 2,
            "distance_scaled": distance / 1000.0,
            "abs_distance_scaled": np.abs(distance) / 1000.0,
            "minute_of_day_scaled": np.repeat(dtf["minute_of_day"].to_numpy(dtype=float) / 1440.0, n_cols),
            "days_to_expiry_scaled": np.repeat(dtf["days_to_expiry"].to_numpy(dtype=float) / 30.0, n_cols),
            "minutes_to_expiry_scaled": np.repeat(dtf["minutes_to_expiry"].to_numpy(dtype=float) / 30000.0, n_cols),
            "is_expiry_day": np.repeat(dtf["is_expiry_day"].to_numpy(dtype=float), n_cols),
            "ce_mean": np.repeat(stat_map["CE_mean"], n_cols),
            "ce_median": np.repeat(stat_map["CE_median"], n_cols),
            "ce_std": np.repeat(stat_map["CE_std"], n_cols),
            "pe_mean": np.repeat(stat_map["PE_mean"], n_cols),
            "pe_median": np.repeat(stat_map["PE_median"], n_cols),
            "pe_std": np.repeat(stat_map["PE_std"], n_cols),
            "row_type_mean": row_type_mean,
            "row_type_median": row_type_median,
            "row_type_std": row_type_std,
            "nearest_lower_iv": lower_iv,
            "nearest_upper_iv": upper_iv,
            "lower_upper_spread": upper_iv - lower_iv,
            "observed_count_scaled": observed_count,
            "atm_rank_scaled": np.tile(
                np.argsort(np.argsort(np.abs(strikes[None, :] - spot[:, None]), axis=1), axis=1).reshape(-1)
                / max(n_cols, 1),
                1,
            ),
            "previous_iv": prev_values,
            "rolling_past_iv": rolling_values,
        }
    )
    feature_cols = ml_feature_columns()
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def reshape_long_predictions(
    feature_df: pd.DataFrame,
    values: np.ndarray,
    train_df: pd.DataFrame,
    option_cols: List[str],
    type_cols: Dict[str, List[str]],
) -> pd.DataFrame:
    arr = values.reshape(len(train_df), len(option_cols))
    pred = pd.DataFrame(arr, index=train_df.index, columns=option_cols)
    for col in option_cols:
        observed = train_df[col].notna()
        pred.loc[observed, col] = train_df.loc[observed, col]
    return fill_remaining(pred, train_df, option_cols, type_cols)


def expiry_day_special_model(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
    cross: pd.DataFrame,
    time: pd.DataFrame,
    row: pd.DataFrame,
) -> pd.DataFrame:
    dtf = datetime_features(train_df, meta)
    pred = 0.82 * cross[option_cols] + 0.10 * time[option_cols] + 0.08 * row[option_cols]
    expiry_rows = dtf["is_expiry_day"].to_numpy(dtype=bool)
    pred.loc[expiry_rows, option_cols] = (
        0.94 * cross.loc[expiry_rows, option_cols]
        + 0.03 * time.loc[expiry_rows, option_cols]
        + 0.03 * row.loc[expiry_rows, option_cols]
    )
    return fill_remaining(pred, train_df, option_cols, type_cols)


def clip_predictions(
    pred: pd.DataFrame,
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
    multiplier: float = 3.0,
) -> pd.DataFrame:
    out = pred.copy()
    dtf = datetime_features(train_df, meta)
    expiry_day = dtf["is_expiry_day"].to_numpy(dtype=bool)
    for option_type, cols in type_cols.items():
        row_max = train_df[cols].max(axis=1).ffill().fillna(0.25).to_numpy(dtype=float)
        seen_max = np.maximum.accumulate(row_max)
        factor = np.where(expiry_day, max(multiplier, 3.6), multiplier)
        upper = np.maximum(np.maximum(row_max, seen_max) * factor, 0.05)
        clipped = np.clip(out[cols].to_numpy(dtype=float), 0.0001, upper[:, None])
        out.loc[:, cols] = clipped
    return out


def build_base_predictions(
    train_df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    row = row_type_mean_baseline(train_df, option_cols, type_cols)
    cross = cross_sectional_interpolation(train_df, option_cols, meta, type_cols)
    time = past_time_series_model(train_df, option_cols, type_cols)
    ml_combined, ml_separate = fast_ml_prediction_frames(train_df, option_cols, meta, type_cols)
    expiry = expiry_day_special_model(train_df, option_cols, meta, type_cols, cross, time, row)
    bases = {
        "row_type_mean_baseline": row,
        "cross_sectional_interpolation": cross,
        "past_time_series_model": time,
        "ml_combined_model": ml_combined,
        "ml_separate_ce_pe_model": ml_separate,
        "expiry_day_special_model": expiry,
    }
    return {
        name: clip_predictions(frame, train_df, option_cols, meta, type_cols, multiplier=3.2)
        for name, frame in bases.items()
    }


def make_empty_mask(df: pd.DataFrame, option_cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(False, index=df.index, columns=option_cols)


def choose_positions(
    mask: pd.DataFrame,
    positions: np.ndarray,
    selected_indices: np.ndarray,
    option_cols: List[str],
) -> None:
    for pos_idx in selected_indices:
        row_idx, col_idx = positions[pos_idx]
        mask.at[mask.index[row_idx], option_cols[col_idx]] = True


def validation_masks(
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    seed: int,
    mask_fraction: float,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    observed = df[option_cols].notna()
    real_missing = df[option_cols].isna()
    positions = np.argwhere(observed.to_numpy())
    masks: Dict[str, pd.DataFrame] = {}

    random_mask = make_empty_mask(df, option_cols)
    random_n = max(1, int(len(positions) * mask_fraction))
    choose_positions(random_mask, positions, rng.choice(len(positions), random_n, replace=False), option_cols)
    masks["random_cell_mask"] = random_mask

    public_private_like = make_empty_mask(df, option_cols)
    pp_n = max(1, int(len(positions) * min(mask_fraction * 1.25, 0.25)))
    choose_positions(public_private_like, positions, rng.choice(len(positions), pp_n, replace=False), option_cols)
    masks["public_private_like_mask"] = public_private_like

    pattern_mask = make_empty_mask(df, option_cols)
    for row_idx in range(len(df)):
        real_cols = [col for col in option_cols if real_missing.at[row_idx, col]]
        if not real_cols:
            continue
        for option_type in ["CE", "PE"]:
            missing_type_cols = [col for col in real_cols if meta[col].option_type == option_type]
            available = [
                col
                for col in option_cols
                if meta[col].option_type == option_type and observed.at[row_idx, col]
            ]
            if not missing_type_cols or not available:
                continue
            wanted = min(len(missing_type_cols), len(available))
            target_strikes = np.array([meta[col].strike for col in missing_type_cols], dtype=float)
            candidate_strikes = np.array([meta[col].strike for col in available], dtype=float)
            distances = np.min(np.abs(candidate_strikes[:, None] - target_strikes[None, :]), axis=1)
            weights = 1.0 / (1.0 + distances)
            weights = weights / weights.sum()
            chosen = rng.choice(available, size=wanted, replace=False, p=weights)
            pattern_mask.loc[row_idx, chosen] = True
    if int(pattern_mask.to_numpy().sum()) == 0:
        pattern_mask = random_mask.copy()
    masks["missing_pattern_like_mask"] = pattern_mask

    strikes = sorted({item.strike for item in meta.values()})
    strike_n = max(2, int(round(len(strikes) * mask_fraction)))
    chosen_strikes = set(rng.choice(strikes, strike_n, replace=False).tolist())
    strike_cols = [col for col in option_cols if meta[col].strike in chosen_strikes]
    strike_mask = make_empty_mask(df, option_cols)
    strike_mask.loc[:, strike_cols] = observed.loc[:, strike_cols]
    masks["strike_block_mask"] = strike_mask

    block_len = max(12, int(len(df) * mask_fraction))
    start = int(rng.integers(0, max(1, len(df) - block_len + 1)))
    time_mask = make_empty_mask(df, option_cols)
    time_mask.iloc[start : start + block_len, :] = observed.iloc[start : start + block_len, :]
    masks["time_block_mask"] = time_mask

    dtf = datetime_features(df, meta)
    expiry_rows = dtf["is_expiry_day"].to_numpy(dtype=bool)
    expiry_mask = make_empty_mask(df, option_cols)
    expiry_positions = np.argwhere(observed.loc[expiry_rows, option_cols].to_numpy())
    expiry_indices = observed.index[expiry_rows].to_numpy()
    if len(expiry_positions):
        expiry_n = max(1, int(len(expiry_positions) * max(mask_fraction, 0.20)))
        for local_row, col_idx in expiry_positions[rng.choice(len(expiry_positions), expiry_n, replace=False)]:
            expiry_mask.at[expiry_indices[local_row], option_cols[col_idx]] = True
    masks["expiry_day_mask"] = expiry_mask

    grouped_mask = make_empty_mask(df, option_cols)
    dates = pd.Series(dtf["date"].unique())
    chosen_dates = rng.choice(dates, size=max(1, min(3, len(dates) // 4)), replace=False)
    for chosen_date in chosen_dates:
        rows = np.where(dtf["date"].to_numpy() == chosen_date)[0]
        if len(rows) == 0:
            continue
        segment = max(3, len(rows) // 3)
        seg_start = int(rng.integers(0, max(1, len(rows) - segment + 1)))
        selected_rows = rows[seg_start : seg_start + segment]
        grouped_mask.iloc[selected_rows, :] = observed.iloc[selected_rows, :]
    masks["grouped_day_mask"] = grouped_mask

    return {name: mask for name, mask in masks.items() if int(mask.to_numpy().sum()) > 0}


def stable_strategy_offset(strategy: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(strategy))


def split_public_private(mask: pd.DataFrame, seed: int, strategy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + stable_strategy_offset(strategy))
    positions = np.argwhere(mask.to_numpy())
    public = make_empty_mask(mask, list(mask.columns))
    private = make_empty_mask(mask, list(mask.columns))
    if len(positions) == 0:
        return public, private
    selected = set(rng.choice(len(positions), size=max(1, int(len(positions) * 0.30)), replace=False).tolist())
    cols = list(mask.columns)
    for pos_idx, (row_idx, col_idx) in enumerate(positions):
        target = public if pos_idx in selected else private
        target.at[mask.index[row_idx], cols[col_idx]] = True
    return public, private


def apply_hidden_mask(df: pd.DataFrame, mask: pd.DataFrame, option_cols: List[str]) -> pd.DataFrame:
    train = df.copy()
    train.loc[:, option_cols] = train[option_cols].mask(mask)
    return train


def masked_values(frame: pd.DataFrame, mask: pd.DataFrame, option_cols: List[str]) -> np.ndarray:
    return frame[option_cols].to_numpy(dtype=float)[mask.to_numpy(dtype=bool)]


def masked_types(mask: pd.DataFrame, option_cols: List[str], meta: Dict[str, OptionMeta]) -> np.ndarray:
    type_row = np.array([meta[col].option_type for col in option_cols], dtype=object)
    tiled = np.tile(type_row, (len(mask), 1))
    return tiled[mask.to_numpy(dtype=bool)]


def masked_expiry(mask: pd.DataFrame, option_cols: List[str], dtf: pd.DataFrame) -> np.ndarray:
    expiry = dtf["is_expiry_day"].to_numpy(dtype=bool)[:, None]
    tiled = np.tile(expiry, (1, len(option_cols)))
    return tiled[mask.to_numpy(dtype=bool)]


def error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    option_types: np.ndarray,
    expiry_flags: np.ndarray,
) -> Dict[str, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "max_abs_error": np.nan,
            "ce_mse": np.nan,
            "pe_mse": np.nan,
            "normal_day_mse": np.nan,
            "expiry_day_mse": np.nan,
        }
    err = y_pred[ok] - y_true[ok]
    types = option_types[ok]
    expiry = expiry_flags[ok]

    def subset_mse(mask: np.ndarray) -> float:
        if not mask.any():
            return np.nan
        sub_err = err[mask]
        return float(np.mean(sub_err * sub_err))

    mse_value = float(np.mean(err * err))
    return {
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "ce_mse": subset_mse(types == "CE"),
        "pe_mse": subset_mse(types == "PE"),
        "normal_day_mse": subset_mse(~expiry),
        "expiry_day_mse": subset_mse(expiry),
    }


def build_experiments(
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
    mask_fraction: float,
) -> List[ExperimentBundle]:
    bundles: List[ExperimentBundle] = []
    for seed in RANDOM_SEEDS:
        for strategy, hidden_mask in validation_masks(df, option_cols, meta, seed, mask_fraction).items():
            train = apply_hidden_mask(df, hidden_mask, option_cols)
            public_mask, private_mask = split_public_private(hidden_mask, seed, strategy)
            bases = build_base_predictions(train, option_cols, meta, type_cols)
            bundles.append(
                ExperimentBundle(
                    seed=seed,
                    strategy=strategy,
                    train_df=train,
                    hidden_mask=hidden_mask,
                    public_mask=public_mask,
                    private_mask=private_mask,
                    base_predictions=bases,
                )
            )
            print(f"Built validation experiment seed={seed} strategy={strategy} hidden={int(hidden_mask.to_numpy().sum())}")
    return bundles


def weight_grid(step: float) -> List[Dict[str, float]]:
    units = int(round(1.0 / step))
    weights: List[Dict[str, float]] = []
    for a in range(units + 1):
        for b in range(units + 1 - a):
            for c in range(units + 1 - a - b):
                d = units - a - b - c
                weights.append(
                    {
                        "cross_sectional_interpolation": a / units,
                        "past_time_series_model": b / units,
                        "ml_separate_ce_pe_model": c / units,
                        "row_type_mean_baseline": d / units,
                    }
                )
    return weights


def blended_frame(
    bases: Dict[str, pd.DataFrame],
    normal_weights: Dict[str, float],
    expiry_weights: Dict[str, float],
    option_cols: List[str],
    expiry_rows: np.ndarray,
) -> pd.DataFrame:
    normal = sum(normal_weights[name] * bases[name][option_cols] for name in BLEND_COMPONENTS)
    expiry = sum(expiry_weights[name] * bases[name][option_cols] for name in BLEND_COMPONENTS)
    out = normal.copy()
    out.loc[expiry_rows, option_cols] = expiry.loc[expiry_rows, option_cols]
    return out


def evaluate_prediction(
    method: str,
    bundle: ExperimentBundle,
    pred: pd.DataFrame,
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    dtf: pd.DataFrame,
) -> Dict[str, float | int | str]:
    y_public = masked_values(df, bundle.public_mask, option_cols)
    p_public = masked_values(pred, bundle.public_mask, option_cols)
    t_public = masked_types(bundle.public_mask, option_cols, meta)
    e_public = masked_expiry(bundle.public_mask, option_cols, dtf)
    public_metrics = error_metrics(y_public, p_public, t_public, e_public)

    y_private = masked_values(df, bundle.private_mask, option_cols)
    p_private = masked_values(pred, bundle.private_mask, option_cols)
    t_private = masked_types(bundle.private_mask, option_cols, meta)
    e_private = masked_expiry(bundle.private_mask, option_cols, dtf)
    private_metrics = error_metrics(y_private, p_private, t_private, e_private)

    y_all = masked_values(df, bundle.hidden_mask, option_cols)
    p_all = masked_values(pred, bundle.hidden_mask, option_cols)
    t_all = masked_types(bundle.hidden_mask, option_cols, meta)
    e_all = masked_expiry(bundle.hidden_mask, option_cols, dtf)
    all_metrics = error_metrics(y_all, p_all, t_all, e_all)

    public_mse = public_metrics["mse"]
    private_mse = private_metrics["mse"]
    avg_mse = float(np.nanmean([public_mse, private_mse]))
    worst = float(np.nanmax([public_mse, private_mse]))
    return {
        "method": method,
        "validation_strategy": bundle.strategy,
        "seed": bundle.seed,
        "public_30_mse": public_mse,
        "private_70_mse": private_mse,
        "public_private_gap": abs(public_mse - private_mse),
        "avg_mse": avg_mse,
        "worst_case_mse": worst,
        "rmse": all_metrics["rmse"],
        "mae": all_metrics["mae"],
        "max_abs_error": all_metrics["max_abs_error"],
        "ce_mse": all_metrics["ce_mse"],
        "pe_mse": all_metrics["pe_mse"],
        "normal_day_mse": all_metrics["normal_day_mse"],
        "expiry_day_mse": all_metrics["expiry_day_mse"],
        "robust_score": np.nan,
    }


def summarize_for_selection(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby("method", as_index=False)
        .agg(
            public_30_mean_mse=("public_30_mse", "mean"),
            private_70_mean_mse=("private_70_mse", "mean"),
            public_private_gap=("public_private_gap", "mean"),
            avg_validation_mse=("avg_mse", "mean"),
            mse_std=("avg_mse", "std"),
            expiry_day_mse=("expiry_day_mse", "mean"),
            worst_case_mse=("worst_case_mse", "max"),
            ce_mse=("ce_mse", "mean"),
            pe_mse=("pe_mse", "mean"),
        )
        .fillna(0.0)
    )
    grouped["ce_pe_gap"] = (grouped["ce_mse"] - grouped["pe_mse"]).abs()
    grouped["robust_score"] = (
        grouped["private_70_mean_mse"]
        + 0.25 * grouped["public_private_gap"]
        + 0.20 * grouped["mse_std"]
        + 0.20 * grouped["expiry_day_mse"]
        + 0.10 * grouped["worst_case_mse"]
    )
    return grouped.sort_values(["robust_score", "ce_pe_gap"]).reset_index(drop=True)


def select_robust_model(validation_results: pd.DataFrame) -> pd.Series:
    summary = summarize_for_selection(validation_results)
    return summary.iloc[0]


def evaluate_base_methods(
    bundles: List[ExperimentBundle],
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    dtf: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        for method in BASE_METHODS:
            rows.append(evaluate_prediction(method, bundle, bundle.base_predictions[method], df, option_cols, meta, dtf))
    return pd.DataFrame(rows)


def tune_blends(
    bundles: List[ExperimentBundle],
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    dtf: pd.DataFrame,
    step: float,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], str]:
    grids = weight_grid(step)
    expiry_rows = dtf["is_expiry_day"].to_numpy(dtype=bool)

    normal_eval = prepare_blend_eval_records(
        bundles, df, option_cols, meta, dtf, expiry_only=False
    )
    expiry_eval = prepare_blend_eval_records(
        [bundle for bundle in bundles if bundle.strategy == "expiry_day_mask"],
        df,
        option_cols,
        meta,
        dtf,
        expiry_only=True,
    )
    best_normal = tune_weight_grid_from_arrays(grids, normal_eval, "normal")
    best_expiry = tune_weight_grid_from_arrays(grids, expiry_eval, "expiry")

    final_rows = []
    final_method_name = "robust_blend_private_tuned"
    for bundle in bundles:
        pred = blended_frame(bundle.base_predictions, best_normal, best_expiry, option_cols, expiry_rows)
        final_rows.append(evaluate_prediction(final_method_name, bundle, pred, df, option_cols, meta, dtf))

    return pd.DataFrame(final_rows), best_normal, best_expiry, final_method_name


def prepare_blend_eval_records(
    bundles: List[ExperimentBundle],
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    dtf: pd.DataFrame,
    expiry_only: bool,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for bundle in bundles:
        hidden = bundle.hidden_mask.copy()
        expiry_by_row = dtf["is_expiry_day"].to_numpy(dtype=bool)
        if expiry_only:
            hidden.loc[~expiry_by_row, option_cols] = False
        else:
            hidden.loc[expiry_by_row, option_cols] = False
        if int(hidden.to_numpy().sum()) == 0:
            continue
        public, private = split_public_private(
            hidden, bundle.seed, bundle.strategy + ("_expiry" if expiry_only else "_normal")
        )
        records.append(
            {
                "strategy": bundle.strategy,
                "seed": bundle.seed,
                "hidden": hidden,
                "public": public,
                "private": private,
                "y_public": masked_values(df, public, option_cols),
                "y_private": masked_values(df, private, option_cols),
                "y_all": masked_values(df, hidden, option_cols),
                "types_all": masked_types(hidden, option_cols, meta),
                "expiry_all": masked_expiry(hidden, option_cols, dtf),
                "public_preds": {
                    name: masked_values(bundle.base_predictions[name], public, option_cols)
                    for name in BLEND_COMPONENTS
                },
                "private_preds": {
                    name: masked_values(bundle.base_predictions[name], private, option_cols)
                    for name in BLEND_COMPONENTS
                },
                "all_preds": {
                    name: masked_values(bundle.base_predictions[name], hidden, option_cols)
                    for name in BLEND_COMPONENTS
                },
            }
        )
    return records


def weighted_array(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return sum(weights[name] * preds[name] for name in BLEND_COMPONENTS)


def tune_weight_grid_from_arrays(
    grids: List[Dict[str, float]],
    eval_records: List[Dict[str, object]],
    label: str,
) -> Dict[str, float]:
    best_score = float("inf")
    best_weights = grids[0]
    for weights in grids:
        rows = []
        for record in eval_records:
            y_public = record["y_public"]  
            y_private = record["y_private"]  
            y_all = record["y_all"] 
            public_pred = weighted_array(record["public_preds"], weights)  
            private_pred = weighted_array(record["private_preds"], weights)  
            all_pred = weighted_array(record["all_preds"], weights)  
            all_metrics = error_metrics(
                y_all,  
                all_pred,
                record["types_all"], 
                record["expiry_all"],  
            )
            public_mse = mse_arrays(y_public, public_pred)  
            private_mse = mse_arrays(y_private, private_pred)  
            rows.append(
                {
                    "method": f"blend_{label}",
                    "public_30_mse": public_mse,
                    "private_70_mse": private_mse,
                    "public_private_gap": abs(public_mse - private_mse),
                    "avg_mse": float(np.nanmean([public_mse, private_mse])),
                    "worst_case_mse": float(np.nanmax([public_mse, private_mse])),
                    "ce_mse": all_metrics["ce_mse"],
                    "pe_mse": all_metrics["pe_mse"],
                    "expiry_day_mse": all_metrics["expiry_day_mse"],
                }
            )
        if not rows:
            continue
        summary = summarize_for_selection(pd.DataFrame(rows))
        score = float(summary.iloc[0]["robust_score"])
        if score < best_score:
            best_score = score
            best_weights = weights
    return best_weights


def mse_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        return float("nan")
    err = y_pred[ok] - y_true[ok]
    return float(np.mean(err * err))


def add_robust_scores_to_rows(results: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_for_selection(results)
    mapping = summary.set_index("method")["robust_score"].to_dict()
    out = results.copy()
    out["robust_score"] = out["method"].map(mapping)
    return out


def save_best_config(
    path: str | Path,
    best_row: pd.Series,
    best_normal_weights: Dict[str, float],
    best_expiry_weights: Dict[str, float],
    public_best: str,
) -> Dict[str, object]:
    best_method = str(best_row["method"])
    if best_method == "robust_blend_private_tuned":
        best_weights = {
            "normal_day_weights": best_normal_weights,
            "expiry_day_weights": best_expiry_weights,
        }
    else:
        best_weights = {"single_method": best_method}
    config = {
        "best_method": best_method,
        "best_weights": best_weights,
        "best_normal_day_weights": best_normal_weights,
        "best_expiry_day_weights": best_expiry_weights,
        "public_best_method": public_best,
        "private_70_mean_mse": float(best_row["private_70_mean_mse"]),
        "public_30_mean_mse": float(best_row["public_30_mean_mse"]),
        "public_private_gap": float(best_row["public_private_gap"]),
        "expiry_day_mse": float(best_row["expiry_day_mse"]),
        "worst_case_mse": float(best_row["worst_case_mse"]),
        "ce_mse": float(best_row["ce_mse"]),
        "pe_mse": float(best_row["pe_mse"]),
        "mse_std": float(best_row["mse_std"]),
        "robust_score": float(best_row["robust_score"]),
        "random_seeds": RANDOM_SEEDS,
        "selection_formula": (
            "private_70_mean_mse + 0.25*public_private_gap_mean + 0.20*mse_std "
            "+ 0.20*expiry_day_mse + 0.10*worst_case_mse"
        ),
    }
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def choose_final_prediction(
    bases: Dict[str, pd.DataFrame],
    config: Dict[str, object],
    option_cols: List[str],
    expiry_rows: np.ndarray,
) -> pd.DataFrame:
    method = str(config["best_method"])
    if method == "robust_blend_private_tuned":
        weights = config["best_weights"]  
        normal = weights["normal_day_weights"]  
        expiry = weights["expiry_day_weights"]  
        return blended_frame(bases, normal, expiry, option_cols, expiry_rows)  
    return bases[method]


def fill_real_missing(
    df: pd.DataFrame,
    option_cols: List[str],
    meta: Dict[str, OptionMeta],
    type_cols: Dict[str, List[str]],
    config: Dict[str, object],
) -> pd.DataFrame:
    dtf = datetime_features(df, meta)
    bases = build_base_predictions(df, option_cols, meta, type_cols)
    pred = choose_final_prediction(bases, config, option_cols, dtf["is_expiry_day"].to_numpy(dtype=bool))
    filled = df.copy()
    original_missing = filled[option_cols].isna()
    for col in option_cols:
        filled.loc[original_missing[col], col] = pred.loc[original_missing[col], col]
    filled.loc[:, option_cols] = fill_remaining(filled[option_cols], filled, option_cols, type_cols)
    return filled


def save_outputs(
    original: pd.DataFrame,
    filled: pd.DataFrame,
    option_cols: List[str],
    filled_path: str | Path,
    submission_path: str | Path,
) -> pd.DataFrame:
    save_filled = filled.sort_values("_original_index").drop(
        columns=["_original_index", "_parsed_datetime"], errors="ignore"
    )
    save_filled.to_csv(filled_path, index=False)

    rows = []
    original_missing = original[option_cols].isna()
    for col in option_cols:
        for row_idx in original.index[original_missing[col]]:
            rows.append(
                {
                    "id": f"{original.at[row_idx, 'datetime']}{SEPARATOR}{col}",
                    "value": float(filled.at[row_idx, col]),
                }
            )
    submission = pd.DataFrame(rows, columns=["id", "value"])
    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv(submission_path, index=False)
    return submission


def print_leaderboard(results: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    summary = summarize_for_selection(results)
    public_best = str(
        summary.sort_values(["public_30_mean_mse", "robust_score"]).iloc[0]["method"]
    )
    robust_best = str(summary.iloc[0]["method"])
    print("\nLeaderboard sorted by robust_score:")
    display = summary[
        [
            "method",
            "public_30_mean_mse",
            "private_70_mean_mse",
            "public_private_gap",
            "mse_std",
            "expiry_day_mse",
            "worst_case_mse",
            "ce_mse",
            "pe_mse",
            "robust_score",
        ]
    ].copy()
    for col in display.columns:
        if col != "method":
            display[col] = display[col].map(lambda value: f"{value:.8f}")
    print(display.to_string(index=False))
    print(f"\nBest public-style model: {public_best}")
    print(f"Best robust/private-style model: {robust_best}")
    if public_best != robust_best:
        print(
            "Public-best model differs from robust-best model. For final submission, "
            "using robust-best model to reduce private leaderboard risk."
        )
    return summary, public_best, robust_best


def write_readme(path: str | Path) -> None:
    text = """# Model Selection For NIFTY IV Reconstruction

Kaggle's public leaderboard uses only around 30% of hidden test targets, while the final/private ranking uses the remaining 70%. A model can look good on the public 30% and still generalize poorly to the private 70%.

This project therefore simulates public/private validation. For each validation experiment, known non-missing IV cells are hidden, then split into pseudo-public 30% and pseudo-private 70%. Models are selected by a robust score rather than by one public-style MSE.

## Validation Strategies

- `random_cell_mask`
- `missing_pattern_like_mask`
- `public_private_like_mask`
- `strike_block_mask`
- `time_block_mask`
- `expiry_day_mask`
- `grouped_day_mask`

The validation runs across seeds: `7, 11, 21, 42, 77, 101, 2026`.

## Robust Selection

The selected model minimizes:

```text
private_70_mean_mse
+ 0.25 * public_private_gap_mean
+ 0.20 * mse_std
+ 0.20 * expiry_day_mse
+ 0.10 * worst_case_mse
```

CE/PE balance is reported separately and used as a tie-breaker after the robust score.

## Run

```bash
python src/model_selection_iv_solution.py \
--input data/dataset.csv \
--output outputs/submission.csv
```

Outputs:

- `validation_leaderboard.csv`
- `best_model_config.json`
- `filled_dataset.csv`
- `submission.csv`

`submission.csv` contains only originally missing cells and uses:

```text
id,value
datetime||contract,value
```
"""
    Path(path).write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    df, option_cols, meta = load_data(args.input)
    original = df.copy()
    type_cols = type_columns(option_cols, meta)
    dtf = datetime_features(df, meta)
    original_missing = int(original[option_cols].isna().sum().sum())

    print("Building multi-seed public/private validation experiments...")
    bundles = build_experiments(df, option_cols, meta, type_cols, args.mask_fraction)

    print("\nEvaluating base methods...")
    base_results = evaluate_base_methods(bundles, df, option_cols, meta, dtf)

    print("\nTuning robust blend weights...")
    blend_results, normal_weights, expiry_weights, blend_name = tune_blends(
        bundles, df, option_cols, meta, dtf, args.weight_step
    )

    validation_results = pd.concat([base_results, blend_results], ignore_index=True)
    validation_results = add_robust_scores_to_rows(validation_results)
    validation_results.to_csv(args.leaderboard_output, index=False)

    summary, public_best, robust_best = print_leaderboard(validation_results)
    best_row = select_robust_model(validation_results)
    config = save_best_config(args.config_output, best_row, normal_weights, expiry_weights, public_best)

    print("\nFilling real missing cells with robust-best model...")
    filled = fill_real_missing(df, option_cols, meta, type_cols, config)
    submission = save_outputs(original, filled, option_cols, args.filled_output, args.output)
    write_readme("outputs/README_MODEL_SELECTION.md")

    remaining = int(filled[option_cols].isna().sum().sum())
    positive = bool((filled[option_cols].to_numpy(dtype=float) > 0).all())
    best_summary = summary[summary["method"] == robust_best].iloc[0]

    print("\nFinal output:")
    print(f"1. Total original missing cells: {original_missing}")
    print(f"2. Total filled cells: {original_missing}")
    print(f"3. Submission row count: {len(submission)}")
    print(f"4. Best public-style model: {public_best}")
    print(f"5. Best robust/private-style model: {robust_best}")
    print(f"6. Best robust_score: {float(best_summary['robust_score']):.8f}")
    print(f"7. Public_30 validation MSE: {float(best_summary['public_30_mean_mse']):.8f}")
    print(f"8. Private_70 validation MSE: {float(best_summary['private_70_mean_mse']):.8f}")
    print(f"9. Public-private gap: {float(best_summary['public_private_gap']):.8f}")
    print(f"10. Expiry-day MSE: {float(best_summary['expiry_day_mse']):.8f}")
    print(f"11. CE MSE: {float(best_summary['ce_mse']):.8f}")
    print(f"12. PE MSE: {float(best_summary['pe_mse']):.8f}")
    print(f"13. Best blend weights: normal={normal_weights}, expiry={expiry_weights}")
    print(f"14. submission.csv is ready for Kaggle: {remaining == 0 and positive and len(submission) == original_missing}")

    if remaining:
        raise RuntimeError(f"Filled dataset still has {remaining} missing IV values.")
    if len(submission) != original_missing:
        raise RuntimeError("Submission row count does not match original missing cell count.")
    if not positive:
        raise RuntimeError("Some filled IV values are not positive.")


if __name__ == "__main__":
    main()
