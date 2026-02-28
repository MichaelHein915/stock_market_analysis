"""
CSIS 4260 - Assignment 1, Part 2
Data Manipulation, Analysis & Prediction Models

- Compares Pandas vs Polars for dataframe operations (load, indicator calc, filter, groupby)
- Adds technical indicators: SMA (20-day), RSI (14-period)
- Two prediction algorithms: Linear Regression, Random Forest
- 80-20 train/test split for next-day closing price prediction
"""

import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "all_stocks_5yr.csv"
OUTPUT_DIR = DATA_DIR / "model_outputs"
SMA_PERIOD = 20
RSI_PERIOD = 14
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Technical Indicators – Pandas
# ---------------------------------------------------------------------------

def add_sma_pandas(df: pd.DataFrame, period: int = SMA_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df["sma"] = df.groupby("name")["close"].transform(
        lambda x: x.rolling(window=period, min_periods=1).mean()
    )
    return df


def add_rsi_pandas(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()

    def rsi_series(series):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df["rsi"] = df.groupby("name")["close"].transform(rsi_series)
    return df


def add_indicators_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return add_rsi_pandas(add_sma_pandas(df))


# ---------------------------------------------------------------------------
# Technical Indicators – Polars
# ---------------------------------------------------------------------------

def add_indicators_polars(df_pl: pl.DataFrame) -> pl.DataFrame:
    return (
        df_pl
        .with_columns(
            pl.col("close")
            .rolling_mean(window_size=SMA_PERIOD, min_samples=1)
            .over("name")
            .alias("sma")
        )
        .with_columns(
            pl.col("close").diff().over("name").alias("_delta")
        )
        .with_columns(
            pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0).alias("_gain"),
            pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0).alias("_loss"),
        )
        .with_columns(
            pl.col("_gain")
            .rolling_mean(window_size=RSI_PERIOD, min_samples=1)
            .over("name")
            .alias("_avg_gain"),
            pl.col("_loss")
            .rolling_mean(window_size=RSI_PERIOD, min_samples=1)
            .over("name")
            .alias("_avg_loss"),
        )
        .with_columns(
            (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10)))
            .alias("rsi")
        )
        .drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])
    )


# ---------------------------------------------------------------------------
# Benchmark: Pandas vs Polars on multiple operations
# ---------------------------------------------------------------------------

def benchmark_pandas_vs_polars():
    print("=" * 60)
    print("PANDAS vs POLARS BENCHMARK")
    print("=" * 60)

    timings = {}

    # --- Load ---
    start = time.perf_counter()
    df_pd = pd.read_csv(CSV_PATH, parse_dates=["date"])
    timings["pandas_load"] = time.perf_counter() - start

    start = time.perf_counter()
    df_pl = pl.read_csv(CSV_PATH).with_columns(pl.col("date").str.to_date())
    timings["polars_load"] = time.perf_counter() - start

    # --- Add indicators ---
    start = time.perf_counter()
    df_pd = add_indicators_pandas(df_pd)
    timings["pandas_indicators"] = time.perf_counter() - start

    df_pl = pl.read_csv(CSV_PATH).with_columns(pl.col("date").str.to_date())
    start = time.perf_counter()
    df_pl = add_indicators_polars(df_pl)
    timings["polars_indicators"] = time.perf_counter() - start

    # --- Filter ---
    start = time.perf_counter()
    _ = df_pd[df_pd["close"] > df_pd["sma"]]
    timings["pandas_filter"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = df_pl.filter(pl.col("close") > pl.col("sma"))
    timings["polars_filter"] = time.perf_counter() - start

    # --- GroupBy aggregation ---
    start = time.perf_counter()
    _ = df_pd.groupby("name").agg({"close": "mean", "volume": "sum"})
    timings["pandas_groupby"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = df_pl.group_by("name").agg(pl.col("close").mean(), pl.col("volume").sum())
    timings["polars_groupby"] = time.perf_counter() - start

    # --- Sort ---
    start = time.perf_counter()
    _ = df_pd.sort_values(["name", "date"])
    timings["pandas_sort"] = time.perf_counter() - start

    start = time.perf_counter()
    _ = df_pl.sort(["name", "date"])
    timings["polars_sort"] = time.perf_counter() - start

    # Print results
    operations = ["load", "indicators", "filter", "groupby", "sort"]
    print(f"\n  {'Operation':<18} {'Pandas (s)':>12} {'Polars (s)':>12} {'Speedup':>10}")
    print("  " + "-" * 54)
    for op in operations:
        pd_t = timings[f"pandas_{op}"]
        pl_t = timings[f"polars_{op}"]
        ratio = pd_t / pl_t if pl_t > 0 else float("inf")
        winner = "Polars" if ratio > 1 else "Pandas"
        print(f"  {op:<18} {pd_t:>12.4f} {pl_t:>12.4f} {ratio:>8.1f}x ({winner})")

    return df_pd


# ---------------------------------------------------------------------------
# Prediction Models
# ---------------------------------------------------------------------------

FEATURE_COLS = ["open", "high", "low", "close", "volume", "sma", "rsi"]


def prepare_features_target(df: pd.DataFrame):
    df = df.dropna(subset=FEATURE_COLS).copy()
    df["target"] = df.groupby("name")["close"].shift(-1)
    df = df.dropna(subset=["target"])
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y, df


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    return {
        "model": model,
        "name": model_name,
        "train_time": train_time,
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }


def run_prediction_pipeline(df: pd.DataFrame):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)

    df = df.dropna(subset=FEATURE_COLS)
    df.to_csv(OUTPUT_DIR / "enriched_stocks.csv", index=False)
    print(f"Enriched data saved ({len(df):,} rows)")

    X, y, df_clean = prepare_features_target(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    print(f"\n  {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Train (s)':>12}")
    print("  " + "-" * 66)

    trained = []
    for name, model in models.items():
        r = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        trained.append(r)
        print(f"  {name:<22} {r['mae']:>10.4f} {r['rmse']:>10.4f} {r['r2']:>10.4f} {r['train_time']:>12.3f}")

    # Save scaler and models
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    for r in trained:
        fname = f"model_{r['name'].lower().replace(' ', '_')}.pkl"
        with open(OUTPUT_DIR / fname, "wb") as f:
            pickle.dump(r["model"], f)

    # Generate per-company predictions for the dashboard.
    # We use the already-fitted scaler + models (trained on 80 % split) to
    # predict on the full dataset so every company has predictions to display.
    X_full_scaled = scaler.transform(X)
    df_predictions = df_clean[["date", "name", "close", "target"]].copy()
    for r in trained:
        df_predictions[f"{r['name']}_pred"] = r["model"].predict(X_full_scaled)

    df_predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"\nPredictions saved ({len(df_predictions):,} rows)")
    return trained


if __name__ == "__main__":
    enriched_df = benchmark_pandas_vs_polars()
    run_prediction_pipeline(enriched_df)
