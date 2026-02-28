# CSIS 4260 – Assignment 1: Stock Price Analysis & Prediction

Time-series analysis of S&P 500 daily stock prices (2013-02-08 to 2018-02-07, 619,040 rows across 505 companies).

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
# Part 1 – Storage benchmarks (takes ~5-10 minutes due to 100x scale)
python part1_storage_benchmark.py

# Part 2 – Data analysis & prediction models (~3 minutes)
python part2_data_analysis.py

# Part 3 – Dashboard
streamlit run part3_dashboard.py
# Opens at http://localhost:8501
```

## Project Structure

```
stock_analysis_assgn1/
├── all_stocks_5yr.csv            # Original dataset (~29 MB, 619,040 rows)
├── part1_storage_benchmark.py    # CSV vs Parquet benchmarking
├── part2_data_analysis.py        # Pandas vs Polars, indicators, ML models
├── part3_dashboard.py            # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── benchmark_results/            # Part 1 outputs (generated on run)
│   ├── benchmark_summary.csv     # Full benchmark results table
│   ├── data_1x.csv / .parquet   # 1x scale files
│   └── ...
└── model_outputs/                # Part 2 outputs (generated on run)
    ├── enriched_stocks.csv       # Dataset with SMA & RSI indicators
    ├── predictions.csv           # Per-company predictions (all models)
    ├── scaler.pkl                # Fitted StandardScaler
    ├── model_linear_regression.pkl
    └── model_random_forest.pkl
```

---

## Part 1: Storing and Retrieving Data

### Research: CSV vs Parquet

**CSV** is a row-oriented text format. It is human-readable, universally supported, and requires no special libraries. However, it has no built-in compression, no schema enforcement (types are inferred on read), and reading specific columns requires scanning the entire file.

**Parquet** is a columnar binary format designed for analytics. It embeds the schema, supports built-in compression, and enables column pruning (reading only the columns you need). The trade-off is that it requires a library like PyArrow and is not human-readable.

**Compression codecs tested:**

| Codec  | Characteristics |
|--------|----------------|
| Snappy | Fast compression/decompression, moderate ratio. Good default. |
| Gzip   | Higher compression ratio, but slower writes. Best when storage cost matters. |
| Zstd   | High compression ratio with fast decompression. Tunable. Best overall balance. |

Reference: [Apache Arrow Parquet documentation](https://arrow.apache.org/docs/python/parquet.html)

### Benchmark Results

Benchmarked on macOS (Apple Silicon). Each write averaged over 2 iterations, each read over 3 iterations.

| Scale | Format | Compression | Size (MB) | Write (s) | Read (s) |
|-------|--------|-------------|-----------|-----------|----------|
| 1x    | CSV    | none        | 28.21     | 1.312     | 0.215    |
| 1x    | Parquet| snappy      | 10.15     | 0.088     | 0.035    |
| 1x    | Parquet| gzip        | 8.06      | 0.622     | 0.027    |
| 1x    | Parquet| zstd        | 8.09      | 0.096     | 0.019    |
| 10x   | CSV    | none        | 282.10    | 13.362    | 2.255    |
| 10x   | Parquet| snappy      | 95.37     | 0.798     | 0.199    |
| 10x   | Parquet| gzip        | 76.00     | 5.799     | 0.186    |
| 10x   | Parquet| zstd        | 75.46     | 0.844     | 0.149    |
| 100x  | CSV    | none        | 2,821.02  | 137.465   | 46.349   |
| 100x  | Parquet| snappy      | 951.91    | 8.639     | 9.329    |
| 100x  | Parquet| gzip        | 758.32    | 87.530    | 13.038   |
| 100x  | Parquet| zstd        | 751.95    | 25.781    | 32.168   |

Full results are saved to `benchmark_results/benchmark_summary.csv`.

### Analysis & Recommendations

**At 1x (619K rows):** Parquet/zstd reads 11x faster (0.019s vs 0.215s) and is 3.5x smaller (8 MB vs 28 MB). Even at the smallest scale, Parquet is measurably better. However, CSV remains acceptable because absolute times are small and CSV is simpler to inspect.

**At 10x (6.2M rows):** Parquet/zstd reads 15x faster (0.149s vs 2.255s) and is 3.7x smaller (75 MB vs 282 MB). The write speedup is even more dramatic — CSV takes 13.4s vs 0.84s for Parquet/zstd (16x). At this scale, Parquet is strongly recommended.

**At 100x (62M rows):** CSV becomes 2.8 GB and takes 137s to write and 46s to read. Parquet/snappy reads in 9.3s (5x faster) and is 3x smaller at 952 MB. Parquet/snappy offers the best read speed at this scale, while gzip provides the best compression (758 MB). At this scale, CSV is impractical and Parquet is the clear choice.

**Overall recommendation:** Use **Parquet with snappy compression** as the default — it offers the best read performance at larger scales and the fastest writes across all scales. For storage-constrained environments, **gzip** or **zstd** provide better compression at the cost of slower writes. At 1x, **zstd** provides the fastest reads. CSV should only be used when human readability is required.

---

## Part 2: Data Manipulation & Prediction Models

### Library Comparison: Pandas vs Polars

Both libraries were benchmarked on the original 1x dataset (619,040 rows) across five operations:

| Operation        | Pandas (s) | Polars (s) | Speedup       |
|-----------------|-----------|-----------|---------------|
| CSV Load         | 0.275     | 0.163     | 1.7x (Polars) |
| Add Indicators   | 0.454     | 0.087     | 5.2x (Polars) |
| Filter Rows      | 0.014     | 0.008     | 1.7x (Polars) |
| GroupBy Aggregate | 0.021     | 0.007     | 3.2x (Polars) |
| Sort             | 0.031     | 0.027     | 1.2x (Polars) |

**Analysis:** Polars is faster across every operation tested, with the biggest advantage in grouped window calculations (5.2x for indicator computation). This is because Polars uses a multi-threaded Rust engine and lazy evaluation, whereas Pandas operates single-threaded on Python objects. For this dataset size, both libraries are fast enough in absolute terms, but at larger scales Polars' advantage would compound. The prediction pipeline uses Pandas because scikit-learn expects NumPy arrays (which Pandas integrates with directly), but Polars is the better choice for the data transformation stage.

### Technical Indicators

Two indicators were calculated per company (grouped by ticker, sorted by date):

1. **SMA(20) — Simple Moving Average (20-day):** The mean closing price over the past 20 trading days. Smooths short-term volatility and indicates trend direction. When the price is above SMA, it suggests an uptrend.

2. **RSI(14) — Relative Strength Index (14-period):** Measures the speed and magnitude of recent price changes on a 0–100 scale. Values above 70 indicate overbought conditions; below 30 indicates oversold. Calculated as: RSI = 100 − 100/(1 + avg_gain/avg_loss).

Both indicators are added to the dataframe and saved to `model_outputs/enriched_stocks.csv`.

### Prediction Models

**Task:** Predict the next day's closing price for each company.

**Features:** open, high, low, close, volume, sma, rsi (7 features, standardized with StandardScaler).

**Target:** Next day's closing price (created by shifting close price by −1 within each company group).

**Split:** 80% train / 20% test (random split, `random_state=42`).

| Model              | MAE    | RMSE   | R²     | Train Time |
|-------------------|--------|--------|--------|------------|
| Linear Regression  | 0.8455 | 2.0479 | 0.9996 | 0.06s      |
| Random Forest      | 0.8970 | 2.1032 | 0.9995 | 162.5s     |

**Analysis:** Both models achieve very high R² (>0.999) because stock prices are highly autocorrelated — today's price is a strong predictor of tomorrow's price. Linear Regression slightly outperforms Random Forest on all error metrics while training ~2700x faster. This is because the relationship between current features and next-day close is predominantly linear. Random Forest's added complexity (100 trees, ~3 GB model) does not improve accuracy for this task. For a production system, Linear Regression is the practical choice due to its speed, interpretability, and comparable accuracy.

---

## Part 3: Dashboard

### Library Research

| Library   | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Streamlit** | Minimal code, rapid prototyping, built-in widgets, auto-rerun on change | Limited layout control, full page reload on interaction | Fast dashboards, demos, data apps |
| **Dash**      | Fine-grained layout control, production-ready, React components, callbacks | Steeper learning curve, more boilerplate | Complex production dashboards |
| **Reflex**    | Full-stack Python (no JS), reactive model, modern UI | Newer ecosystem, smaller community, evolving API | Web apps with Python-only stack |

### Choice: Streamlit

Streamlit was selected because:

1. **Simplicity:** A searchable company selector and interactive Plotly charts require only ~160 lines of code. Dash would need explicit callback wiring for the same functionality.
2. **Built-in data caching:** `@st.cache_data` makes loading the predictions CSV fast on repeated interactions.
3. **Plotly integration:** Native `st.plotly_chart()` support with full interactivity (zoom, hover, pan).
4. **Ecosystem maturity:** Widely used in data science, extensive documentation, and easy deployment.

### Dashboard Features

- **Sidebar:** Searchable dropdown to select any of the 505 company tickers
- **KPI metrics row:** Data points count, per-company MAE for both models, date range
- **Chart 1 — Predictions:** Actual next-day close vs Linear Regression and Random Forest predictions
- **Chart 2 — Technical Indicators:** Close price with SMA(20) overlay, and RSI(14) on a secondary axis with overbought/oversold reference lines (70/30)

Run with: `streamlit run part3_dashboard.py`

---

## Scale Assumptions

| Scale | Rows       | Approximate Size (CSV) | Use Case          |
|-------|-----------|----------------------|-------------------|
| 1x    | 619,040   | ~29 MB               | Current dataset   |
| 10x   | 6,190,400 | ~282 MB              | Medium expansion  |
| 100x  | 61,904,000| ~2.8 GB              | Large-scale       |
