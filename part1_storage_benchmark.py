"""
CSIS 4260 - Assignment 1, Part 1
Storage Format Benchmarking: CSV vs Parquet

Benchmarks read/write performance and file sizes at 1x, 10x, and 100x data scales.
Parquet tested with snappy, gzip, and zstd compression via PyArrow.
"""

import time
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "all_stocks_5yr.csv"
RESULTS_DIR = DATA_DIR / "benchmark_results"
SCALES = [1, 10, 100]
PARQUET_COMPRESSIONS = ["snappy", "gzip", "zstd"]
NUM_READ_ITERATIONS = 3
NUM_WRITE_ITERATIONS = 2


def load_csv_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


def create_scaled_data(df: pd.DataFrame, scale: int) -> pd.DataFrame:
    if scale == 1:
        return df.copy()
    return pd.concat([df] * scale, ignore_index=True)


def benchmark_write_csv(df: pd.DataFrame, path: Path) -> float:
    times = []
    for _ in range(NUM_WRITE_ITERATIONS):
        start = time.perf_counter()
        df.to_csv(path, index=False)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def benchmark_read_csv(path: Path) -> float:
    times = []
    for _ in range(NUM_READ_ITERATIONS):
        start = time.perf_counter()
        pd.read_csv(path, parse_dates=["date"])
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def benchmark_write_parquet(df: pd.DataFrame, path: Path, compression: str) -> float:
    times = []
    table = pa.Table.from_pandas(df, preserve_index=False)
    for _ in range(NUM_WRITE_ITERATIONS):
        start = time.perf_counter()
        pq.write_table(table, path, compression=compression)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def benchmark_read_parquet(path: Path) -> float:
    times = []
    for _ in range(NUM_READ_ITERATIONS):
        start = time.perf_counter()
        pq.read_table(path).to_pandas()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def get_file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def run_benchmark():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading base CSV data...")
    base_df = load_csv_data()
    base_rows = len(base_df)
    print(f"Base dataset: {base_rows:,} rows, {base_df.shape[1]} columns")
    print(f"Columns: {list(base_df.columns)}")
    print(f"Dtypes:\n{base_df.dtypes}\n")

    results = []

    for scale in SCALES:
        print(f"\n{'='*60}")
        print(f"Scale: {scale}x ({base_rows * scale:,} rows)")
        print("=" * 60)

        df = create_scaled_data(base_df, scale)

        # --- CSV ---
        csv_path = RESULTS_DIR / f"data_{scale}x.csv"
        write_time = benchmark_write_csv(df, csv_path)
        read_time = benchmark_read_csv(csv_path)
        csv_size = get_file_size_mb(csv_path)

        results.append({
            "scale": f"{scale}x",
            "format": "csv",
            "compression": "none",
            "rows": len(df),
            "size_mb": round(csv_size, 2),
            "write_sec": round(write_time, 3),
            "read_sec": round(read_time, 3),
        })
        print(f"  CSV          : {csv_size:8.2f} MB | Write: {write_time:.3f}s | Read: {read_time:.3f}s")

        # --- Parquet with different compressions ---
        for compression in PARQUET_COMPRESSIONS:
            parquet_path = RESULTS_DIR / f"data_{scale}x_{compression}.parquet"
            write_time = benchmark_write_parquet(df, parquet_path, compression)
            read_time = benchmark_read_parquet(parquet_path)
            parquet_size = get_file_size_mb(parquet_path)

            results.append({
                "scale": f"{scale}x",
                "format": "parquet",
                "compression": compression,
                "rows": len(df),
                "size_mb": round(parquet_size, 2),
                "write_sec": round(write_time, 3),
                "read_sec": round(read_time, 3),
            })
            print(
                f"  Parquet/{compression:6s}: {parquet_size:8.2f} MB | "
                f"Write: {write_time:.3f}s | Read: {read_time:.3f}s"
            )

        del df

    summary_df = pd.DataFrame(results)
    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nBenchmark summary saved to {summary_path}")

    return summary_df


def print_recommendations(summary_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS TABLE")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS BY SCALE")
    print("=" * 60)

    for scale in ["1x", "10x", "100x"]:
        subset = summary_df[summary_df["scale"] == scale]
        csv_row = subset[subset["format"] == "csv"].iloc[0]
        parquet_rows = subset[subset["format"] == "parquet"]

        best_read = parquet_rows.loc[parquet_rows["read_sec"].idxmin()]
        best_size = parquet_rows.loc[parquet_rows["size_mb"].idxmin()]

        csv_size = csv_row["size_mb"]
        best_pq_size = best_size["size_mb"]
        compression_ratio = csv_size / best_pq_size if best_pq_size > 0 else 0

        print(f"\n  {scale} ({csv_row['rows']:,} rows):")
        print(f"    CSV:                  {csv_size:8.2f} MB | Read: {csv_row['read_sec']:.3f}s")
        print(
            f"    Best read (parquet):  {best_read['size_mb']:8.2f} MB | Read: {best_read['read_sec']:.3f}s "
            f"[{best_read['compression']}]"
        )
        print(
            f"    Smallest (parquet):   {best_pq_size:8.2f} MB "
            f"[{best_size['compression']}] ({compression_ratio:.1f}x compression)"
        )

        read_speedup = csv_row["read_sec"] / best_read["read_sec"] if best_read["read_sec"] > 0 else 0
        if read_speedup > 1:
            print(f"    -> Parquet ({best_read['compression']}) reads {read_speedup:.1f}x faster. Recommend Parquet.")
        else:
            print(f"    -> CSV reads faster at this scale. CSV acceptable for simplicity.")


if __name__ == "__main__":
    summary = run_benchmark()
    print_recommendations(summary)
