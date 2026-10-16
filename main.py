"""EddyP1: Satellite ocean data ingestion and exploration.

Fetches predigested satellite CO2 and wind data (originating from F4CR/f4cr-cdrr),
extracts wind vector and geospatial coordinates (u_wind, v_wind, latitude, longitude),
and saves the subsetted dataset for exploratory analysis (e.g. Google Sheets / Polars / DuckDB).
"""

from __future__ import annotations

import gzip
import io
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import polars as pl

# Base endpoint for predigested satellite datasets
BASE_DATA_URL = "https://oco.carbonviews.org/data"

# Primary columns of interest for wind vector and eddy exploration
SELECTED_COLUMNS: tuple[str, ...] = ("u_wind", "v_wind", "latitude", "longitude")


def compose_dataset_url(
    region: str,
    year: int,
    start_month: int,
    end_month: int,
) -> str:
    """Compose the URL for an averaged satellite dataset given region and date range."""
    date_str = f"{region} {year} {start_month:02d} to {end_month:02d}"
    encoded_folder = urllib.parse.quote(date_str)
    filename = f"data_averaged_{date_str}.csv"
    encoded_filename = urllib.parse.quote(filename)
    return f"{BASE_DATA_URL}/{encoded_folder}/{encoded_filename}"


def fetch_csv_bytes(url: str) -> io.BytesIO:
    """Fetch CSV content from URL. If .csv is absent (404), fallback to .csv.gz and un-gzip it."""
    headers = {"User-Agent": "EddyP1-Explorer/1.0"}

    def _download(target_url: str) -> bytes:
        req = urllib.request.Request(target_url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            return resp.read()

    target_url = url
    try:
        print(f"Fetching: {target_url}")
        content = _download(target_url)
    except urllib.error.HTTPError as err:
        if err.code == 404 and target_url.endswith(".csv"):
            gz_url = f"{target_url}.gz"
            print(f".csv returned 404. Falling back to: {gz_url}")
            content = _download(gz_url)
        else:
            raise

    # If the payload is gzipped (checks magic bytes 0x1f 0x8b), decompress it
    if content.startswith(b"\x1f\x8b"):
        print("Decompressing gzipped payload...")
        content = gzip.decompress(content)

    return io.BytesIO(content)


def process_dataset(
    url: str,
    output_csv_path: Path,
    columns: Sequence[str] = SELECTED_COLUMNS,
) -> pl.DataFrame:
    """Download dataset, extract selected columns, and save to a local CSV file."""
    csv_stream = fetch_csv_bytes(url)

    # Load with Polars, selecting only the requested columns
    df = pl.read_csv(csv_stream, columns=columns)

    # Ensure output directory exists and write CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_csv_path)

    # Compute metrics
    total_bytes = output_csv_path.stat().st_size
    num_rows = len(df)
    num_cols = len(df.columns)

    print("\n" + "=" * 50)
    print(f"Dataset successfully saved to: {output_csv_path}")
    print(f"Total Rows:     {num_rows:,}")
    print(f"Total Columns:  {num_cols} ({', '.join(df.columns)})")
    print(f"File Size:      {total_bytes / (1024 * 1024):.2f} MB ({total_bytes:,} bytes)")
    print("=" * 50)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())

    return df


def main():
    # Target dataset: Baja 2020 01 to 04
    dataset_url = compose_dataset_url(
        region="Baja",
        year=2020,
        start_month=1,
        end_month=4,
    )
    output_file = Path("data/baja_2020_01_to_04_wind_geo.csv")

    process_dataset(url=dataset_url, output_csv_path=output_file)


if __name__ == "__main__":
    main()
