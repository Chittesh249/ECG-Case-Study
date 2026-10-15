"""Download the Credit Card Dataset for Clustering (CC GENERAL.csv).

Usage:
    python data/clustering/get_dataset.py
    python data/clustering/get_dataset.py --output data/clustering/CC_GENERAL.csv
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Mirror URLs hosting the standard Arjun Bhasin CC GENERAL.csv dataset
DATASET_URLS = [
    "https://raw.githubusercontent.com/RAHULKASHYAP02/Credit-Card-Segmentation/master/CC%20GENERAL.csv",
    "https://raw.githubusercontent.com/KonuTech/credit-card-dataset-clustering-techniques/master/CC%20GENERAL.csv",
    "https://raw.githubusercontent.com/hamzagorgulu/Credit-Card-Dataset-Analysis/main/CC%20GENERAL.csv",
]

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "clustering" / "CC GENERAL.csv"


def download_dataset(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"Dataset already exists at: {output_path} ({output_path.stat().st_size} bytes)")
        return

    print("Downloading CC GENERAL.csv dataset...")
    success = False
    for url in DATASET_URLS:
        try:
            print(f"Trying download from {url} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                if len(content) > 1000:
                    with open(output_path, "wb") as f:
                        f.write(content)
                    print(f"Successfully downloaded {len(content)} bytes to {output_path}")
                    success = True
                    break
        except Exception as e:
            print(f"Failed from {url}: {e}")

    if not success:
        raise RuntimeError("Could not download CC GENERAL.csv from available mirrors.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the downloaded CC GENERAL.csv.",
    )
    args = parser.parse_args()
    download_dataset(args.output)


if __name__ == "__main__":
    main()
