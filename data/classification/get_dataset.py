"""Download the MIT-BIH Arrhythmia Database using WFDB.

Usage:
	python data/classification/get_dataset.py
	python data/classification/get_dataset.py --output data/mitdb

The downloaded files are raw WFDB records. The classification notebook can
then be used to create model-ready NumPy arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "mitdb"


def download_database(output_path: Path) -> None:
	try:
		import wfdb
	except ImportError as error:
		raise SystemExit(
			"WFDB is required. Install it with: python -m pip install wfdb"
		) from error

	output_path.mkdir(parents=True, exist_ok=True)
	print("Downloading the MIT-BIH Arrhythmia Database...")
	wfdb.dl_database("mitdb", dl_dir=str(output_path))
	print(f"Saved raw WFDB records to: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help="Directory where raw MIT-BIH records will be saved.",
	)
	args = parser.parse_args()
	download_database(args.output)


if __name__ == "__main__":
	main()


# New York hospital inpatient Dataset
