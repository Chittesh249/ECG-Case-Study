"""Download the 2010 NY SPARCS hospital inpatient discharge dataset.

Usage:
	python data/regression/get_dataset.py
	python data/regression/get_dataset.py --output regression_dataset/custom.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path


CATALOG_URL = "https://www.kaggle.com/datasets/thedevastator/2010-new-york-state-hospital-inpatient-discharge"
DATASET_TITLE = "Hospital Inpatient Discharges (SPARCS De-Identified): 2010"
DEFAULT_OUTPUT = (
	Path(__file__).resolve().parents[2]
	/ "regression_dataset"
	/ "hospital-inpatient-discharges-sparcs-de-identified-2010-1.csv"
)


def find_dataset_id() -> str:
	"""Find the Socrata identifier instead of hard-coding a catalog ID."""
	query = urllib.parse.urlencode({"q": DATASET_TITLE, "limit": 100})
	with urllib.request.urlopen(f"{CATALOG_URL}?{query}", timeout=60) as response:
		catalog = json.load(response)

	expected = DATASET_TITLE.casefold()
	for result in catalog.get("results", []):
		resource = result.get("resource", {})
		name = resource.get("name", "").casefold()
		if name == expected or expected in name:
			identifier = resource.get("id")
			if identifier:
				return identifier

	raise RuntimeError(
		f"Could not find '{DATASET_TITLE}' in the NY Open Data catalog."
	)


def download_dataset(output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataset_id = find_dataset_id()
	download_url = f"https://data.ny.gov/resource/{dataset_id}.csv?$limit=3000000"

	print(f"Downloading dataset {dataset_id}...")
	with urllib.request.urlopen(download_url, timeout=600) as response:
		with output_path.open("wb") as output_file:
			while chunk := response.read(1024 * 1024):
				output_file.write(chunk)

	print(f"Saved dataset to: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help="Where to save the downloaded CSV.",
	)
	args = parser.parse_args()
	download_dataset(args.output)


if __name__ == "__main__":
	main()
