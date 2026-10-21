r"""
download_data.py  --  Milestone M1, step 1
Downloads the 4 raw UCI Heart Disease files into data/raw/ and combines them
into data/heart_combined.csv (one row per patient, plus a "hospital" column).

Why these 4 files?
  Cleveland (303) + Hungarian (294) + Switzerland (123) + VA Long Beach (200)
  = 920 patients total.  That is enough for the later federated step of
  3 hospitals x 300 patients = 900.

Run:  venv\Scripts\python.exe ml\download_data.py
"""

import csv
import io
import os
import urllib.request

# Folder locations, built from this file's location (works from any folder)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
COMBINED_CSV = os.path.join(ROOT, "data", "heart_combined.csv")

# The 4 official UCI "processed" files (same 14 columns in all of them)
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SOURCES = [
    ("cleveland", "processed.cleveland.data"),
    ("hungarian", "processed.hungarian.data"),
    ("switzerland", "processed.switzerland.data"),
    ("va_longbeach", "processed.va.data"),
]

# Plain-English names for the 14 columns
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Each source is tagged with the hospital it will belong to later.
# Round-robin A/B/C/A/B/C... keeps the three hospitals roughly the same size
# and roughly the same mix of sick/healthy patients.
HOSPITALS = ["A", "B", "C"]


def download_sources():
    """Download the 4 raw files (skips files we already have)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, filename in SOURCES:
        dest = os.path.join(RAW_DIR, name + ".csv")
        if os.path.exists(dest):
            print("already have", dest)
            continue
        url = BASE_URL + filename
        print("downloading", url)
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode("utf-8")
        # The UCI files have no header row, only comma-separated values.
        with open(dest, "w", newline="", encoding="utf-8") as f:
            f.write("\n".join([",".join(COLUMNS)] + text.strip().splitlines()) + "\n")
        print("saved", dest)


def combine():
    """Merge the 4 files into one CSV and add the 'hospital' column."""
    rows = []
    for name, _ in SOURCES:
        path = os.path.join(RAW_DIR, name + ".csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # "?" means "value is missing" in the UCI files -> store empty
                clean = {k: ("" if v.strip() == "?" else v.strip()) for k, v in row.items()}
                clean["source"] = name                      # where the patient came from
                clean["hospital"] = HOSPITALS[i % 3]        # A / B / C
                rows.append(clean)

    fieldnames = COLUMNS + ["source", "hospital"]
    os.makedirs(os.path.dirname(COMBINED_CSV), exist_ok=True)
    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Small summary so you can see what happened
    counts = {}
    for r in rows:
        counts[r["hospital"]] = counts.get(r["hospital"], 0) + 1
    print("\nCombined file:", COMBINED_CSV)
    print("Total patients:", len(rows))
    print("Patients per hospital:", counts)


if __name__ == "__main__":
    download_sources()
    combine()
