import os
import pandas as pd

# ✅ Define file paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RACE_FILE = os.path.join(DATA_DIR, "racecards.csv")
SIRE_FILE = os.path.join(DATA_DIR, "unique_sires.csv")  # Output file

# ✅ Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Extract Unique Sires
def extract_unique_sires():
    """Extract unique sire IDs and names and save them to a CSV file."""
    if not os.path.exists(RACE_FILE):
        print(f"❌ Race data not found: {RACE_FILE}")
        return

    df_races = pd.read_csv(RACE_FILE)

    if "sire_id" not in df_races.columns or "sire" not in df_races.columns:
        print("❌ 'sire_id' or 'sire' column missing from race data.")
        return

    # ✅ Get unique sire_id & sire pairs
    unique_sires = df_races[["sire_id", "sire"]].dropna().drop_duplicates()

    # ✅ Save to CSV
    unique_sires.to_csv(SIRE_FILE, index=False)
    print(f"✅ Extracted {len(unique_sires)} unique sires and saved to {SIRE_FILE}")

if __name__ == "__main__":
    extract_unique_sires()