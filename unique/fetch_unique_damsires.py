import os
import pandas as pd

# ✅ Define file paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RACE_FILE = os.path.join(DATA_DIR, "racecards.csv")
DAMSIRE_FILE = os.path.join(DATA_DIR, "unique_damsires.csv")  # Output file

# ✅ Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Extract Unique Damsires
def extract_unique_damsires():
    """Extract unique damsire IDs and names and save them to a CSV file."""
    if not os.path.exists(RACE_FILE):
        print(f"❌ Race data not found: {RACE_FILE}")
        return

    df_races = pd.read_csv(RACE_FILE)

    if "damsire_id" not in df_races.columns or "damsire" not in df_races.columns:
        print("❌ 'damsire_id' or 'damsire' column missing from race data.")
        return

    # ✅ Get unique damsire_id & damsire pairs
    unique_damsires = df_races[["damsire_id", "damsire"]].dropna().drop_duplicates()

    # ✅ Save to CSV
    unique_damsires.to_csv(DAMSIRE_FILE, index=False)
    print(f"✅ Extracted {len(unique_damsires)} unique damsires and saved to {DAMSIRE_FILE}")

if __name__ == "__main__":
    extract_unique_damsires()