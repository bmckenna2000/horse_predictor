import os
import pandas as pd

# ✅ Define file paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..", "data"))
RACE_FILE = os.path.join(DATA_DIR, "racecards.csv")
DAM_FILE = os.path.join(DATA_DIR, "unique_dams.csv")  # Output file

# ✅ Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Extract Unique Dams
def extract_unique_dams():
    """Extract unique dam IDs and names and save them to a CSV file."""
    if not os.path.exists(RACE_FILE):
        print(f"❌ Race data not found: {RACE_FILE}")
        return

    df_races = pd.read_csv(RACE_FILE)

    if "dam_id" not in df_races.columns or "dam" not in df_races.columns:
        print("❌ 'dam_id' or 'dam' column missing from race data.")
        return

    # ✅ Get unique dam_id & dam pairs
    unique_dams = df_races[["dam_id", "dam"]].dropna().drop_duplicates()

    # ✅ Save to CSV
    unique_dams.to_csv(DAM_FILE, index=False)
    print(f"✅ Extracted {len(unique_dams)} unique dams and saved to {DAM_FILE}")

if __name__ == "__main__":
    extract_unique_dams()