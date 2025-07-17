import os
import pandas as pd

# ✅ Define file paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RACE_FILE = os.path.join(DATA_DIR, "race_results.csv")
OWNER_FILE = os.path.join(DATA_DIR, "unique_owners.csv")  # Output file

# ✅ Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Extract Unique Owners
def extract_unique_owners():
    """Extract unique owner IDs and save them to a CSV file."""
    if not os.path.exists(RACE_FILE):
        print(f"❌ Race data not found: {RACE_FILE}")
        return

    df_races = pd.read_csv(RACE_FILE)

    if "owner_id" not in df_races.columns or "owner" not in df_races.columns:
        print("❌ 'owner_id' or 'owner' column missing from race data.")
        return

    # ✅ Get unique owner_id & owner pairs
    unique_owners = df_races[["owner_id", "owner"]].dropna().drop_duplicates()

    # ✅ Save to CSV
    unique_owners.to_csv(OWNER_FILE, index=False)
    print(f"✅ Extracted {len(unique_owners)} unique owners and saved to {OWNER_FILE}")

if __name__ == "__main__":
    extract_unique_owners()
