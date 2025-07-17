import os
import pandas as pd

# ✅ Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RACES_FILE = os.path.join(DATA_DIR, "race_results.csv")
JOCKEY_FILE = os.path.join(DATA_DIR, "unique_jockeys.csv")

def extract_unique_jockeys():
    """Extract and save unique jockey IDs and names from race results."""
    if not os.path.exists(RACES_FILE):
        print(f"❌ Race data not found: {RACES_FILE}")
        return

    df_races = pd.read_csv(RACES_FILE)
    
    if "jockey_id" not in df_races.columns or "jockey" not in df_races.columns:
        print("❌ 'jockey_id' or 'jockey' column missing from race data.")
        return

    # ✅ Extract unique jockeys
    df_unique_jockeys = df_races[["jockey_id", "jockey"]].dropna().drop_duplicates()

    # ✅ Save to CSV
    df_unique_jockeys.to_csv(JOCKEY_FILE, index=False)
    print(f"✅ Extracted {len(df_unique_jockeys)} unique jockeys and saved to {JOCKEY_FILE}")

if __name__ == "__main__":
    extract_unique_jockeys()
