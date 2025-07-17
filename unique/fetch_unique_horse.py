import os
import pandas as pd

# ✅ Define the data directory: adjust path to point to cheltenham-ai/data
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RACE_FILE = os.path.join(DATA_DIR, "race_results.csv")
HORSE_FILE = os.path.join(DATA_DIR, "unique_horses.csv")  # Output file

# ✅ Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Check if race results data exists
if not os.path.exists(RACE_FILE):
    print(f"❌ Race data not found: {RACE_FILE}")
else:
    # ✅ Load race results
    df_races = pd.read_csv(RACE_FILE)
    
    # ✅ Extract unique horse IDs
    if 'horse_id' in df_races.columns:
        unique_horses = df_races['horse_id'].dropna().unique()
    else:
        print("❌ 'horse_id' column missing from race data.")
        unique_horses = []

    # ✅ Create DataFrame for unique horse IDs and save to CSV
    df_horses = pd.DataFrame(unique_horses, columns=["horse_id"])
    df_horses.to_csv(HORSE_FILE, index=False)
    print(f"✅ Unique horse IDs extracted and saved to {HORSE_FILE}")