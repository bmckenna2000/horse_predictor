import pandas as pd
import os

# ✅ Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RACE_FILE = os.path.join(DATA_DIR, "race_results.csv")  # 12-month race data
TRAINER_FILE = os.path.join(DATA_DIR, "unique_trainers.csv")  # Output file for unique trainers

# ✅ Extract unique trainers
def extract_unique_trainers():
    """Extract unique trainer_id and trainer from race results."""
    if not os.path.exists(RACE_FILE):
        print(f"❌ Race data file not found: {RACE_FILE}")
        return

    df = pd.read_csv(RACE_FILE)

    # Check if required columns exist
    if "trainer_id" not in df.columns or "trainer" not in df.columns:
        print("❌ 'trainer_id' or 'trainer' column missing from race data.")
        return

    # Drop duplicates and save unique trainer data
    df_unique = df[["trainer_id", "trainer"]].dropna().drop_duplicates()
    df_unique.to_csv(TRAINER_FILE, index=False)

    print(f"✅ Extracted {len(df_unique)} unique trainers and saved to {TRAINER_FILE}")

# ✅ Run the function
if __name__ == "__main__":
    extract_unique_trainers()
