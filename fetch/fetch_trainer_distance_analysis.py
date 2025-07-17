import requests
import os
import time
import pandas as pd
import sys

# ✅ Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import USERNAME, PASSWORD, BASE_URL  # Import API credentials

# ✅ Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TRAINER_FILE = os.path.join(DATA_DIR, "unique_trainers.csv")  # Input: Unique trainers
OUTPUT_FILE = os.path.join(DATA_DIR, "trainer_distance_stats.csv")  # Output: Trainer distance stats

# ✅ Setup API headers
auth = (USERNAME, PASSWORD)

# ✅ Load unique trainer IDs
def get_trainer_ids():
    """Load trainer IDs from pre-extracted file to improve speed."""
    if not os.path.exists(TRAINER_FILE):
        print(f"❌ Trainer data not found: {TRAINER_FILE}")
        return []

    df_trainers = pd.read_csv(TRAINER_FILE)

    if "trainer_id" not in df_trainers.columns:
        print("❌ 'trainer_id' column missing from trainer data.")
        return []

    return df_trainers["trainer_id"].dropna().tolist()  # Return unique trainer IDs

# ✅ Fetch Trainer Distance Stats
def fetch_trainer_distance_stats(trainer_id):
    """Fetch distance performance stats for a trainer."""
    url = f"{BASE_URL}/trainers/{trainer_id}/analysis/distances"

    try:
        response = requests.get(url, auth=auth)

        if response.status_code == 429:
            print("⚠️ Rate limit exceeded. Waiting 2 seconds before retrying...")
            time.sleep(2)
            return fetch_trainer_distance_stats(trainer_id)  # Retry

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code} for trainer {trainer_id}: {response.text}")
            return None

        data = response.json()
        total_rides = data.get("total_rides", 0)  # ✅ Extract total_rides

        if "distances" in data and data["distances"]:
            return [{"trainer_id": trainer_id, "total_rides": total_rides, **distance} for distance in data["distances"]]

    except Exception as e:
        print(f"❌ Error fetching trainer distance stats: {e}")

    return None


# ✅ Main function
def main():
    trainer_ids = get_trainer_ids()
    if not trainer_ids:
        print("⚠️ No trainer IDs found.")
        return

    results = []
    for trainer_id in trainer_ids:
        print(f"📡 Fetching distance stats for trainer {trainer_id}...")

        trainer_data = fetch_trainer_distance_stats(trainer_id)

        if trainer_data:
            results.extend(trainer_data)
            print(f"✅ Trainer {trainer_id} - {len(trainer_data)} distance records found!")
        else:
            print(f"⚠️ No distance stats found for trainer {trainer_id}. Skipping.")

        time.sleep(1)  # Avoid hitting API limits

    # ✅ Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(results)} trainer distance stats to {OUTPUT_FILE}")
    else:
        print("⚠️ No trainer distance stats found.")

if __name__ == "__main__":
    main()
