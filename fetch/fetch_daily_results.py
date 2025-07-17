import requests
import os
import sys
import time
import json
import base64
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import USERNAME, PASSWORD, BASE_URL  # Import API credentials

# Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "todays_position.csv")

# Manually encode Basic Auth credentials
auth_str = f"{USERNAME}:{PASSWORD}"
auth_encoded = base64.b64encode(auth_str.encode()).decode()
HEADERS = {"Authorization": f"Basic {auth_encoded}"}

# API endpoint and pagination settings
URL = f"{BASE_URL}/results"
LIMIT = 50  # Maximum results per request

def fetch_day_results(day, region="gb"):
    """
    Fetch all race results for a single day using pagination.
    The day is both the start_date and end_date.
    Returns a list of race result objects for that day.
    """
    all_results = []
    skip = 0
    while True:
        params = {
            "start_date": day,
            "end_date": day,
            "region": region,
            "limit": LIMIT,
            "skip": skip
        }
        try:
            response = requests.get(URL, params=params, headers=HEADERS)
            if response.status_code == 429:
                print(f"⚠️ Rate limit exceeded for day {day} at skip {skip}. Waiting 2 seconds before retrying...")
                time.sleep(2)
                continue
            if response.status_code != 200:
                print(f"❌ API Error {response.status_code} for day {day} at skip {skip}: {response.text}")
                break
            data = response.json()
            results = data.get("results", [])
            if not results:
                break  # No more results for this day
            all_results.extend(results)
            skip += LIMIT
            time.sleep(1)  # Brief pause to avoid hitting rate limits
        except Exception as e:
            print(f"❌ Error fetching day {day} at skip {skip}: {e}")
            break
    return all_results

def flatten_results(results):
    """
    Flatten a list of race result objects into runner-level rows.
    Each row will include:
      - race_id, horse_id, and position.
    """
    rows = []
    for race in results:
        # Only process GB races
        if race.get("region", "").lower() != "gb":
            continue
        race_id = race.get("race_id")
        for runner in race.get("runners", []):
            row = {
                "race_id": race_id,
                "horse_id": runner.get("horse_id"),
                "position": runner.get("position")
            }
            rows.append(row)
    return rows

def main():
    # Get today's date
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"📡 Fetching results for today: {today}")

    # Fetch results for today
    all_results = fetch_day_results(today)
    print(f"✅ Fetched {len(all_results)} race results for today.")

    # Flatten the results to get only race_id, horse_id, and position
    flattened = flatten_results(all_results)
    print(f"✅ Total flattened rows: {len(flattened)}")

    # Create DataFrame, standardize keys, and fill missing positions with -1
    df = pd.DataFrame(flattened)
    for col in ["race_id", "horse_id"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["position"] = df["position"].fillna("-1")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved today's results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()