import requests
import os
import sys
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

# ✅ Import credentials from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import USERNAME, PASSWORD, BASE_URL

# ✅ Define output file path
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "racecards.csv")

# ✅ Define API endpoint
API_URL = f"{BASE_URL}/racecards/pro"

# ✅ Define headers & authentication
auth = HTTPBasicAuth(USERNAME, PASSWORD)

def safe_extract(value, key=None):
    """ Extracts specific key from a dictionary or safely converts lists to strings """
    if isinstance(value, dict):
        return value.get(key, "") if key else json.dumps(value)  # Extract specific key if provided
    elif isinstance(value, list):
        return ", ".join(str(item) if isinstance(item, str) else json.dumps(item) for item in value)
    return value  # Return as is if already a string or number

def extract_odds_decimal(odds):
    """ Safely extract decimal odds whether odds is a list or dictionary """
    if isinstance(odds, list) and len(odds) > 0:  # If it's a list, use the first item
        return odds[0].get("decimal", "")
    elif isinstance(odds, dict):  # If it's a dictionary, get decimal directly
        return odds.get("decimal", "")
    return ""  # Return empty if odds is missing

def get_past_dates(months=12):
    """ Generate a list of past dates for the last 12 months """
    today = datetime.today()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, months * 30 + 1)]

def fetch_racecards(date):
    """ Fetch racecards for a specific date """
    params = {"date": date}
    try:
        response = requests.get(API_URL, auth=auth, params=params)

        print(f"📡 Fetching {date} - Status: {response.status_code}")  # Debugging log

        if response.status_code == 429:
            print(f"⚠️ Rate limit exceeded on {date}. Retrying after 2 seconds...")
            time.sleep(2)
            return fetch_racecards(date)  # Retry

        if response.status_code != 200:
            print(f"❌ Error {response.status_code} for {date}: {response.text}")
            return []

        data = response.json()

        # 🔹 Print full response to debug the issue
        print(f"📜 Full API Response for {date}: {json.dumps(data, indent=2)[:500]}...")  # Print first 500 chars

        racecards = data.get("racecards", [])

# 🔹 Check what regions are available
        available_regions = set(race["region"] for race in racecards)
        print(f"🌍 Available regions for {date}: {available_regions}")

# ✅ Filter for GB races
        racecards_gb = [race for race in racecards if race.get("region", "").strip().upper() == "GB"]

        if not racecards_gb:
            print(f"⚠️ No GB races found for {date}. Skipping.")

        return racecards_gb


    except Exception as e:
        print(f"❌ Error fetching racecards for {date}: {e}")
        return []



def fetch_all_racecards():
    """ Fetch racecards for the last 12 months """
    dates = get_past_dates()
    all_racecards = []
    for date in dates:
        print(f"📡 Fetching racecards for {date}...")
        racecards = fetch_racecards(date)
        print(f"🔹 Retrieved {len(racecards)} racecards for {date}")
        
        for race in racecards:
            region = race.get("region", "").strip().lower()  # Ensure consistent lowercase matching
    
            if region != "gb":
                print(f"⚠️ Skipping {race.get('race_name', 'Unknown Race')} - Region: {region}")
                continue  # Skip non-GB races

            print(f"✅ Processing GB race: {race.get('race_name', 'Unknown Race')} at {race.get('course', 'Unknown Course')}")

            
            for runner in race.get("runners", []):
                row = {
                    "race_id": race.get("race_id"),
                    "course": race.get("course"),
                    "course_id": race.get("course_id"),
                    "date": race.get("date"),
                    "off_time": race.get("off_time"),
                    "distance_round": race.get("distance_round"),
                    "distance": race.get("distance"),
                    "distance_f": race.get("distance_f"),
                    "region": race.get("region"),
                    "race_class": race.get("race_class"),
                    "type": race.get("type"),
                    "age_band": race.get("age_band"),
                    "field_size": race.get("field_size"),
                    "going": race.get("going"),
                    "surface": race.get("surface"),
                    "jumps": race.get("jumps"),
                    "horse_id": runner.get("horse_id"),
                    "horse": runner.get("horse"),
                    "age": runner.get("age"),
                    "trainer": runner.get("trainer"),
                    "trainer_id": runner.get("trainer_id"),
                    "trainer_location": runner.get("trainer_location"),
                    "trainer_14_days_runs": safe_extract(runner.get("trainer_14_days", {}), "runs"),
                    "trainer_14_days_wins": safe_extract(runner.get("trainer_14_days", {}), "wins"),
                    "trainer_14_days_percent": safe_extract(runner.get("trainer_14_days", {}), "percent"),
                    "owner": runner.get("owner"),
                    "owner_id": runner.get("owner_id"),
                    "draw": runner.get("draw"),
                    "lbs": runner.get("lbs"),
                    "rpr": runner.get("rpr"),
                    "ts": runner.get("ts"),
                    "jockey": runner.get("jockey"),
                    "jockey_id": runner.get("jockey_id"),
                    "last_run": runner.get("last_run"),
                    "form": runner.get("form"),
                    "trainer_rtf": runner.get("trainer_rtf"),
                    "odds_decimal": extract_odds_decimal(runner.get("odds", [])),
                    # New fields added below
                    "dam": runner.get("dam"),
                    "dam_id": runner.get("dam_id"),
                    "sire": runner.get("sire"),
                    "sire_id": runner.get("sire_id"),
                    "damsire": runner.get("damsire"),
                    "damsire_id": runner.get("damsire_id"),
                }
                all_racecards.append(row)
        
        print(f"✅ Processed {len(all_racecards)} race entries so far")
        time.sleep(1)  # Avoid hitting rate limits
    return all_racecards

if __name__ == "__main__":
    print("📡 Fetching racecards for the past 12 months...")
    race_data = fetch_all_racecards()
    if race_data:
        df = pd.DataFrame(race_data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(df)} racecards to {OUTPUT_FILE}")
    else:
        print("❌ No data retrieved. Check API or credentials.")
