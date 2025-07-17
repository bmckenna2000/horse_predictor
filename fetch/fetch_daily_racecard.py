import requests
import os
import sys
import json
import pandas as pd
from datetime import datetime
from requests.auth import HTTPBasicAuth
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import credentials from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import USERNAME, PASSWORD, BASE_URL

# Define output file path
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
today = datetime.today()
OUTPUT_FILE = os.path.join(DATA_DIR, f"daily_racecard.csv")

# Define API endpoint
API_URL = f"{BASE_URL}/racecards/pro"

# Define headers & authentication
auth = HTTPBasicAuth(USERNAME, PASSWORD)

def safe_extract(value, key=None):
    """Extracts specific key from a dictionary or safely converts lists to strings."""
    if isinstance(value, dict):
        return value.get(key, "") if key else json.dumps(value)
    elif isinstance(value, list):
        return ", ".join(str(item) if isinstance(item, str) else json.dumps(item) for item in value)
    return value

def extract_odds_decimal(odds):
    """Safely extract decimal odds whether odds is a list or dictionary."""
    if isinstance(odds, list) and len(odds) > 0:
        return odds[0].get("decimal", "")
    elif isinstance(odds, dict):
        return odds.get("decimal", "")
    return ""

@retry(
    stop=stop_after_attempt(5),  # Max 5 retries
    wait=wait_exponential(multiplier=1, min=2, max=60),  # 2s, 4s, 8s, up to 60s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def fetch_racecards(date):
    """Fetch racecards for a specific date with timeout and retry logic."""
    params = {"date": date}
    print(f"📡 Fetching racecards for {date}...")
    try:
        response = requests.get(API_URL, auth=auth, params=params, timeout=10)

        print(f"📡 Status: {response.status_code}")

        if response.status_code == 429:
            print(f"⚠️ Rate limit exceeded on {date}. Retrying...")
            raise requests.exceptions.RequestException("Rate limit exceeded")

        if response.status_code != 200:
            print(f"❌ Error {response.status_code} for {date}: {response.text}")
            return []

        data = response.json()
        print(f"📜 Full API Response (first 500 chars): {json.dumps(data, indent=2)[:500]}...")

        racecards = data.get("racecards", [])
        available_regions = set(race["region"] for race in racecards)
        print(f"🌍 Available regions for {date}: {available_regions}")

        # Filter for GB races
        racecards_gb = [race for race in racecards if race.get("region", "").strip().upper() == "GB"]
        if not racecards_gb:
            print(f"⚠️ No GB races found for {date}.")
        else:
            print(f"✅ Found {len(racecards_gb)} GB races for {date}")

        return racecards_gb

    except Exception as e:
        print(f"❌ Error fetching racecards for {date}: {e}")
        return []

def fetch_today_racecards():
    """Fetch racecards for today."""
    today_date = datetime.today().strftime("%Y-%m-%d")
    all_racecards = []
    
    racecards = fetch_racecards(today_date)
    print(f"🔹 Retrieved {len(racecards)} racecards for {today_date}")
    
    for race in racecards:
        region = race.get("region", "").strip().lower()
        if region != "gb":
            print(f"⚠️ Skipping {race.get('race_name', 'Unknown Race')} - Region: {region}")
            continue

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
    
    print(f"✅ Processed {len(all_racecards)} race entries for {today_date}")
    return all_racecards

if __name__ == "__main__":
    print("📡 Fetching today's racecards...")
    race_data = fetch_today_racecards()
    if race_data:
        df = pd.DataFrame(race_data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(df)} racecards to {OUTPUT_FILE}")
    else:
        print("❌ No data retrieved. Check API or credentials.")