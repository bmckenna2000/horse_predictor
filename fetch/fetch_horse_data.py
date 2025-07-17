import requests
import os
import sys
import json
import time
import pandas as pd
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor, as_completed

# ✅ Import credentials from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import USERNAME, PASSWORD, BASE_URL

# ✅ Define input and output files
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "data"))
HORSE_IDS_FILE = os.path.join(DATA_DIR, "unique_horses.csv")  # Contains horse IDs
OUTPUT_FILE = os.path.join(DATA_DIR, "horse_distance_stats.csv")

# ✅ Define API endpoint
API_URL_TEMPLATE = f"{BASE_URL}/horses/{{horse_id}}/analysis/distance-times"

# ✅ Load horse IDs
horse_df = pd.read_csv(HORSE_IDS_FILE)
horse_ids = horse_df["horse_id"].unique()

# ✅ Define authentication
auth = HTTPBasicAuth(USERNAME, PASSWORD)

# ✅ Store results
all_horse_stats = []

# ✅ Function to fetch data for a single horse
def fetch_horse_stats(horse_id):
    url = API_URL_TEMPLATE.format(horse_id=horse_id)
    print(f"📡 Fetching data for horse: {horse_id}...")  # Debugging print

    try:
        response = requests.get(url, auth=auth, timeout=10)

        # ✅ Check for API errors
        if response.status_code == 429:
            print(f"⚠️ Rate limit hit for {horse_id}, retrying in 3s...")
            time.sleep(3)
            return fetch_horse_stats(horse_id)  # Retry

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code} for {horse_id}: {response.text}")
            return []

        # ✅ Parse response
        data = response.json()
        distances = data.get("distances", [])

        if not distances:
            print(f"⚠️ No distance data found for horse: {horse_id}")
            return []

        stats = []
        for dist_entry in distances:
            stats.append({
                "horse_id": horse_id,
                "dist": dist_entry.get("dist"),
                "dist_m": dist_entry.get("dist_m"),
                "dist_f": dist_entry.get("dist_f"),
                "runs": dist_entry.get("runs"),
                "1st": dist_entry.get("1st"),
                "2nd": dist_entry.get("2nd"),
                "3rd": dist_entry.get("3rd"),
                "4th": dist_entry.get("4th"),
                "win_%": dist_entry.get("win_%")
            })

        print(f"✅ Successfully fetched stats for horse: {horse_id} ({len(stats)} distances)")

        # ✅ Log the first bit of data retrieved
        print(f"🔍 First entry for {horse_id}: {json.dumps(stats[0], indent=2)}")

        return stats

    except requests.RequestException as e:
        print(f"❌ Request failed for {horse_id}: {e}")
        return []

# ✅ Run requests in parallel
MAX_WORKERS = 2# Reduce workers to avoid API overload
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_horse = {executor.submit(fetch_horse_stats, horse_id): horse_id for horse_id in horse_ids}

    for future in as_completed(future_to_horse):
        horse_id = future_to_horse[future]
        try:
            result = future.result()
            if result:  # Only append if data was retrieved
                all_horse_stats.extend(result)
        except Exception as exc:
            print(f"❌ Error processing {horse_id}: {exc}")

# ✅ Save results to CSV
df = pd.DataFrame(all_horse_stats)

if not df.empty:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} horse distance stats to {OUTPUT_FILE}")
else:
    print("❌ No data retrieved. Check API or credentials.")
