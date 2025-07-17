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
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
DAMSIRE_IDS_FILE = os.path.join(DATA_DIR, "unique_damsires.csv")  # Contains damsire IDs
OUTPUT_FILE = os.path.join(DATA_DIR, "damsire_distance_stats.csv")

# ✅ Define API endpoint
API_URL_TEMPLATE = f"{BASE_URL}/damsires/{{damsire_id}}/analysis/distances"

# ✅ Load damsire IDs
damsire_df = pd.read_csv(DAMSIRE_IDS_FILE)
damsire_ids = damsire_df["damsire_id"].unique()

# ✅ Define authentication
auth = HTTPBasicAuth(USERNAME, PASSWORD)

# ✅ Store results
all_damsire_stats = []

# ✅ Function to fetch data for a single damsire
def fetch_damsire_stats(damsire_id):
    url = API_URL_TEMPLATE.format(damsire_id=damsire_id)
    print(f"📡 Fetching data for damsire: {damsire_id}...")  # Debugging print

    try:
        response = requests.get(url, auth=auth, timeout=10)

        # ✅ Check for API errors
        if response.status_code == 429:
            print(f"⚠️ Rate limit hit for {damsire_id}, retrying in 3s...")
            time.sleep(3)
            return fetch_damsire_stats(damsire_id)  # Retry

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code} for {damsire_id}: {response.text}")
            return []

        # ✅ Parse response
        data = response.json()
        distances = data.get("distances", [])

        if not distances:
            print(f"⚠️ No distance data found for damsire: {damsire_id}")
            return []

        stats = []
        for dist_entry in distances:
            stats.append({
                "damsire_id": data.get("id"),
                "damsire_name": data.get("damsire"),
                "total_runners": data.get("total_runners"),
                "dist_f": dist_entry.get("dist_f"),
                "runners": dist_entry.get("runners"),
                "1st": dist_entry.get("1st"),
                "2nd": dist_entry.get("2nd"),
                "3rd": dist_entry.get("3rd"),
                "4th": dist_entry.get("4th"),
                "a/e": dist_entry.get("a/e"),
                "win_%": dist_entry.get("win_%"),
                "1_pl": dist_entry.get("1_pl")
            })

        print(f"✅ Successfully fetched stats for damsire: {damsire_id} ({len(stats)} distances)")

        # ✅ Log the first bit of data retrieved
        print(f"🔍 First entry for {damsire_id}: {json.dumps(stats[0], indent=2)}")

        return stats

    except requests.RequestException as e:
        print(f"❌ Request failed for damsire: {damsire_id}: {e}")
        return []

# ✅ Run requests in parallel
MAX_WORKERS = 2  # Reduce workers to avoid API overload
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_damsire = {executor.submit(fetch_damsire_stats, damsire_id): damsire_id for damsire_id in damsire_ids}

    for future in as_completed(future_to_damsire):
        damsire_id = future_to_damsire[future]
        try:
            result = future.result()
            if result:  # Only append if data was retrieved
                all_damsire_stats.extend(result)
        except Exception as exc:
            print(f"❌ Error processing {damsire_id}: {exc}")

# ✅ Save results to CSV
df = pd.DataFrame(all_damsire_stats)

if not df.empty:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} damsire distance stats to {OUTPUT_FILE}")
else:
    print("❌ No data retrieved. Check API or credentials.")