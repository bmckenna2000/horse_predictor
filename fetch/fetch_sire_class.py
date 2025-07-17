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
SIRE_IDS_FILE = os.path.join(DATA_DIR, "unique_sires.csv")  # Contains sire IDs
OUTPUT_FILE = os.path.join(DATA_DIR, "sire_class_stats.csv")

# ✅ Define API endpoint
API_URL_TEMPLATE = f"{BASE_URL}/sires/{{sire_id}}/analysis/classes"

# ✅ Load sire IDs
sire_df = pd.read_csv(SIRE_IDS_FILE)
sire_ids = sire_df["sire_id"].unique()

# ✅ Define authentication
auth = HTTPBasicAuth(USERNAME, PASSWORD)

# ✅ Store results
all_sire_stats = []

# ✅ Function to fetch data for a single sire
def fetch_sire_stats(sire_id):
    url = API_URL_TEMPLATE.format(sire_id=sire_id)
    print(f"📡 Fetching data for sire: {sire_id}...")  # Debugging print

    try:
        response = requests.get(url, auth=auth, timeout=10)

        # ✅ Check for API errors
        if response.status_code == 429:
            print(f"⚠️ Rate limit hit for {sire_id}, retrying in 3s...")
            time.sleep(3)
            return fetch_sire_stats(sire_id)  # Retry

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code} for {sire_id}: {response.text}")
            return []

        # ✅ Parse response
        data = response.json()
        classes = data.get("classes", [])

        if not classes:
            print(f"⚠️ No class data found for sire: {sire_id}")
            return []

        stats = []
        for class_entry in classes:
            stats.append({
                "sire_id": data.get("id"),
                "sire_name": data.get("sire"),
                "total_runners": data.get("total_runners"),
                "class": class_entry.get("class"),
                "runners": class_entry.get("runners"),
                "1st": class_entry.get("1st"),
                "2nd": class_entry.get("2nd"),
                "3rd": class_entry.get("3rd"),
                "4th": class_entry.get("4th"),
                "a/e": class_entry.get("a/e"),
                "win_%": class_entry.get("win_%"),
                "1_pl": class_entry.get("1_pl")
            })

        print(f"✅ Successfully fetched stats for sire: {sire_id} ({len(stats)} classes)")

        # ✅ Log the first bit of data retrieved
        print(f"🔍 First entry for {sire_id}: {json.dumps(stats[0], indent=2)}")

        return stats

    except requests.RequestException as e:
        print(f"❌ Request failed for sire: {sire_id}: {e}")
        return []

# ✅ Run requests in parallel
MAX_WORKERS = 2  # Reduce workers to avoid API overload
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_sire = {executor.submit(fetch_sire_stats, sire_id): sire_id for sire_id in sire_ids}

    for future in as_completed(future_to_sire):
        sire_id = future_to_sire[future]
        try:
            result = future.result()
            if result:  # Only append if data was retrieved
                all_sire_stats.extend(result)
        except Exception as exc:
            print(f"❌ Error processing {sire_id}: {exc}")

# ✅ Save results to CSV
df = pd.DataFrame(all_sire_stats)

if not df.empty:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} sire class stats to {OUTPUT_FILE}")
else:
    print("❌ No data retrieved. Check API or credentials.")