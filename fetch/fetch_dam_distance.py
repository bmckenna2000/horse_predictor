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
DAM_IDS_FILE = os.path.join(DATA_DIR, "unique_dams.csv")  # Contains dam IDs
OUTPUT_FILE = os.path.join(DATA_DIR, "dam_distance_stats.csv")

# ✅ Define API endpoint
API_URL_TEMPLATE = f"{BASE_URL}/dams/{{dam_id}}/analysis/distances"

# ✅ Load dam IDs
dam_df = pd.read_csv(DAM_IDS_FILE)
dam_ids = dam_df["dam_id"].unique()

# ✅ Define authentication
auth = HTTPBasicAuth(USERNAME, PASSWORD)

# ✅ Store results
all_dam_stats = []

# ✅ Function to fetch data for a single dam
def fetch_dam_stats(dam_id):
    url = API_URL_TEMPLATE.format(dam_id=dam_id)
    print(f"📡 Fetching data for dam: {dam_id}...")  # Debugging print

    try:
        response = requests.get(url, auth=auth, timeout=10)

        # ✅ Check for API errors
        if response.status_code == 429:
            print(f"⚠️ Rate limit hit for {dam_id}, retrying in 3s...")
            time.sleep(3)
            return fetch_dam_stats(dam_id)  # Retry

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code} for {dam_id}: {response.text}")
            return []

        # ✅ Parse response
        data = response.json()
        distances = data.get("distances", [])

        if not distances:
            print(f"⚠️ No distance data found for dam: {dam_id}")
            return []

        stats = []
        for dist_entry in distances:
            stats.append({
                "dam_id": data.get("id"),
                "dam_name": data.get("dam"),
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

        print(f"✅ Successfully fetched stats for dam: {dam_id} ({len(stats)} distances)")

        # ✅ Log the first bit of data retrieved
        print(f"🔍 First entry for {dam_id}: {json.dumps(stats[0], indent=2)}")

        return stats

    except requests.RequestException as e:
        print(f"❌ Request failed for dam: {dam_id}: {e}")
        return []

# ✅ Run requests in parallel
MAX_WORKERS = 2  # Reduce workers to avoid API overload
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_dam = {executor.submit(fetch_dam_stats, dam_id): dam_id for dam_id in dam_ids}

    for future in as_completed(future_to_dam):
        dam_id = future_to_dam[future]
        try:
            result = future.result()
            if result:  # Only append if data was retrieved
                all_dam_stats.extend(result)
        except Exception as exc:
            print(f"❌ Error processing {dam_id}: {exc}")

# ✅ Save results to CSV
df = pd.DataFrame(all_dam_stats)

if not df.empty:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} dam distance stats to {OUTPUT_FILE}")
else:
    print("❌ No data retrieved. Check API or credentials.")