import requests
import os
import sys
import pandas as pd
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from config import USERNAME, PASSWORD, BASE_URL  # Import API credentials

# Define paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "data"))
JOCKEY_FILE = os.path.join(DATA_DIR, "unique_jockeys.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "jockey_distance_stats.csv")

# Setup API headers
auth = HTTPBasicAuth(USERNAME, PASSWORD)

# Load unique jockey IDs
def get_jockey_ids():
    """Load jockey IDs from pre-extracted file to improve speed."""
    if not os.path.exists(JOCKEY_FILE):
        print(f"❌ Jockey data not found: {JOCKEY_FILE}")
        return []

    df_jockeys = pd.read_csv(JOCKEY_FILE)
    if "jockey_id" not in df_jockeys.columns:
        print("❌ 'jockey_id' column missing from jockey data.")
        return []

    return df_jockeys["jockey_id"].dropna().tolist()  # Return unique IDs

# Store results with thread-safe list
all_jockey_stats = []
stats_lock = threading.Lock()

# Fetch Jockey Distance Stats with retry logic
@retry(
    stop=stop_after_attempt(5),  # Max 5 retries
    wait=wait_exponential(multiplier=1, min=2, max=60),  # 2s, 4s, 8s, up to 60s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def fetch_jockey_distance_stats(jockey_id):
    """Fetch distance performance stats for a jockey."""
    url = f"{BASE_URL}/jockeys/{jockey_id}/analysis/distances"
    print(f"📡 Fetching distance stats for jockey {jockey_id}...")

    response = requests.get(url, auth=auth, timeout=5)  # Reduced timeout for faster failure

    if response.status_code == 429:
        print(f"⚠️ Rate limit hit for {jockey_id}, retrying...")
        raise requests.exceptions.RequestException("Rate limit exceeded")

    if response.status_code != 200:
        print(f"❌ API Error {response.status_code} for {jockey_id}: {response.text}")
        return None

    data = response.json()
    if "distances" in data and data["distances"]:
        stats = [{"jockey_id": jockey_id, "total_rides": data.get("total_rides", 0), **distance} 
                 for distance in data["distances"]]
        print(f"✅ Jockey {jockey_id} - {len(stats)} distances found! (Total Rides: {data.get('total_rides', 0)})")
        return stats

    print(f"⚠️ No distance stats found for jockey {jockey_id}")
    return None

def main():
    jockey_ids = get_jockey_ids()
    if not jockey_ids:
        print("⚠️ No jockey IDs found.")
        return

    # Load existing data if available
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
    else:
        df_existing = pd.DataFrame(columns=["jockey_id", "total_rides", "dist", "dist_y", "dist_m", "dist_f", 
                                            "rides", "1st", "2nd", "3rd", "4th", "a/e", "win_%", "1_pl"])

    # Run requests in parallel with 5 workers
    MAX_WORKERS = 2  # Set to your confirmed limit
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_jockey = {executor.submit(fetch_jockey_distance_stats, jockey_id): jockey_id 
                            for jockey_id in jockey_ids}

        for future in as_completed(future_to_jockey):
            jockey_id = future_to_jockey[future]
            try:
                result = future.result()
                if result:  # Only append if data was retrieved
                    with stats_lock:
                        all_jockey_stats.extend(result)
            except Exception as exc:
                print(f"❌ Error processing {jockey_id}: {exc}")

    # Combine new and existing data
    df_new = pd.DataFrame(all_jockey_stats)
    df_list = [df_existing, df_new]
    df_list = [df for df in df_list if not df.empty]  # Remove empty DataFrames

    if df_list:  # If there's valid data, concatenate
        df_updated = pd.concat(df_list).drop_duplicates(subset=["jockey_id", "dist"], keep="last")
    else:
        df_updated = df_existing  # Retain empty DataFrame if no new data

    # Save results
    if not df_updated.empty:
        df_updated.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(df_updated)} jockey distance stats to {OUTPUT_FILE}")
    else:
        print("⚠️ No jockey distance stats found.")

if __name__ == "__main__":
    main()