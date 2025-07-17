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
OUTPUT_FILE = os.path.join(DATA_DIR, "jockey_owner_stats.csv")

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

# Fetch Jockey-Owner Stats with retry logic
@retry(
    stop=stop_after_attempt(5),  # Max 5 retries
    wait=wait_exponential(multiplier=1, min=2, max=60),  # 2s, 4s, 8s, up to 60s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def fetch_jockey_owner_stats(jockey_id):
    """Fetch owner performance stats for a jockey."""
    url = f"{BASE_URL}/jockeys/{jockey_id}/analysis/owners"
    print(f"📡 Fetching owner stats for jockey {jockey_id}...")

    response = requests.get(url, auth=auth, timeout=5)  # Reduced timeout for faster failure

    if response.status_code == 429:
        print(f"⚠️ Rate limit hit for {jockey_id}, retrying...")
        raise requests.exceptions.RequestException("Rate limit exceeded")

    if response.status_code != 200:
        print(f"❌ API Error {response.status_code} for {jockey_id}: {response.text}")
        return None

    data = response.json()
    total_rides = data.get("total_rides", 0)
    if "owners" in data and data["owners"]:
        stats = [{"jockey_id": jockey_id, "total_rides": total_rides, **owner} 
                 for owner in data["owners"]]
        print(f"✅ Jockey {jockey_id} - {len(stats)} owners found! (Total Rides: {total_rides})")
        return stats

    print(f"⚠️ No owner stats found for jockey {jockey_id}")
    return None

def main():
    jockey_ids = get_jockey_ids()
    if not jockey_ids:
        print("⚠️ No jockey IDs found.")
        return

    # Run requests in parallel with 3 workers
    MAX_WORKERS = 2  # Set to your specified limit
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_jockey = {executor.submit(fetch_jockey_owner_stats, jockey_id): jockey_id 
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

    # Save results to CSV
    if all_jockey_stats:
        df = pd.DataFrame(all_jockey_stats)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(df)} jockey-owner stats to {OUTPUT_FILE}")
    else:
        print("⚠️ No jockey-owner stats found.")

if __name__ == "__main__":
    main()