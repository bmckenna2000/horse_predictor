import pandas as pd
import os
from datetime import datetime

# Directory setup
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DAILY_RACECARD_FILE = os.path.join(DATA_DIR, "daily_racecard.csv")
RETRAIN_DATA_FILE = os.path.join(DATA_DIR, "retrain_data.csv")

def load_daily_racecard():
    """Load daily racecard data."""
    try:
        data = pd.read_csv(DAILY_RACECARD_FILE, low_memory=False)
        print(f"✅ Loaded {len(data)} rows from {DAILY_RACECARD_FILE}")
        print(f"Columns: {data.columns.tolist()}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: {DAILY_RACECARD_FILE} not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading {DAILY_RACECARD_FILE}: {e}")
        return None

def load_retrain_data():
    """Load existing retrain data, if it exists."""
    if os.path.exists(RETRAIN_DATA_FILE):
        try:
            data = pd.read_csv(RETRAIN_DATA_FILE, low_memory=False)
            print(f"✅ Loaded {len(data)} rows from {RETRAIN_DATA_FILE}")
            return data
        except Exception as e:
            print(f"❌ Error loading {RETRAIN_DATA_FILE}: {e}")
            return pd.DataFrame()
    else:
        print(f"ℹ️ {RETRAIN_DATA_FILE} does not exist. Will create it.")
        return pd.DataFrame()

def append_to_retrain_data(daily_data, retrain_data):
    """Append daily data to retrain data, avoiding duplicates."""
    if daily_data is None or daily_data.empty:
        print("❌ No daily data to append.")
        return retrain_data

    # Ensure date is in correct format
    daily_data['date'] = pd.to_datetime(daily_data['date'], errors='coerce')
    
    # Create a unique key for deduplication (e.g., race_id + horse_id)
    daily_data['unique_key'] = daily_data['race_id'].astype(str) + '_' + daily_data['horse_id'].astype(str)
    if not retrain_data.empty:
        retrain_data['unique_key'] = retrain_data['race_id'].astype(str) + '_' + retrain_data['horse_id'].astype(str)
    
    # Filter out duplicates
    if not retrain_data.empty:
        existing_keys = set(retrain_data['unique_key'])
        new_data = daily_data[~daily_data['unique_key'].isin(existing_keys)].copy()
        print(f"ℹ️ Found {len(daily_data) - len(new_data)} duplicate entries. Keeping {len(new_data)} new entries.")
    else:
        new_data = daily_data.copy()
    
    # Drop temporary unique_key column
    new_data = new_data.drop(columns=['unique_key'], errors='ignore')
    
    # Append new data
    if retrain_data.empty:
        updated_retrain_data = new_data
    else:
        updated_retrain_data = pd.concat([retrain_data.drop(columns=['unique_key'], errors='ignore'), new_data], ignore_index=True)
    
    print(f"✅ Appended {len(new_data)} new rows. Total rows in retrain_data: {len(updated_retrain_data)}")
    return updated_retrain_data

def save_retrain_data(data):
    """Save retrain data to CSV."""
    if data.empty:
        print(f"❌ No data to save to {RETRAIN_DATA_FILE}")
        return
    try:
        data.to_csv(RETRAIN_DATA_FILE, index=False)
        print(f"✅ Saved {len(data)} rows to {RETRAIN_DATA_FILE}")
    except Exception as e:
        print(f"❌ Error saving {RETRAIN_DATA_FILE}: {e}")

def main():
    print(f"📡 Starting racecard append process for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    # Load data
    daily_data = load_daily_racecard()
    retrain_data = load_retrain_data()
    
    # Append new data
    updated_retrain_data = append_to_retrain_data(daily_data, retrain_data)
    
    # Save result
    save_retrain_data(updated_retrain_data)
    
    print("✅ Append process completed")

if __name__ == "__main__":
    main()