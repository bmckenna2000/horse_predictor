import schedule
import time
import subprocess
import os
from datetime import datetime

# Directory setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts")
FETCH_DIR = os.path.join(SCRIPT_DIR, "fetch")
MERGE_DIR = os.path.join(SCRIPT_DIR, "merge")
# Script paths
FETCH_HORSE_STATS = os.path.join(FETCH_DIR, "fetch_horse_stats.py")
FETCH_JOCKEY_COURSE = os.path.join(FETCH_DIR, "fetch_jockey_course_stats.py")
FETCH_JOCKEY_DISTANCE = os.path.join(FETCH_DIR, "fetch_jockey_distance_stats.py")
FETCH_JOCKEY_OWNER = os.path.join(FETCH_DIR, "fetch_jockey_owner_stats.py")
FETCH_JOCKEY_TRAINER = os.path.join(FETCH_DIR, "fetch_jockey_trainer_stats.py")
FETCH_DAILY_RESULTS = os.path.join(FETCH_DIR, "fetch_daily_results.py")
MERGE_HORSE_DATA = os.path.join(MERGE_DIR, "merge_horse_data.py")
MERGE_RACECARD = os.path.join(MERGE_DIR, "merge_racecard.py")
PREDICTION = os.path.join(SCRIPT_DIR, "prediction.py")
# EVALUATION = os.path.join(SCRIPT_DIR, "evaluate_predictions.py")  # Placeholder for tomorrow

def run_script(script_path, description):
    """Run a Python script and log the result."""
    print(f"📡 Starting {description} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}: {e}")
        print(e.output)

# Fetch functions
def fetch_horse_stats():
    run_script(FETCH_HORSE_STATS, "Fetch Horse Stats (5 workers)")

def fetch_jockey_course_stats():
    run_script(FETCH_JOCKEY_COURSE, "Fetch Jockey Course Stats (5 workers)")

def fetch_jockey_distance_stats():
    run_script(FETCH_JOCKEY_DISTANCE, "Fetch Jockey Distance Stats (5 workers)")

def fetch_jockey_owner_stats():
    run_script(FETCH_JOCKEY_OWNER, "Fetch Jockey Owner Stats (3 workers)")

def fetch_jockey_trainer_stats():
    run_script(FETCH_JOCKEY_TRAINER, "Fetch Jockey Trainer Stats (3 workers)")
    
def fetch_daily_results():
    run_script(FETCH_DAILY_RESULTS, "Fetch Daily Results (4 workers)")
    
# Merge functions
def merge_horse_data():
    run_script(MERGE_HORSE_DATA, "Merge Horse Data")

def merge_racecard():
    run_script(MERGE_RACECARD, "Merge Racecard")

# Schedule tasks (Evening only)
schedule.every().day.at("20:00").do(fetch_horse_stats)          # 5 workers
schedule.every().day.at("20:10").do(fetch_jockey_course_stats)  # 5 workers
schedule.every().day.at("20:20").do(fetch_jockey_distance_stats) # 5 workers, staggered
schedule.every().day.at("20:30").do(fetch_jockey_owner_stats)   # 3 workers, staggered
schedule.every().day.at("20:40").do(fetch_jockey_trainer_stats) # 3 workers, staggered
schedule.every().day.at("21:15").do(fetch_daily_results)
schedule.every().day.at("21:30").do(merge_horse_data)           # Local processing


# Morning: Fetch racecard, merge racecard, predict (to be activated tomorrow)
# schedule.every().day.at("08:00").do(fetch_daily_racecard)      # Single request
# schedule.every().day.at("08:05").do(merge_racecard)            # Local processing
# schedule.every().day.at("08:10").do(run_predictions)           # Local processing
# schedule.every().day.at("21:20").do(evaluate_predictions)      # Local processing, after results

def main():
    print(f"🚀 Starting automated horse racing pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Scheduled tasks (Evening - March 19):")
    print("- 20:00: Fetch Horse Stats (5 workers)")
    print("- 20:10: Fetch Jockey Course Stats (5 workers)")
    print("- 20:20: Fetch Jockey Distance Stats (5 workers)")
    print("- 20:30: Fetch Jockey Owner Stats (3 workers)")
    print("- 20:40: Fetch Jockey Trainer Stats (3 workers)")
    print("- 21:15: Fetch Daily Results (4 workers)")
    print("- 21:30: Merge Horse Data")
    print("Pending tasks (to be added March 20):")
    print("- 08:00: Fetch Daily Racecard")
    print("- 08:05: Merge Racecard")
    print("- 08:10: Run Predictions")
    print("- 21:20: Evaluate Predictions (TBD tomorrow)")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()