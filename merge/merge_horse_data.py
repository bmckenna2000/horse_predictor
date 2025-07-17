import pandas as pd
import os
import re

# Define the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Load CSV files
racecards = pd.read_csv(os.path.join(DATA_DIR, 'racecards.csv'), low_memory=False)
jockey_course_stats = pd.read_csv(os.path.join(DATA_DIR, 'jockey_course_stats.csv'))
jockey_owner_stats = pd.read_csv(os.path.join(DATA_DIR, 'jockey_owner_stats.csv'))
jockey_trainer_stats = pd.read_csv(os.path.join(DATA_DIR, 'jockey_trainer_stats.csv'))
jockey_distance_stats = pd.read_csv(os.path.join(DATA_DIR, 'jockey_distance_stats.csv'))
horse_distance_stats = pd.read_csv(os.path.join(DATA_DIR, 'horse_distance_stats.csv'))
historical_position = pd.read_csv(os.path.join(DATA_DIR, 'historical_position.csv'))
damsire_distance_stats = pd.read_csv(os.path.join(DATA_DIR, 'damsire_distance_stats.csv'))
dam_distance_stats = pd.read_csv(os.path.join(DATA_DIR, 'dam_distance_stats.csv'))
sire_distance_stats = pd.read_csv(os.path.join(DATA_DIR, 'sire_distance_stats.csv'))
damsire_class_stats = pd.read_csv(os.path.join(DATA_DIR, 'damsire_class_stats.csv'))
dam_class_stats = pd.read_csv(os.path.join(DATA_DIR, 'dam_class_stats.csv'))
sire_class_stats = pd.read_csv(os.path.join(DATA_DIR, 'sire_class_stats.csv'))

# --- Process jockey_course_stats ---
jockey_course_stats = jockey_course_stats.rename(
    columns=lambda x: x if x in ['jockey_id', 'course_id'] else f"jockey_course_{x}"
)

# --- Process jockey_owner_stats ---
jockey_owner_stats = jockey_owner_stats.rename(
    columns=lambda x: x if x in ['jockey_id', 'owner_id'] else f"jockey_owner_{x}"
)

# --- Process jockey_trainer_stats ---
jockey_trainer_stats = jockey_trainer_stats.rename(
    columns=lambda x: x if x in ['jockey_id', 'trainer_id'] else f"jockey_trainer_{x}"
)

# --- Process jockey_distance_stats ---
jockey_distance_stats['dist_f'] = jockey_distance_stats['dist_f'].astype(str).str.rstrip('f').astype(float)
jockey_distance_stats = jockey_distance_stats.rename(columns={'dist_f': 'distance_f'})
jockey_distance_stats = jockey_distance_stats.rename(
    columns=lambda x: x if x in ['jockey_id', 'distance_f'] else f"jockey_distance_{x}"
)

# --- Process horse_distance_stats ---
horse_distance_stats['dist_f'] = horse_distance_stats['dist_f'].astype(str).str.rstrip('f').astype(float)
horse_distance_stats = horse_distance_stats.rename(columns={'dist_f': 'distance_f'})
horse_distance_stats = horse_distance_stats.rename(
    columns=lambda x: x if x in ['horse_id', 'distance_f'] else f"horse_distance_{x}"
)

# --- Process damsire_distance_stats ---
damsire_distance_stats['dist_f'] = damsire_distance_stats['dist_f'].astype(str).str.rstrip('f').astype(float)
damsire_distance_stats = damsire_distance_stats.rename(columns={'dist_f': 'distance_f'})
damsire_distance_stats = damsire_distance_stats.rename(
    columns=lambda x: x if x in ['damsire_id', 'distance_f'] else f"damsire_distance_{x}"
)

# --- Process dam_distance_stats ---
dam_distance_stats['dist_f'] = dam_distance_stats['dist_f'].astype(str).str.rstrip('f').astype(float)
dam_distance_stats = dam_distance_stats.rename(columns={'dist_f': 'distance_f'})
dam_distance_stats = dam_distance_stats.rename(
    columns=lambda x: x if x in ['dam_id', 'distance_f'] else f"dam_distance_{x}"
)

# --- Process sire_distance_stats ---
sire_distance_stats['dist_f'] = sire_distance_stats['dist_f'].astype(str).str.rstrip('f').astype(float)
sire_distance_stats = sire_distance_stats.rename(columns={'dist_f': 'distance_f'})
sire_distance_stats = sire_distance_stats.rename(
    columns=lambda x: x if x in ['sire_id', 'distance_f'] else f"sire_distance_{x}"
)

# --- Process damsire_class_stats ---
damsire_class_stats = damsire_class_stats.rename(
    columns=lambda x: 'race_class' if x == 'class' else (x if x == 'damsire_id' else f"damsire_class_{x}")
)
damsire_class_stats['race_class'] = pd.to_numeric(
    damsire_class_stats['race_class'].astype(str).str.replace("Class ", "", regex=False),
    errors='coerce'
).fillna(0).astype(int)

# --- Process dam_class_stats ---
dam_class_stats = dam_class_stats.rename(
    columns=lambda x: 'race_class' if x == 'class' else (x if x == 'dam_id' else f"dam_class_{x}")
)
dam_class_stats['race_class'] = pd.to_numeric(
    dam_class_stats['race_class'].astype(str).str.replace("Class ", "", regex=False),
    errors='coerce'
).fillna(0).astype(int)

# --- Process sire_class_stats ---
sire_class_stats = sire_class_stats.rename(
    columns=lambda x: 'race_class' if x == 'class' else (x if x == 'sire_id' else f"sire_class_{x}")
)
sire_class_stats['race_class'] = pd.to_numeric(
    sire_class_stats['race_class'].astype(str).str.replace("Class ", "", regex=False),
    errors='coerce'
).fillna(0).astype(int)

# --- Process race_class before merges ---
if 'race_class' in racecards.columns:
    racecards['race_class'] = racecards['race_class'].str.replace("Class ", "", regex=False).astype(int)
else:
    print("Warning: Column 'race_class' not found in racecards.")

# --- Merge DataFrames ---
racecards = racecards.merge(jockey_course_stats, on=['jockey_id', 'course_id'], how='left')
racecards = racecards.merge(jockey_owner_stats, on=['jockey_id', 'owner_id'], how='left')
racecards = racecards.merge(jockey_trainer_stats, on=['jockey_id', 'trainer_id'], how='left')
racecards = racecards.merge(jockey_distance_stats, on=['jockey_id', 'distance_f'], how='left')
racecards = racecards.merge(horse_distance_stats, on=['horse_id', 'distance_f'], how='left')
racecards = racecards.merge(damsire_distance_stats, on=['damsire_id', 'distance_f'], how='left')
racecards = racecards.merge(dam_distance_stats, on=['dam_id', 'distance_f'], how='left')
racecards = racecards.merge(sire_distance_stats, on=['sire_id', 'distance_f'], how='left')
racecards = racecards.merge(historical_position[['race_id', 'horse_id', 'position']],
                            on=['race_id', 'horse_id'],
                            how='left')
racecards = racecards.merge(damsire_class_stats, on=['damsire_id', 'race_class'], how='left')
racecards = racecards.merge(dam_class_stats, on=['dam_id', 'race_class'], how='left')
racecards = racecards.merge(sire_class_stats, on=['sire_id', 'race_class'], how='left')

# --- Add New Features ---
# Class win rates
racecards['damsire_class_win_rate'] = (racecards['damsire_class_1st'] / racecards['damsire_class_runners']).fillna(0).round(2)
racecards['dam_class_win_rate'] = (racecards['dam_class_1st'] / racecards['dam_class_runners']).fillna(0).round(2)
racecards['sire_class_win_rate'] = (racecards['sire_class_1st'] / racecards['sire_class_runners']).fillna(0).round(2)

# Distance win rates
racecards['damsire_distance_win_rate'] = (racecards['damsire_distance_1st'] / racecards['damsire_distance_runners']).fillna(0).round(2)
racecards['dam_distance_win_rate'] = (racecards['dam_distance_1st'] / racecards['dam_distance_runners']).fillna(0).round(2)
racecards['sire_distance_win_rate'] = (racecards['sire_distance_1st'] / racecards['sire_distance_runners']).fillna(0).round(2)

# Class place rates (top 3)
racecards['damsire_class_place_rate'] = (
    (racecards['damsire_class_1st'] + racecards['damsire_class_2nd'] + racecards['damsire_class_3rd']) / 
    racecards['damsire_class_runners']
).fillna(0).round(2)
racecards['dam_class_place_rate'] = (
    (racecards['dam_class_1st'] + racecards['dam_class_2nd'] + racecards['dam_class_3rd']) / 
    racecards['dam_class_runners']
).fillna(0).round(2)
racecards['sire_class_place_rate'] = (
    (racecards['sire_class_1st'] + racecards['sire_class_2nd'] + racecards['sire_class_3rd']) / 
    racecards['sire_class_runners']
).fillna(0).round(2)

# Distance place rates (top 3)
racecards['damsire_distance_place_rate'] = (
    (racecards['damsire_distance_1st'] + racecards['damsire_distance_2nd'] + racecards['damsire_distance_3rd']) / 
    racecards['damsire_distance_runners']
).fillna(0).round(2)
racecards['dam_distance_place_rate'] = (
    (racecards['dam_distance_1st'] + racecards['dam_distance_2nd'] + racecards['dam_distance_3rd']) / 
    racecards['dam_distance_runners']
).fillna(0).round(2)
racecards['sire_distance_place_rate'] = (
    (racecards['sire_distance_1st'] + racecards['sire_distance_2nd'] + racecards['sire_distance_3rd']) / 
    racecards['sire_distance_runners']
).fillna(0).round(2)

# Above expected differential
racecards['damsire_class_ae_diff'] = (racecards['damsire_class_a/e'] - 1).fillna(0).round(2)
racecards['dam_class_ae_diff'] = (racecards['dam_class_a/e'] - 1).fillna(0).round(2)
racecards['sire_class_ae_diff'] = (racecards['sire_class_a/e'] - 1).fillna(0).round(2)
racecards['damsire_distance_ae_diff'] = (racecards['damsire_distance_a/e'] - 1).fillna(0).round(2)
racecards['dam_distance_ae_diff'] = (racecards['dam_distance_a/e'] - 1).fillna(0).round(2)
racecards['sire_distance_ae_diff'] = (racecards['sire_distance_a/e'] - 1).fillna(0).round(2)

# Combined ancestor win rates
racecards['ancestor_class_win_rate'] = (
    (racecards['damsire_class_win_rate'] + racecards['dam_class_win_rate'] + racecards['sire_class_win_rate']) / 3
).fillna(0).round(2)
racecards['ancestor_distance_win_rate'] = (
    (racecards['damsire_distance_win_rate'] + racecards['dam_distance_win_rate'] + racecards['sire_distance_win_rate']) / 3
).fillna(0).round(2)

# Jockey-Horse synergy (distance win rate)
racecards['jockey_horse_distance_win_rate'] = (
    (racecards['jockey_distance_1st'] + racecards['horse_distance_1st']) / 
    (racecards['jockey_distance_rides'] + racecards['horse_distance_runs'])
).fillna(0).round(2)

# New Damsire, Dam, Sire Features
# Consistency Score (Top 4 Finishes)
racecards['damsire_class_consistency'] = (
    (racecards['damsire_class_1st'] + racecards['damsire_class_2nd'] + racecards['damsire_class_3rd'] + racecards['damsire_class_4th']) / 
    racecards['damsire_class_runners']
).fillna(0).round(2)
racecards['dam_class_consistency'] = (
    (racecards['dam_class_1st'] + racecards['dam_class_2nd'] + racecards['dam_class_3rd'] + racecards['dam_class_4th']) / 
    racecards['dam_class_runners']
).fillna(0).round(2)
racecards['sire_class_consistency'] = (
    (racecards['sire_class_1st'] + racecards['sire_class_2nd'] + racecards['sire_class_3rd'] + racecards['sire_class_4th']) / 
    racecards['sire_class_runners']
).fillna(0).round(2)
racecards['damsire_distance_consistency'] = (
    (racecards['damsire_distance_1st'] + racecards['damsire_distance_2nd'] + racecards['damsire_distance_3rd'] + racecards['damsire_distance_4th']) / 
    racecards['damsire_distance_runners']
).fillna(0).round(2)
racecards['dam_distance_consistency'] = (
    (racecards['dam_distance_1st'] + racecards['dam_distance_2nd'] + racecards['dam_distance_3rd'] + racecards['dam_distance_4th']) / 
    racecards['dam_distance_runners']
).fillna(0).round(2)
racecards['sire_distance_consistency'] = (
    (racecards['sire_distance_1st'] + racecards['sire_distance_2nd'] + racecards['sire_distance_3rd'] + racecards['sire_distance_4th']) / 
    racecards['sire_distance_runners']
).fillna(0).round(2)

# Win-to-Place Ratio
racecards['damsire_class_win_to_place'] = (
    racecards['damsire_class_1st'] / 
    (racecards['damsire_class_1st'] + racecards['damsire_class_2nd'] + racecards['damsire_class_3rd'])
).fillna(0).replace([float('inf'), -float('inf')], 0).round(2)
racecards['dam_class_win_to_place'] = (
    racecards['dam_class_1st'] / 
    (racecards['dam_class_1st'] + racecards['dam_class_2nd'] + racecards['dam_class_3rd'])
).fillna(0).replace([float('inf'), -float('inf')], 0).round(2)
racecards['sire_class_win_to_place'] = (
    racecards['sire_class_1st'] / 
    (racecards['sire_class_1st'] + racecards['sire_class_2nd'] + racecards['sire_class_3rd'])
).fillna(0).replace([float('inf'), -float('inf')], 0).round(2)
racecards['damsire_distance_win_to_place'] = (
    racecards['damsire_distance_1st'] / 
    (racecards['damsire_distance_1st'] + racecards['damsire_distance_2nd'] + racecards['damsire_distance_3rd'])
).fillna(0).replace([float('inf'), -float('inf')], 0).round(2)
racecards['dam_distance_win_to_place'] = (
    racecards['dam_distance_1st'] / 
    (racecards['dam_distance_1st'] + racecards['dam_distance_2nd'] + racecards['dam_distance_3rd'])
).fillna(0).replace([float('inf'), -float('inf')], 0).round(2)
racecards['sire_distance_win_to_place'] = (
    racecards['sire_distance_1st'] / 
    (racecards['sire_distance_1st'] + racecards['sire_distance_2nd'] + racecards['sire_distance_3rd'])
).fillna(0).replace([float('inf'), -float('inf')], 0).round(2)

# Above Expected Impact
racecards['damsire_class_ae_impact'] = (
    (racecards['damsire_class_a/e'] - 1) * racecards['damsire_class_runners']
).fillna(0).round(2)
racecards['dam_class_ae_impact'] = (
    (racecards['dam_class_a/e'] - 1) * racecards['dam_class_runners']
).fillna(0).round(2)
racecards['sire_class_ae_impact'] = (
    (racecards['sire_class_a/e'] - 1) * racecards['sire_class_runners']
).fillna(0).round(2)
racecards['damsire_distance_ae_impact'] = (
    (racecards['damsire_distance_a/e'] - 1) * racecards['damsire_distance_runners']
).fillna(0).round(2)
racecards['dam_distance_ae_impact'] = (
    (racecards['dam_distance_a/e'] - 1) * racecards['dam_distance_runners']
).fillna(0).round(2)
racecards['sire_distance_ae_impact'] = (
    (racecards['sire_distance_a/e'] - 1) * racecards['sire_distance_runners']
).fillna(0).round(2)

# Runner Volume Ratio
racecards['damsire_class_runner_ratio'] = (
    racecards['damsire_class_runners'] / racecards['damsire_class_total_runners']
).fillna(0).round(2)
racecards['dam_class_runner_ratio'] = (
    racecards['dam_class_runners'] / racecards['dam_class_total_runners']
).fillna(0).round(2)
racecards['sire_class_runner_ratio'] = (
    racecards['sire_class_runners'] / racecards['sire_class_total_runners']
).fillna(0).round(2)
racecards['damsire_distance_runner_ratio'] = (
    racecards['damsire_distance_runners'] / racecards['damsire_distance_total_runners']
).fillna(0).round(2)
racecards['dam_distance_runner_ratio'] = (
    racecards['dam_distance_runners'] / racecards['dam_distance_total_runners']
).fillna(0).round(2)
racecards['sire_distance_runner_ratio'] = (
    racecards['sire_distance_runners'] / racecards['sire_distance_total_runners']
).fillna(0).round(2)

# --- Print sample of new class and distance columns with features (before form) ---
class_columns = [col for col in racecards.columns if col.startswith(('damsire_class_', 'dam_class_', 'sire_class_'))]
distance_columns = [col for col in racecards.columns if col.startswith(('damsire_distance_', 'dam_distance_', 'sire_distance_'))]
new_feature_columns = [
    'damsire_class_win_rate', 'dam_class_win_rate', 'sire_class_win_rate',
    'damsire_distance_win_rate', 'dam_distance_win_rate', 'sire_distance_win_rate',
    'damsire_class_place_rate', 'dam_class_place_rate', 'sire_class_place_rate',
    'damsire_distance_place_rate', 'dam_distance_place_rate', 'sire_distance_place_rate',
    'damsire_class_ae_diff', 'dam_class_ae_diff', 'sire_class_ae_diff',
    'damsire_distance_ae_diff', 'dam_distance_ae_diff', 'sire_distance_ae_diff',
    'ancestor_class_win_rate', 'ancestor_distance_win_rate',
    'jockey_horse_distance_win_rate',
    'damsire_class_consistency', 'dam_class_consistency', 'sire_class_consistency',
    'damsire_distance_consistency', 'dam_distance_consistency', 'sire_distance_consistency',
    'damsire_class_win_to_place', 'dam_class_win_to_place', 'sire_class_win_to_place',
    'damsire_distance_win_to_place', 'dam_distance_win_to_place', 'sire_distance_win_to_place',
    'damsire_class_ae_impact', 'dam_class_ae_impact', 'sire_class_ae_impact',
    'damsire_distance_ae_impact', 'dam_distance_ae_impact', 'sire_distance_ae_impact',
    'damsire_class_runner_ratio', 'dam_class_runner_ratio', 'sire_class_runner_ratio',
    'damsire_distance_runner_ratio', 'dam_distance_runner_ratio', 'sire_distance_runner_ratio'
]
print("Sample of new class and distance columns with features (before form processing):")
print(racecards[['race_id', 'horse_id', 'damsire_id', 'dam_id', 'sire_id', 'race_class'] + 
                class_columns + distance_columns + new_feature_columns].head().to_string(index=False))

# --- Process date column ---
if 'date' in racecards.columns:
    racecards['date'] = pd.to_datetime(racecards['date'])
    racecards['date'] = racecards['date'].dt.strftime("%b %d, %Y")
else:
    raise KeyError("Column 'date' not found in racecards. Please verify the column name.")

# --- Drop unnecessary columns ---
columns_to_drop = ['distance_round', 'distance', "region", "age_band", "trainer_location",
                  "jockey_course_course", "jockey_course_region", "jockey_owner_owner",
                  "jockey_trainer_trainer", "jockey_distance_dist", "jockey_distance_dist_y",
                  "jockey_distance_dist_m", "horse_distance_dist", "horse_distance_dist_m",
                  "damsire", "dam", "sire"]
racecards = racecards.drop(columns=columns_to_drop, errors='ignore')

# --- Map categorical columns ---
type_mapping = {'Hurdle': 1, 'Chase': 2, 'NH Flat': 3, 'Flat': 4}
racecards["type"] = racecards["type"].map(type_mapping)

going_mapping = {
    'Good': 1, 'Good To Soft': 2, 'Standard': 3, 'Soft': 4, 'Standard To Slow': 5,
    'Heavy': 6, 'Abandoned': 7, 'Firm': 8, 'Good To Firm': 9
}
racecards['going'] = racecards['going'].map(going_mapping).fillna(0).astype(int)

surface_mapping = {'Turf': 1, 'AW': 2}
racecards['surface'] = racecards['surface'].map(surface_mapping).fillna(0).astype(int)

# --- Clean jumps column ---
racecards['jumps'] = racecards['jumps'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)

# --- Process trainer_14_days_percent ---
if 'trainer_14_days_percent' in racecards.columns:
    racecards['trainer_14_days_percent'] = racecards['trainer_14_days_percent'].astype(float) / 100

# --- Convert specific columns to numeric ---
for col in ['draw', 'lbs', 'rpr', 'ts']:
    racecards[col] = pd.to_numeric(racecards[col], errors='coerce').fillna(0)

# --- Process last_run ---
racecards['last_run'] = (
    racecards['last_run']
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .fillna(0)
    .astype(int)
)

# --- Process form column and add recent_form_score ---
def extract_top4_form(form_str):
    if not isinstance(form_str, str):
        form_str = str(form_str)
    tokens = re.split(r'[-/]', form_str)
    results = []
    for token in tokens:
        token = token.strip()
        if token == "":
            continue
        if token.isdigit():
            results.extend([int(ch) for ch in token])
        else:
            if token[0].isalpha():
                if token[0] == 'P':
                    if len(token) > 1:
                        results.append(0)
                        for ch in token[1:]:
                            if ch.isdigit():
                                results.append(int(ch))
                    else:
                        results.append(0)
                else:
                    results.append(0)
            else:
                for ch in token:
                    if ch.isdigit():
                        results.append(int(ch))
                    else:
                        results.append(0)
        if len(results) >= 4:
            break
    results = results[:4]
    while len(results) < 4:
        results.append(0)
    return results

racecards['form_top4'] = racecards['form'].apply(extract_top4_form)
form_df = pd.DataFrame(racecards['form_top4'].tolist(), columns=['form_1', 'form_2', 'form_3', 'form_4'])
racecards = racecards.join(form_df)
racecards.drop(columns=['form', 'form_top4'], inplace=True)

# Add recent_form_score after form columns are created
racecards['recent_form_score'] = (
    (racecards['form_1'] + racecards['form_2'] + racecards['form_3'] + racecards['form_4']) / 4
).fillna(0).round(2)

# --- Print sample with all features including recent_form_score ---
new_feature_columns = [
    'damsire_class_win_rate', 'dam_class_win_rate', 'sire_class_win_rate',
    'damsire_distance_win_rate', 'dam_distance_win_rate', 'sire_distance_win_rate',
    'damsire_class_place_rate', 'dam_class_place_rate', 'sire_class_place_rate',
    'damsire_distance_place_rate', 'dam_distance_place_rate', 'sire_distance_place_rate',
    'damsire_class_ae_diff', 'dam_class_ae_diff', 'sire_class_ae_diff',
    'damsire_distance_ae_diff', 'dam_distance_ae_diff', 'sire_distance_ae_diff',
    'ancestor_class_win_rate', 'ancestor_distance_win_rate',
    'jockey_horse_distance_win_rate', 'recent_form_score',
    'damsire_class_consistency', 'dam_class_consistency', 'sire_class_consistency',
    'damsire_distance_consistency', 'dam_distance_consistency', 'sire_distance_consistency',
    'damsire_class_win_to_place', 'dam_class_win_to_place', 'sire_class_win_to_place',
    'damsire_distance_win_to_place', 'dam_distance_win_to_place', 'sire_distance_win_to_place',
    'damsire_class_ae_impact', 'dam_class_ae_impact', 'sire_class_ae_impact',
    'damsire_distance_ae_impact', 'dam_distance_ae_impact', 'sire_distance_ae_impact',
    'damsire_class_runner_ratio', 'dam_class_runner_ratio', 'sire_class_runner_ratio',
    'damsire_distance_runner_ratio', 'dam_distance_runner_ratio', 'sire_distance_runner_ratio'
]
print("Sample of new class and distance columns with all new features:")
print(racecards[['race_id', 'horse_id', 'damsire_id', 'dam_id', 'sire_id', 'race_class'] + 
                class_columns + distance_columns + new_feature_columns].head().to_string(index=False))

# --- Update numeric columns list with new features ---
numeric_cols = [
    'distance_f', 'odds_decimal', 'trainer_14_days_percent', 'draw', 'lbs', 'rpr', 'ts',
    'last_run', 'jockey_course_total_rides', 'jockey_course_rides', 'jockey_course_1st',
    'jockey_course_2nd', 'jockey_course_3rd', 'jockey_course_4th', 'jockey_course_a/e',
    'jockey_course_win_%', 'jockey_course_1_pl', 'jockey_owner_total_rides', 'jockey_owner_rides',
    'jockey_owner_1st', 'jockey_owner_2nd', 'jockey_owner_3rd', 'jockey_owner_4th',
    'jockey_owner_a/e', 'jockey_owner_win_%', 'jockey_owner_1_pl', 'jockey_trainer_total_rides',
    'jockey_trainer_rides', 'jockey_trainer_1st', 'jockey_trainer_2nd', 'jockey_trainer_3rd',
    'jockey_trainer_4th', 'jockey_trainer_a/e', 'jockey_trainer_win_%', 'jockey_trainer_1_pl',
    'jockey_distance_total_rides', 'jockey_distance_rides', 'jockey_distance_1st',
    'jockey_distance_2nd', 'jockey_distance_3rd', 'jockey_distance_4th', 'jockey_distance_a/e',
    'jockey_distance_win_%', 'jockey_distance_1_pl', 'horse_distance_runs', 'horse_distance_1st',
    'horse_distance_2nd', 'horse_distance_3rd', 'horse_distance_4th', 'horse_distance_win_%',
    'damsire_distance_total_runners', 'damsire_distance_runners', 'damsire_distance_1st',
    'damsire_distance_2nd', 'damsire_distance_3rd', 'damsire_distance_4th', 'damsire_distance_a/e',
    'damsire_distance_win_%', 'damsire_distance_1_pl', 'dam_distance_total_runners',
    'dam_distance_runners', 'dam_distance_1st', 'dam_distance_2nd', 'dam_distance_3rd',
    'dam_distance_4th', 'dam_distance_a/e', 'dam_distance_win_%', 'dam_distance_1_pl',
    'sire_distance_total_runners', 'sire_distance_runners', 'sire_distance_1st',
    'sire_distance_2nd', 'sire_distance_3rd', 'sire_distance_4th', 'sire_distance_a/e',
    'sire_distance_win_%', 'sire_distance_1_pl',
    'damsire_class_total_runners', 'damsire_class_runners', 'damsire_class_1st',
    'damsire_class_2nd', 'damsire_class_3rd', 'damsire_class_4th', 'damsire_class_a/e',
    'damsire_class_win_%', 'damsire_class_1_pl', 'dam_class_total_runners',
    'dam_class_runners', 'dam_class_1st', 'dam_class_2nd', 'dam_class_3rd',
    'dam_class_4th', 'dam_class_a/e', 'dam_class_win_%', 'dam_class_1_pl',
    'sire_class_total_runners', 'sire_class_runners', 'sire_class_1st',
    'sire_class_2nd', 'sire_class_3rd', 'sire_class_4th', 'sire_class_a/e',
    'sire_class_win_%', 'sire_class_1_pl', 'form_1', 'form_2', 'form_3', 'form_4',
    'type', 'race_class', 'position',
    # New features
    'damsire_class_win_rate', 'dam_class_win_rate', 'sire_class_win_rate',
    'damsire_distance_win_rate', 'dam_distance_win_rate', 'sire_distance_win_rate',
    'damsire_class_place_rate', 'dam_class_place_rate', 'sire_class_place_rate',
    'damsire_distance_place_rate', 'dam_distance_place_rate', 'sire_distance_place_rate',
    'damsire_class_ae_diff', 'dam_class_ae_diff', 'sire_class_ae_diff',
    'damsire_distance_ae_diff', 'dam_distance_ae_diff', 'sire_distance_ae_diff',
    'ancestor_class_win_rate', 'ancestor_distance_win_rate',
    'jockey_horse_distance_win_rate', 'recent_form_score',
    'damsire_class_consistency', 'dam_class_consistency', 'sire_class_consistency',
    'damsire_distance_consistency', 'dam_distance_consistency', 'sire_distance_consistency',
    'damsire_class_win_to_place', 'dam_class_win_to_place', 'sire_class_win_to_place',
    'damsire_distance_win_to_place', 'dam_distance_win_to_place', 'sire_distance_win_to_place',
    'damsire_class_ae_impact', 'dam_class_ae_impact', 'sire_class_ae_impact',
    'damsire_distance_ae_impact', 'dam_distance_ae_impact', 'sire_distance_ae_impact',
    'damsire_class_runner_ratio', 'dam_class_runner_ratio', 'sire_class_runner_ratio',
    'damsire_distance_runner_ratio', 'dam_distance_runner_ratio', 'sire_distance_runner_ratio'
]

# --- Convert numeric columns to appropriate types ---
# Count columns (integers)
count_cols = [col for col in numeric_cols if any(x in col for x in ['runners', 'runs', '1st', '2nd', '3rd', '4th', 'rides'])]
racecards[count_cols] = racecards[count_cols].fillna(0).astype(int)

# Rate columns (rounded floats)
rate_cols = [col for col in numeric_cols if any(x in col for x in ['win_%', 'a/e', '1_pl', 'win_rate', 'place_rate', 'ae_diff', 'form_score', 'consistency', 'win_to_place', 'ae_impact', 'runner_ratio'])]
racecards[rate_cols] = racecards[rate_cols].fillna(0).round(2)

# Remaining numeric columns (default to float, then fill NaN; optionally cast to int if no decimals needed)
for col in numeric_cols:
    if col not in count_cols and col not in rate_cols:
        racecards[col] = pd.to_numeric(racecards[col], errors='coerce').fillna(0)
        if col in ['draw', 'lbs', 'rpr', 'ts', 'last_run', 'form_1', 'form_2', 'form_3', 'form_4', 'type', 'race_class', 'position']:
            racecards[col] = racecards[col].astype(int)

# --- Process categorical columns ---
categorical_cols = ['course', 'trainer', 'owner', 'jockey']
for col in categorical_cols:
    racecards[col] = racecards[col].astype('category')
    racecards[col] = racecards[col].cat.add_categories([0]).fillna(0)

# --- Drop name and specific ID columns ---
columns_to_drop_final = [
    'damsire_class_damsire_name', 'dam_class_dam_name', 'sire_class_sire_name',
    'damsire_distance_damsire_name', 'dam_distance_dam_name', 'sire_distance_sire_name',
    'sire_id', 'damsire_id', 'dam_id'
]
racecards = racecards.drop(columns=columns_to_drop_final, errors='ignore')

# --- Final cleanup and save ---
racecards.fillna(0, inplace=True)
output_file = os.path.join(DATA_DIR, 'merged_test_data.csv')
racecards.to_csv(output_file, index=False)
print(f"Merged file saved to {output_file}")

# --- Print data types for verification ---
print("Numeric Columns:")
print(racecards[numeric_cols].dtypes)
print("\nCategorical Columns:")
print(racecards[categorical_cols].dtypes)