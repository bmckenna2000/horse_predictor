import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define file paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MERGED_INPUT = os.path.join(DATA_DIR, 'merged_retrain.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'preprocessed_retrain_data.csv')
ENCODERS_FILE = os.path.join(DATA_DIR, 'label_encoders_retrain.pkl')
SCALER_FILE = os.path.join(DATA_DIR, 'scaler_retrain.pkl')

# ------------------------------
# 1. Load Merged Retraining Data & Compute Historical Means
# ------------------------------
data = pd.read_csv(MERGED_INPUT, low_memory=False)

# Verify 'date' exists
if 'date' not in data.columns:
    raise ValueError(f"'date' column not found in {MERGED_INPUT}. Please check your input data.")

historical_means = data.groupby('horse_id')[['last_run', 'lbs', 'rpr', 'odds_decimal', 'horse_distance_win_%', 'race_class']].mean().reset_index()
for col in historical_means.columns[1:]:  # Skip 'horse_id'
    historical_means.rename(columns={col: f"{col}_hist_mean"}, inplace=True)
    historical_means[f"{col}_hist_mean"] = historical_means[f"{col}_hist_mean"].replace(0, 1)

# ------------------------------
# 2. Preprocess Basic Features & Outcome Flags
# ------------------------------
data = data.drop_duplicates(subset=['race_id', 'horse_id', 'date', 'off_time'], keep='first')
print(f"Rows after deduplication: {len(data)}")

data['position'] = pd.to_numeric(data['position'], errors='coerce').fillna(0)
data['winner'] = (data['position'] == 1).astype(int)
data['top_2'] = ((data['position'] <= 2) & (data['position'] > 0)).astype(int)
data['top_3'] = ((data['position'] <= 3) & (data['position'] > 0)).astype(int)
data['top_4'] = ((data['position'] <= 4) & (data['position'] > 0)).astype(int)
data['is_winner'] = data['winner']

def assign_target(row):
    fs = row.get('field_size', 0)
    if fs <= 4:
        return row['winner']
    elif 5 <= fs <= 7:
        return row['top_2']
    elif 8 <= fs <= 15:
        return row['top_3']
    else:
        return row['top_4']

data['target'] = data.apply(assign_target, axis=1)

# ------------------------------
# 3. Compute Adjusted Form Metrics
# ------------------------------
def compute_adjusted_form_metrics(row):
    forms = [row.get('form_2', np.nan), row.get('form_3', np.nan), row.get('form_4', np.nan)]
    forms = [f if pd.notna(f) and f > 0 else np.nan for f in forms]
    valid_forms = [f for f in forms if pd.notna(f)]
    std_val = np.nanstd(forms) if len(valid_forms) > 1 else 0  # Avoid warning by checking length
    consistency = 1 / (std_val + 1) if std_val > 0 else 1  # Default to 1 if no variation
    trend = row.get('form_2', 0) - row.get('form_4', 0)
    placements = [1 if pd.notna(f) and f <= 3 else 0 for f in forms]
    placement_ratio = np.nansum(placements) / len(forms)
    recent_improvement = row.get('form_2', 0) - row.get('form_4', 0)
    momentum = ((row.get('form_2', 0) + row.get('form_3', 0)) / 2) - row.get('form_4', 0)
    return pd.Series({
        'form_consistency': consistency,
        'form_trend': trend,
        'form_placement_ratio': placement_ratio,
        'form_recent_improvement': recent_improvement,
        'form_momentum': momentum
    })

data[['form_consistency', 'form_trend', 'form_placement_ratio', 'form_recent_improvement', 'form_momentum']] = data.apply(compute_adjusted_form_metrics, axis=1)

# ------------------------------
# 4. Compute Placement Ratios and Additional Derived Features
# ------------------------------
data['horse_placement_ratio'] = (data[['horse_distance_1st', 'horse_distance_2nd', 'horse_distance_3rd']].sum(axis=1) / data['horse_distance_runs'].replace(0, 1)).astype(float)
data['jockey_course_placement_ratio'] = (data[['jockey_course_1st', 'jockey_course_2nd', 'jockey_course_3rd']].sum(axis=1) / data['jockey_course_rides'].replace(0, 1)).astype(float)
data['jockey_distance_placement_ratio'] = (data[['jockey_distance_1st', 'jockey_distance_2nd', 'jockey_distance_3rd']].sum(axis=1) / data['jockey_distance_rides'].replace(0, 1)).astype(float)
data['jockey_owner_placement_ratio'] = (data[['jockey_owner_1st', 'jockey_owner_2nd', 'jockey_owner_3rd']].sum(axis=1) / data['jockey_owner_rides'].replace(0, 1)).astype(float)
data['jockey_trainer_placement_ratio'] = (data[['jockey_trainer_1st', 'jockey_trainer_2nd', 'jockey_trainer_3rd']].sum(axis=1) / data['jockey_trainer_rides'].replace(0, 1)).astype(float)

placement_ratio_cols = ['horse_placement_ratio', 'jockey_course_placement_ratio', 'jockey_distance_placement_ratio', 'jockey_owner_placement_ratio', 'jockey_trainer_placement_ratio']
jockey_win_cols = ['jockey_course_win_%', 'jockey_distance_win_%', 'jockey_owner_win_%', 'jockey_trainer_win_%']

data['jockey_course_win_%'] = (data['jockey_course_1st'] / data['jockey_course_rides'].replace(0, 1)).astype(float)
data['jockey_distance_win_%'] = (data['jockey_distance_1st'] / data['jockey_distance_rides'].replace(0, 1)).astype(float)
data['jockey_owner_win_%'] = (data['jockey_owner_1st'] / data['jockey_owner_rides'].replace(0, 1)).astype(float)
data['jockey_trainer_win_%'] = (data['jockey_trainer_1st'] / data['jockey_trainer_rides'].replace(0, 1)).astype(float)

data['jockey_win_%'] = data[jockey_win_cols].mean(axis=1, skipna=True)
data['final_placement_%'] = data[placement_ratio_cols].mean(axis=1, skipna=True)
data['consistency_score'] = (data[['horse_distance_1st', 'horse_distance_2nd']].sum(axis=1) / data['horse_distance_runs'].replace(0, 1)).astype(float)

# ------------------------------
# 5. Merge Historical Means
# ------------------------------
data = data.merge(historical_means, on='horse_id', how='left')
for col in [f"{col}_hist_mean" for col in ['last_run', 'lbs', 'rpr', 'odds_decimal', 'horse_distance_win_%', 'race_class']]:
    data[col] = data[col].fillna(1)

# ------------------------------
# 6. Compute Derived Features Using Historical Means
# ------------------------------
data['last_run_relative'] = (data['last_run'] / data['last_run_hist_mean']).fillna(0)
data['weight_relative'] = (data['lbs'] / data['lbs_hist_mean']).fillna(0)
data['relative_weight_impact'] = ((data['weight_relative'] - 1.0) * data['horse_placement_ratio']).fillna(0)
data['rpr_relative'] = (data['rpr'] / data['rpr_hist_mean']).fillna(0)
data['rpr_place_interaction'] = (data['rpr_relative'] * data['final_placement_%']).fillna(0)
data['race_rpr_variance'] = data.groupby('horse_id')['rpr'].shift(1).rolling(3, min_periods=1).var().fillna(0)
data['odds_deviation'] = (data['odds_decimal'] / data['odds_decimal_hist_mean']).fillna(0)
data['field_strength'] = pd.to_numeric(data['horse_distance_win_%_hist_mean'], errors='coerce').fillna(0)
data['trainer_jockey_synergy'] = data['jockey_course_rides'] * (data['trainer_14_days_percent'] / 100)
data['race_pace_interaction'] = data['rpr'] * (data['distance_f'] / data['field_size'].replace(0, 1))
data['draw_advantage'] = data['draw'] / data['field_size'].replace(0, 1)
data['has_draw'] = (data['draw'] > 0).astype(int)
data['top_competitor_rpr'] = pd.to_numeric(data['rpr_hist_mean'], errors='coerce').fillna(0)
data['top_competitor_margin'] = (data['rpr'] - data['top_competitor_rpr']).fillna(0)

# ------------------------------
# 7. Compute Horse Age Distance Feature
# ------------------------------
data['distance_f'] = pd.to_numeric(data['distance_f'], errors='coerce').fillna(0)
data['age'] = pd.to_numeric(data['age'], errors='coerce').fillna(0)
data['distance_f_int'] = data['distance_f'].round(0).astype(int)
data['horse_age_distance'] = data['age'] * data['distance_f_int']

# ------------------------------
# 8. Ranking and Field-Relative Features
# ------------------------------
data['horse_rpr_rank_in_race'] = data.groupby('race_id')['rpr'].rank(ascending=False, method='min').fillna(0)
data['horse_odds_rank_in_race'] = data.groupby('race_id')['odds_decimal'].rank(ascending=True, method='min').fillna(0)
data['horse_rpr_relative_to_field_mean'] = data.groupby('race_id')['rpr'].transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)).fillna(0)
data['pace_pressure_index'] = data.groupby('race_id')['ts'].transform(lambda x: (x >= x.quantile(0.75)).sum()).fillna(0)

# ------------------------------
# 9. Field and Class Related Features
# ------------------------------
data['avg_historical_class'] = data['race_class_hist_mean']
data['race_class_drop_impact'] = ((data['avg_historical_class'] - data['race_class']) * data['horse_distance_win_%']).fillna(0)

# ------------------------------
# 10. Suitability Features
# ------------------------------
data['jockey_distance_suitability'] = (data['jockey_distance_win_%'] / data.groupby('jockey_id')['jockey_distance_win_%'].transform('mean')).fillna(1)
data['jockey_type_suitability'] = (data['jockey_win_%'] / data.groupby(['jockey_id', 'type'])['jockey_win_%'].transform('mean')).fillna(1)
data['horse_distance_suitability'] = (data['horse_distance_win_%'] / data.groupby('horse_id')['horse_distance_win_%'].transform('mean')).fillna(1)
data['horse_type_suitability'] = (data['horse_distance_win_%'] / data.groupby(['horse_id', 'type'])['horse_distance_win_%'].transform('mean')).fillna(1)
data['jockey_course_familiarity_boost'] = data['jockey_course_rides'].apply(lambda x: x * 1.1 if x > 50 else x)
data['track_course_bias'] = np.abs(data['draw'] - data.groupby(['course_id', 'distance_f', 'going'])['draw'].transform('mean'))

# Lag features and moving averages
data['rpr_lag_1'] = data.groupby('horse_id')['rpr'].shift(1).fillna(0)
data['rpr_lag_2'] = data.groupby('horse_id')['rpr'].shift(2).fillna(0)
data['rpr_lag_3'] = data.groupby('horse_id')['rpr'].shift(3).fillna(0)
data['horse_rpr_trend'] = (data['rpr'] - data['rpr_lag_3']) / 3
data.drop(columns=['rpr_lag_1', 'rpr_lag_2', 'rpr_lag_3'], inplace=True)
data['rpr_ema'] = data.sort_values('date').groupby('horse_id')['rpr'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
data['jockey_horse_interaction'] = data['jockey_win_%'] * data['horse_distance_win_%']
data['market_model_disagreement'] = data['horse_odds_rank_in_race'] - data['horse_rpr_rank_in_race']
data['race_par_score'] = data.groupby(['course', 'distance_f', 'race_class'])['rpr'].transform('mean')
data['max_rpr_in_race'] = data.groupby('race_id')['rpr'].transform('max')
data['rpr_diff_to_top'] = data['max_rpr_in_race'] - data['rpr']
data['rpr_rank'] = data.groupby('race_id')['rpr'].rank(method='min')
data['relative_rpr'] = data.groupby('race_id')['rpr'].transform(lambda x: x - x.mean())

data['avg_class_last_5'] = data.sort_values('date').groupby('horse_id')['race_class'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
data['class_diff'] = (data['race_class'] - data['avg_class_last_5']).fillna(0)
data.drop(columns=['avg_class_last_5'], inplace=True)
data['class_diff'] = pd.to_numeric(data['class_diff'], errors='coerce').fillna(0)

winning_distances = data[data['horse_distance_1st'] > 0].groupby('horse_id')['distance_f'].mean().reset_index().rename(columns={'distance_f': 'avg_winning_distance'})
data = data.merge(winning_distances, on='horse_id', how='left')
data['avg_winning_distance'] = data['avg_winning_distance'].fillna(data['distance_f'])
data['distance_fit'] = data['horse_distance_win_%'] * (1 - abs(data['distance_f'] - data['avg_winning_distance']) / data['distance_f'].replace(0, 1))
data.drop(columns=['avg_winning_distance'], inplace=True)
data['distance_fit'] = pd.to_numeric(data['distance_fit'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

going_surface_wins = data.groupby(['horse_id', 'going', 'surface'])['horse_distance_1st'].mean().reset_index().rename(columns={'horse_distance_1st': 'going_surface_win_%'})
data = data.merge(going_surface_wins, on=['horse_id', 'going', 'surface'], how='left')
data['going_surface_win_%'] = data['going_surface_win_%'].fillna(data['horse_distance_win_%'])
data['going_surface_win_%'] = pd.to_numeric(data['going_surface_win_%'], errors='coerce').fillna(0)
data['course_win_%'] = pd.to_numeric(data['jockey_course_1st'] / data['jockey_course_total_rides'].replace(0, 1), errors='coerce').fillna(0)
data['rpr_lag_ema'] = data.sort_values('date').groupby('horse_id')['rpr'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
data['market_model_disagreement'] = data['horse_odds_rank_in_race'] - data['horse_rpr_rank_in_race']


# --- Add New Feature: Relative Performance Index ---
# Compute combined rpr + ts for each horse
data['rpr_ts_combined'] = data['rpr'] + data['ts']
# Calculate the race-specific average of rpr + ts
race_avg_rpr_ts = data.groupby('race_id')['rpr_ts_combined'].mean()
# Map the race average back to the DataFrame
data['avg_rpr_ts'] = data['race_id'].map(race_avg_rpr_ts)
# Compute Relative Performance Index
data['rpi'] = data['rpr_ts_combined'] / data['avg_rpr_ts'].replace(0, 1)
# Drop temporary columns
data.drop(columns=['rpr_ts_combined', 'avg_rpr_ts'], inplace=True)
# Ensure rpi is numeric and handle NaNs or infinities
data['rpi'] = pd.to_numeric(data['rpi'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

avg_lbs = data['lbs'].mean()
# Compute Weight-Adjusted RPR
data['rpr_adjusted'] = data['rpr'] / (data['lbs'] / avg_lbs)
# Ensure rpr_adjusted is numeric and handle NaNs or infinities
data['rpr_adjusted'] = pd.to_numeric(data['rpr_adjusted'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
# ------------------------------
# 11. Encode Categorical Columns and Normalize Numeric Data
# ------------------------------
categorical_cols = ['course', 'horse']
label_encoders = {}
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} unique values")

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
    'form_consistency', 'form_trend', 'form_placement_ratio', 'form_recent_improvement', 
    'form_momentum', 'horse_placement_ratio', 'jockey_course_placement_ratio', 
    'jockey_distance_placement_ratio', 'jockey_owner_placement_ratio', 
    'jockey_trainer_placement_ratio', 'jockey_win_%', 'final_placement_%', 'consistency_score',
    'last_run_hist_mean', 'lbs_hist_mean', 'rpr_hist_mean', 'odds_decimal_hist_mean', 
    'horse_distance_win_%_hist_mean', 'race_class_hist_mean', 'last_run_relative', 
    'weight_relative', 'relative_weight_impact', 'rpr_relative', 'rpr_place_interaction', 
    'race_rpr_variance', 'odds_deviation', 'field_strength', 'trainer_jockey_synergy', 
    'race_pace_interaction', 'draw_advantage', 'has_draw', 'horse_age_distance', 
    'horse_rpr_rank_in_race', 'horse_odds_rank_in_race', 'horse_rpr_relative_to_field_mean', 
    'pace_pressure_index', 'jockey_distance_suitability', 'jockey_type_suitability', 
    'horse_distance_suitability', 'horse_type_suitability', 'race_class_drop_impact', 
    'jockey_course_familiarity_boost', 'track_course_bias', 'horse_rpr_trend', 'rpr_ema', 
    'jockey_horse_interaction', 'market_model_disagreement', 'race_par_score', 
    'max_rpr_in_race', 'rpr_diff_to_top', 'rpr_rank', 'top_competitor_rpr', 
    'top_competitor_margin', 'race_class', 'type', 'going', 'surface', 'jumps', 'age', 
    'trainer_rtf', 'relative_rpr', 'class_diff', 'distance_fit', 'going_surface_win_%', 
    'course_win_%', 'rpi', 'rpr_adjusted',
    # New derived features from merge script
    'damsire_class_win_rate', 'dam_class_win_rate', 'sire_class_win_rate',
    'damsire_distance_win_rate', 'dam_distance_win_rate', 'sire_distance_win_rate',
    'damsire_class_place_rate', 'dam_class_place_rate', 'sire_class_place_rate',
    'damsire_distance_place_rate', 'dam_distance_place_rate', 'sire_distance_place_rate',
    'damsire_class_ae_diff', 'dam_class_ae_diff', 'sire_class_ae_diff',
    'damsire_distance_ae_diff', 'dam_distance_ae_diff', 'sire_distance_ae_diff',
    'ancestor_class_win_rate', 'ancestor_distance_win_rate', 'jockey_horse_distance_win_rate',
    'recent_form_score', 'damsire_class_consistency', 'dam_class_consistency', 
    'sire_class_consistency', 'damsire_distance_consistency', 'dam_distance_consistency', 
    'sire_distance_consistency', 'damsire_class_win_to_place', 'dam_class_win_to_place', 
    'sire_class_win_to_place', 'damsire_distance_win_to_place', 'dam_distance_win_to_place', 
    'sire_distance_win_to_place', 'damsire_class_ae_impact', 'dam_class_ae_impact', 
    'sire_class_ae_impact', 'damsire_distance_ae_impact', 'dam_distance_ae_impact', 
    'sire_distance_ae_impact', 'damsire_class_runner_ratio', 'dam_class_runner_ratio', 
    'sire_class_runner_ratio', 'damsire_distance_runner_ratio', 'dam_distance_runner_ratio', 
    'sire_distance_runner_ratio',
    # Raw columns from merge script
    'damsire_distance_total_runners', 'damsire_distance_runners', 'damsire_distance_1st',
    'damsire_distance_2nd', 'damsire_distance_3rd', 'damsire_distance_4th', 'damsire_distance_a/e',
    'damsire_distance_win_%', 'damsire_distance_1_pl', 'dam_distance_total_runners',
    'dam_distance_runners', 'dam_distance_1st', 'dam_distance_2nd', 'dam_distance_3rd',
    'dam_distance_4th', 'dam_distance_a/e', 'dam_distance_win_%', 'dam_distance_1_pl',
    'sire_distance_total_runners', 'sire_distance_runners', 'sire_distance_1st',
    'sire_distance_2nd', 'sire_distance_3rd', 'sire_distance_4th', 'sire_distance_a/e',
    'sire_distance_win_%', 'sire_distance_1_pl', 'damsire_class_total_runners', 
    'damsire_class_runners', 'damsire_class_1st', 'damsire_class_2nd', 'damsire_class_3rd',
    'damsire_class_4th', 'damsire_class_a/e', 'damsire_class_win_%', 'damsire_class_1_pl', 
    'dam_class_total_runners', 'dam_class_runners', 'dam_class_1st', 'dam_class_2nd', 
    'dam_class_3rd', 'dam_class_4th', 'dam_class_a/e', 'dam_class_win_%', 'dam_class_1_pl',
    'sire_class_total_runners', 'sire_class_runners', 'sire_class_1st', 'sire_class_2nd', 
    'sire_class_3rd', 'sire_class_4th', 'sire_class_a/e', 'sire_class_win_%', 'sire_class_1_pl',
    # Added columns from header
    'trainer_14_days_runs', 'trainer_14_days_wins'
]

scaler = StandardScaler()
features_to_scale = [col for col in numeric_cols if col in data.columns]
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# ------------------------------
# 12. Final Cleanup: Sort and Drop Unneeded Columns
# ------------------------------
# Sort before dropping 'date'
# ------------------------------
# 12. Final Cleanup: Sort and Drop Unneeded Columns
# ------------------------------
# Sort before dropping unnecessary columns
data = data.sort_values(['date', 'race_id'])

# Define id and outcome columns, but keep 'target' and necessary id_cols
id_cols_to_keep = ['race_id', 'date', 'off_time', 'field_size']
outcome_cols_to_drop = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner']  # Exclude 'target'
cols_to_drop = outcome_cols_to_drop  # Keep 'target' and id_cols
data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)

# Ensure 'rpi' and 'rpr_adjusted' are present (add placeholders if missing)
if 'rpi' not in data.columns:
    data['rpi'] = 0  # Placeholder; replace with actual computation if available
    print("Warning: 'rpi' not found in data. Added as placeholder (0).")
if 'rpr_adjusted' not in data.columns:
    data['rpr_adjusted'] = data['rpr']  # Placeholder; assume it’s derived from 'rpr' if not computed
    print("Warning: 'rpr_adjusted' not found in data. Set to 'rpr' as placeholder.")

# ------------------------------
# 13. Save Preprocessed Data and Artifacts
# ------------------------------
data.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
with open(ENCODERS_FILE, 'wb') as f:
    pickle.dump(label_encoders, f)
with open(SCALER_FILE, 'wb') as f:
    pickle.dump(scaler, f)

saved_data = pd.read_csv(OUTPUT_FILE, low_memory=False)
print(f"✅ Saved preprocessed retrain data to {OUTPUT_FILE}")
print(f"✅ Final row count after saving: {len(saved_data)}")
print(f"Columns in saved data: {list(saved_data.columns)}")