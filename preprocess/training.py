import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define file paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MERGED_FILE = os.path.join(DATA_DIR, 'preprocessed_training_data.csv')
ENCODERS_FILE = os.path.join(DATA_DIR, 'label_encoders.pkl')
SCALER_FILE = os.path.join(DATA_DIR, 'scaler.pkl')

# ------------------------------
# 1. Load Historical Data and Compute Aggregated Historical Means
# ------------------------------
historical_data = pd.read_csv(os.path.join(DATA_DIR, 'merged_test_data.csv'))
# Define the columns for which to compute historical means (including race_class)
hist_cols = ['last_run', 'lbs', 'rpr', 'odds_decimal', 'horse_distance_win_%', 'race_class']
historical_means = historical_data.groupby('horse_id')[hist_cols].mean().reset_index()
# Rename columns to include a _hist_mean suffix and replace zeros with 1
for col in hist_cols:
    historical_means.rename(columns={col: f"{col}_hist_mean"}, inplace=True)
    historical_means[f"{col}_hist_mean"] = historical_means[f"{col}_hist_mean"].replace(0, 1)

# ------------------------------
# 2. Load Prediction Data and Preprocess Basic Features
# ------------------------------
test_data = pd.read_csv(os.path.join(DATA_DIR, 'merged_test_data.csv'))
test_data = test_data.drop_duplicates(subset=['race_id', 'horse_id', 'date', 'off_time'], keep='first')
print(f"Rows after initial deduplication: {len(test_data)}")

# Ensure 'position' is numeric and create outcome flags
test_data['position'] = pd.to_numeric(test_data['position'], errors='coerce').fillna(0)
test_data['winner'] = (test_data['position'] == 1).astype(int)
test_data['top_2'] = ((test_data['position'] <= 2) & (test_data['position'] > 0)).astype(int)
test_data['top_3'] = ((test_data['position'] <= 3) & (test_data['position'] > 0)).astype(int)
test_data['top_4'] = ((test_data['position'] <= 4) & (test_data['position'] > 0)).astype(int)
test_data['is_winner'] = (test_data['position'] == 1).astype(int)

def assign_target(row):
    fs = row['field_size']
    if fs <= 4:
        return row['winner']
    elif 5 <= fs <= 7:
        return row['top_2']
    elif 8 <= fs <= 15:
        return row['top_3']
    else:
        return row['top_4']

test_data['target'] = test_data.apply(assign_target, axis=1)

# ------------------------------
# 3. Compute Adjusted Form Metrics (using form_2, form_3, form_4 as historical)
# ------------------------------
def compute_adjusted_form_metrics(row):
    forms = [row['form_2'], row['form_3'], row['form_4']]
    forms = [f if f > 0 else np.nan for f in forms]  # treat 0 as missing if needed
    std_val = np.nanstd(forms)
    consistency = 1 / (std_val + 1)
    trend = row['form_2'] - row['form_4']
    placements = [1 if (f is not np.nan and f <= 3) else 0 for f in forms]
    placement_ratio = np.nansum(placements) / len(forms)
    recent_improvement = row['form_2'] - row['form_4']
    momentum = (row['form_2'] + row['form_3']) / 2 - row['form_4']
    return pd.Series({
        'form_consistency': consistency,
        'form_trend': trend,
        'form_placement_ratio': placement_ratio,
        'form_recent_improvement': recent_improvement,
        'form_momentum': momentum
    })

test_data[['form_consistency', 'form_trend', 'form_placement_ratio', 
           'form_recent_improvement', 'form_momentum']] = test_data.apply(compute_adjusted_form_metrics, axis=1)

# ------------------------------
# 4. Compute Placement Ratios and Other Derived Features
# ------------------------------
test_data['horse_placement_ratio'] = pd.to_numeric(
    (test_data[['horse_distance_1st', 'horse_distance_2nd', 'horse_distance_3rd']].sum(axis=1)
     / test_data['horse_distance_runs'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_course_placement_ratio'] = pd.to_numeric(
    (test_data[['jockey_course_1st', 'jockey_course_2nd', 'jockey_course_3rd']].sum(axis=1)
     / test_data['jockey_course_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_distance_placement_ratio'] = pd.to_numeric(
    (test_data[['jockey_distance_1st', 'jockey_distance_2nd', 'jockey_distance_3rd']].sum(axis=1)
     / test_data['jockey_distance_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_owner_placement_ratio'] = pd.to_numeric(
    (test_data[['jockey_owner_1st', 'jockey_owner_2nd', 'jockey_owner_3rd']].sum(axis=1)
     / test_data['jockey_owner_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_trainer_placement_ratio'] = pd.to_numeric(
    (test_data[['jockey_trainer_1st', 'jockey_trainer_2nd', 'jockey_trainer_3rd']].sum(axis=1)
     / test_data['jockey_trainer_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

placement_ratio_cols = [
    'horse_placement_ratio', 
    'jockey_course_placement_ratio', 
    'jockey_distance_placement_ratio', 
    'jockey_owner_placement_ratio', 
    'jockey_trainer_placement_ratio'
]

jockey_win_cols = [
    'jockey_course_win_%', 
    'jockey_distance_win_%', 
    'jockey_owner_win_%', 
    'jockey_trainer_win_%'
]

test_data['jockey_course_win_%'] = pd.to_numeric(
    (test_data['jockey_course_1st'] / test_data['jockey_course_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_distance_win_%'] = pd.to_numeric(
    (test_data['jockey_distance_1st'] / test_data['jockey_distance_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_owner_win_%'] = pd.to_numeric(
    (test_data['jockey_owner_1st'] / test_data['jockey_owner_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_trainer_win_%'] = pd.to_numeric(
    (test_data['jockey_trainer_1st'] / test_data['jockey_trainer_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['jockey_win_%'] = pd.to_numeric(
    test_data[jockey_win_cols].mean(axis=1, skipna=True),
    errors='coerce'
).fillna(0)

test_data['final_placement_%'] = pd.to_numeric(
    test_data[placement_ratio_cols].mean(axis=1, skipna=True),
    errors='coerce'
).fillna(0)

test_data['consistency_score'] = pd.to_numeric(
    (test_data[['horse_distance_1st', 'horse_distance_2nd']].sum(axis=1)
     / test_data['horse_distance_runs'].replace(0, 1)),
    errors='coerce'
).fillna(0)

# ------------------------------
# 5. Merge Historical Means (via horse_id)
# ------------------------------
test_data = test_data.merge(historical_means, on='horse_id', how='left')
for col in [f"{col}_hist_mean" for col in hist_cols]:
    test_data[col].fillna(1, inplace=True)

# ------------------------------
# 6. Compute Derived Features Using Historical Means
# ------------------------------
test_data['last_run_relative'] = pd.to_numeric(
    test_data['last_run'] / test_data['last_run_hist_mean'], errors='coerce'
).fillna(0)

test_data['weight_relative'] = pd.to_numeric(
    test_data['lbs'] / test_data['lbs_hist_mean'], errors='coerce'
).fillna(0)

test_data['relative_weight_impact'] = pd.to_numeric(
    (test_data['weight_relative'] - 1.0) * test_data['horse_placement_ratio'],
    errors='coerce'
).fillna(0)

test_data['rpr_relative'] = pd.to_numeric(
    test_data['rpr'] / test_data['rpr_hist_mean'], errors='coerce'
).fillna(0)

test_data['rpr_place_interaction'] = pd.to_numeric(
    test_data['rpr_relative'] * test_data['final_placement_%'], errors='coerce'
).fillna(0)

test_data['race_rpr_variance'] = pd.to_numeric(
    test_data.groupby('horse_id')['rpr'].shift(1).rolling(3, min_periods=1).var(),
    errors='coerce'
).fillna(0)

test_data['odds_deviation'] = pd.to_numeric(
    test_data['odds_decimal'] / test_data['odds_decimal_hist_mean'], errors='coerce'
).fillna(0)

test_data['field_strength'] = pd.to_numeric(
    test_data['horse_distance_win_%_hist_mean'], errors='coerce'
).fillna(0)

test_data['trainer_jockey_synergy'] = pd.to_numeric(
    test_data['jockey_course_rides'] * (test_data['trainer_14_days_percent'] / 100),
    errors='coerce'
).fillna(0)

test_data['race_pace_interaction'] = pd.to_numeric(
    test_data['rpr'] * (test_data['distance_f'] / test_data['field_size'].replace(0, 1)),
    errors='coerce'
).fillna(0)

test_data['draw_advantage'] = pd.to_numeric(
    test_data['draw'] / test_data['field_size'].replace(0, 1),
    errors='coerce'
).fillna(0)

test_data['has_draw'] = pd.to_numeric(
    (test_data['draw'] > 0).astype(int),
    errors='coerce'
).fillna(0)

test_data['top_competitor_rpr'] = pd.to_numeric(test_data['rpr_hist_mean'], errors='coerce').fillna(0)
test_data['top_competitor_margin'] = pd.to_numeric(test_data['rpr'] - test_data['top_competitor_rpr'], errors='coerce').fillna(0)

# ------------------------------
# 7. Compute Horse Age Distance Feature
# ------------------------------
test_data['distance_f'] = pd.to_numeric(test_data['distance_f'], errors='coerce').fillna(0)
test_data['age'] = pd.to_numeric(test_data['age'], errors='coerce').fillna(0)
test_data['distance_f_int'] = test_data['distance_f'].round(0).astype(int)
test_data['horse_age_distance'] = test_data['age'] * test_data['distance_f_int']

# ------------------------------
# 8. Compute Ranking and Field-Relative Features
# ------------------------------
test_data['horse_rpr_rank_in_race'] = pd.to_numeric(
    test_data.groupby('race_id')['rpr'].rank(ascending=False, method='min'),
    errors='coerce'
).fillna(0)

test_data['horse_odds_rank_in_race'] = pd.to_numeric(
    test_data.groupby('race_id')['odds_decimal'].rank(ascending=True, method='min'),
    errors='coerce'
).fillna(0)

test_data['horse_rpr_relative_to_field_mean'] = pd.to_numeric(
    test_data.groupby('race_id')['rpr'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    ),
    errors='coerce'
).fillna(0)

test_data['pace_pressure_index'] = pd.to_numeric(
    test_data.groupby('race_id')['ts'].transform(lambda x: (x >= x.quantile(0.75)).sum()),
    errors='coerce'
).fillna(0)

# ------------------------------
# 9. Set avg_historical_class from Historical Data
# ------------------------------
test_data['avg_historical_class'] = test_data['race_class_hist_mean']
test_data['race_class_drop_impact'] = pd.to_numeric(
    (test_data['avg_historical_class'] - test_data['race_class']) * test_data['horse_distance_win_%'],
    errors='coerce'
).fillna(0)

# ------------------------------
# 10. Compute Suitability Features
# ------------------------------
test_data['jockey_distance_suitability'] = test_data['jockey_distance_win_%'] / \
    test_data.groupby('jockey_id')['jockey_distance_win_%'].transform('mean')
test_data['jockey_distance_suitability'] = test_data['jockey_distance_suitability'].fillna(1)

test_data['jockey_type_suitability'] = test_data['jockey_win_%'] / \
    test_data.groupby(['jockey_id', 'type'])['jockey_win_%'].transform('mean')
test_data['jockey_type_suitability'] = test_data['jockey_type_suitability'].fillna(1)

test_data['horse_distance_suitability'] = test_data['horse_distance_win_%'] / \
    test_data.groupby('horse_id')['horse_distance_win_%'].transform('mean')
test_data['horse_distance_suitability'] = test_data['horse_distance_suitability'].fillna(1)

test_data['horse_type_suitability'] = test_data['horse_distance_win_%'] / \
    test_data.groupby(['horse_id', 'type'])['horse_distance_win_%'].transform('mean')
test_data['horse_type_suitability'] = test_data['horse_type_suitability'].fillna(1)

test_data['jockey_course_familiarity_boost'] = pd.to_numeric(test_data['jockey_course_rides'].apply(lambda x: x * 1.1 if x > 50 else x), errors='coerce').fillna(0)
test_data['track_course_bias'] = pd.to_numeric(np.abs(test_data['draw'] - test_data.groupby(['course_id', 'distance_f', 'going'])['draw'].transform('mean')), errors='coerce').fillna(0)
# Final check: print sample outputs

test_data['rpr_lag_1'] = pd.to_numeric(test_data.groupby('horse_id')['rpr'].shift(1), errors='coerce').fillna(0)
test_data['rpr_lag_2'] = pd.to_numeric(test_data.groupby('horse_id')['rpr'].shift(2), errors='coerce').fillna(0)
test_data['rpr_lag_3'] = pd.to_numeric(test_data.groupby('horse_id')['rpr'].shift(3), errors='coerce').fillna(0)
test_data['horse_rpr_trend'] = pd.to_numeric((test_data['rpr'] - test_data['rpr_lag_3']) / 3, errors='coerce').fillna(0)

test_data = test_data.drop(columns=['rpr_lag_1', 'rpr_lag_2', 'rpr_lag_3', 'avg_historical_class'], errors='ignore')

test_data['rpr_ema'] = test_data.sort_values('date').groupby('horse_id')['rpr']\
    .transform(lambda x: x.ewm(span=3, adjust=False).mean())
    
test_data['jockey_horse_interaction'] = test_data['jockey_win_%'] * test_data['horse_distance_win_%']

test_data['market_model_disagreement'] = test_data['horse_odds_rank_in_race'] - test_data['horse_rpr_rank_in_race']
test_data['race_par_score'] = test_data.groupby(['course', 'distance_f', 'race_class'])['rpr'].transform('mean')
test_data['max_rpr_in_race'] = test_data.groupby('race_id')['rpr'].transform('max')
test_data['rpr_diff_to_top'] = test_data['max_rpr_in_race'] - test_data['rpr']
test_data['rpr_rank'] = test_data.groupby('race_id')['rpr'].rank(method='min')

# --- Add New Feature: Relative Field Strength ---
# Compute the average RPR per race_id and calculate relative_rpr
test_data['relative_rpr'] = test_data.groupby('race_id')['rpr'].transform(lambda x: x - x.mean())

test_data['avg_class_last_5'] = test_data.sort_values('date').groupby('horse_id')['race_class']\
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
# Calculate Class Differential
test_data['class_diff'] = test_data['race_class'] - test_data['avg_class_last_5']
# Drop the temporary avg_class_last_5 column
test_data.drop(columns=['avg_class_last_5'], inplace=True)
# Ensure class_diff is numeric and handle NaNs
test_data['class_diff'] = pd.to_numeric(test_data['class_diff'], errors='coerce').fillna(0)

winning_distances = test_data[test_data['horse_distance_1st'] > 0].groupby('horse_id')['distance_f']\
    .mean().reset_index().rename(columns={'distance_f': 'avg_winning_distance'})
# Merge the average winning distance back into the main DataFrame
test_data = test_data.merge(winning_distances, on='horse_id', how='left')
# Fill NaN avg_winning_distance with the current distance_f (neutral assumption for horses with no wins)
test_data['avg_winning_distance'] = test_data['avg_winning_distance'].fillna(test_data['distance_f'])
# Compute Distance Suitability
test_data['distance_fit'] = test_data['horse_distance_win_%'] * \
    (1 - abs(test_data['distance_f'] - test_data['avg_winning_distance']) / test_data['distance_f'].replace(0, 1))
# Drop the temporary avg_winning_distance column
test_data.drop(columns=['avg_winning_distance'], inplace=True)
# Ensure distance_fit is numeric and handle NaNs or infinities
test_data['distance_fit'] = pd.to_numeric(test_data['distance_fit'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

going_surface_wins = test_data.groupby(['horse_id', 'going', 'surface'])['horse_distance_1st']\
    .mean().reset_index().rename(columns={'horse_distance_1st': 'going_surface_win_%'})
# Merge this back into the main DataFrame based on current going and surface
test_data = test_data.merge(going_surface_wins, on=['horse_id', 'going', 'surface'], how='left')
# Fill NaNs with the horse's overall horse_distance_win_% as a fallback
test_data['going_surface_win_%'] = test_data['going_surface_win_%'].fillna(test_data['horse_distance_win_%'])
# Ensure going_surface_win_% is numeric and handle any remaining NaNs
test_data['going_surface_win_%'] = pd.to_numeric(test_data['going_surface_win_%'], errors='coerce').fillna(0)

test_data['course_win_%'] = pd.to_numeric(
    test_data['jockey_course_1st'] / test_data['jockey_course_total_rides'].replace(0, 1),
    errors='coerce'
).fillna(0)

# --- Add New Feature: Relative Performance Index ---
# Compute combined rpr + ts for each horse
test_data['rpr_ts_combined'] = test_data['rpr'] + test_data['ts']
# Calculate the race-specific average of rpr + ts
race_avg_rpr_ts = test_data.groupby('race_id')['rpr_ts_combined'].mean()
# Map the race average back to the DataFrame
test_data['avg_rpr_ts'] = test_data['race_id'].map(race_avg_rpr_ts)
# Compute Relative Performance Index
test_data['rpi'] = test_data['rpr_ts_combined'] / test_data['avg_rpr_ts'].replace(0, 1)
# Drop temporary columns
test_data.drop(columns=['rpr_ts_combined', 'avg_rpr_ts'], inplace=True)
# Ensure rpi is numeric and handle NaNs or infinities
test_data['rpi'] = pd.to_numeric(test_data['rpi'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

avg_lbs = test_data['lbs'].mean()
# Compute Weight-Adjusted RPR
test_data['rpr_adjusted'] = test_data['rpr'] / (test_data['lbs'] / avg_lbs)
# Ensure rpr_adjusted is numeric and handle NaNs or infinities
test_data['rpr_adjusted'] = pd.to_numeric(test_data['rpr_adjusted'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
# ------------------------------
# 11. Encode Categorical Columns Using OneHotEncoder and Save the Encoder
# ------------------------------
categorical_cols = ['course', 'horse']

# ------------------------------
# 12. Normalize Numeric Columns and Prepare Final Data for Training
# ------------------------------
id_cols = ['race_id', 'date', 'off_time', 'field_size']
outcome_cols = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner', 'target']

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

label_encoders = {}  # Dictionary to hold fitted encoders
for col in categorical_cols:
    if col in test_data.columns:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} unique values")
        
for col in numeric_cols:
    if col in test_data.columns:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').fillna(0)
        test_data[col] = test_data[col].replace([np.inf, -np.inf], 0)
        
scaler = StandardScaler()
features_to_scale = [col for col in numeric_cols if col in test_data.columns]
test_data[features_to_scale] = scaler.fit_transform(test_data[features_to_scale])

columns_to_drop = [
    'course_id', 'horse_id', 'trainer', 'trainer_id', 'owner', 'owner_id', 
    'jockey', 'jockey_id', 'jockey_course_total_rides', 'jockey_course_1st', 'jockey_course_2nd', 
    'jockey_course_3rd', 'jockey_course_4th', 'jockey_owner_total_rides', 'jockey_owner_1st', 
    'jockey_owner_2nd', 'jockey_owner_3rd', 'jockey_owner_4th', 'jockey_trainer_total_rides', 
    'jockey_trainer_1st', 'jockey_trainer_2nd', 'jockey_trainer_3rd', 'jockey_trainer_4th', 
    'jockey_distance_total_rides', 'jockey_distance_1st', 'jockey_distance_2nd', 
    'jockey_distance_3rd', 'jockey_distance_4th', 'horse_distance_1st', 'horse_distance_2nd', 
    'horse_distance_3rd', 'horse_distance_4th', 'form_1', 'form_2', 'form_3', 'form_4', 
    'distance_f_int'
]

test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns], inplace=True)

test_data = test_data.sort_values(['date', 'race_id'])
test_data.to_csv(MERGED_FILE, index=False, encoding='utf-8')

# Save the label encoders and scaler for future use
with open(ENCODERS_FILE, 'wb') as f:
    pickle.dump(label_encoders, f)
with open(SCALER_FILE, 'wb') as f:
    pickle.dump(scaler, f)

# Verify saved file
saved_data = pd.read_csv(MERGED_FILE, low_memory=False)
print(f"✅ Saved merged data to {MERGED_FILE}")
print(f"✅ Saved label encoders to {ENCODERS_FILE}")
print(f"✅ Saved scaler to {SCALER_FILE}")
print(f"Final row count after saving: {len(saved_data)}")