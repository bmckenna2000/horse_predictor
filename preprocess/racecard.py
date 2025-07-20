import pandas as pd
import numpy as np
import os
import re
import pickle

# Define the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
ENCODERS_FILE = os.path.join(DATA_DIR, 'label_encoders.pkl')
SCALER_FILE = os.path.join(DATA_DIR, 'scaler.pkl')
# 1. Load historical data (merged_test_data.csv) and compute aggregated historical means per horse.
historical_data = pd.read_csv(os.path.join(DATA_DIR, 'merged_test_data.csv'))
# Define the columns for which to compute historical means
hist_cols = ['last_run', 'lbs', 'rpr', 'odds_decimal', 'horse_distance_win_%', 'race_class']

# Group by horse_id and compute the mean for these columns from historical_data
historical_means = historical_data.groupby('horse_id')[hist_cols].mean().reset_index()
# Rename the aggregated columns to include a _hist_mean suffix and replace zeros with 1
for col in hist_cols:
    historical_means.rename(columns={col: f"{col}_hist_mean"}, inplace=True)
    historical_means[f"{col}_hist_mean"] = historical_means[f"{col}_hist_mean"].replace(0, 1)

# 2. Load the prediction file (merged_daily_racecard.csv) and deduplicate
prediction_data = pd.read_csv(os.path.join(DATA_DIR, 'merged_daily_racecard.csv'))
prediction_data = prediction_data.drop_duplicates(subset=['race_id', 'horse_id', 'date', 'off_time'], keep='first')
print(f"Rows after initial deduplication: {len(prediction_data)}")

# Set initial 'position' to 0 and create target flags
prediction_data['position'] = 0
prediction_data['winner'] = (prediction_data['position'] == 1).astype(int)
prediction_data['top_2'] = ((prediction_data['position'] <= 2) & (prediction_data['position'] > 0)).astype(int)
prediction_data['top_3'] = ((prediction_data['position'] <= 3) & (prediction_data['position'] > 0)).astype(int)
prediction_data['top_4'] = ((prediction_data['position'] <= 4) & (prediction_data['position'] > 0)).astype(int)
prediction_data['is_winner'] = (prediction_data['position'] == 1).astype(int)

def assign_target(row):
    field_size = row['field_size']
    if field_size <= 4:
        return row['winner']
    elif 5 <= field_size <= 7:
        return row['top_2']
    elif 8 <= field_size <= 15:
        return row['top_3']
    else:
        return row['top_4']

prediction_data['target'] = prediction_data.apply(assign_target, axis=1)

# 3. Compute form metrics
def compute_form_metrics(row):
    forms = [row['form_1'], row['form_2'], row['form_3'], row['form_4']]
    std_val = np.std(forms)
    consistency = 1 / (std_val + 1)
    trend = row['form_1'] - row['form_4']
    placements = [1 if (f > 0 and f <= 3) else 0 for f in forms]
    placement_ratio = sum(placements) / len(forms)
    recent_improvement = ((row['form_1'] + row['form_2']) / 2) - ((row['form_3'] + row['form_4']) / 2)
    momentum = ((row['form_1'] + row['form_2'] + row['form_3']) / 3) - row['form_4']
    return pd.Series({
        'form_consistency': consistency,
        'form_trend': trend,
        'form_placement_ratio': placement_ratio,
        'form_recent_improvement': recent_improvement,
        'form_momentum': momentum
    })

prediction_data[['form_consistency', 'form_trend', 'form_placement_ratio',
                 'form_recent_improvement', 'form_momentum']] = prediction_data.apply(compute_form_metrics, axis=1)

# 4. Compute placement ratios
prediction_data['horse_placement_ratio'] = pd.to_numeric(
    (prediction_data[['horse_distance_1st', 'horse_distance_2nd', 'horse_distance_3rd']].sum(axis=1)
     / prediction_data['horse_distance_runs'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_course_placement_ratio'] = pd.to_numeric(
    (prediction_data[['jockey_course_1st', 'jockey_course_2nd', 'jockey_course_3rd']].sum(axis=1)
     / prediction_data['jockey_course_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_distance_placement_ratio'] = pd.to_numeric(
    (prediction_data[['jockey_distance_1st', 'jockey_distance_2nd', 'jockey_distance_3rd']].sum(axis=1)
     / prediction_data['jockey_distance_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_owner_placement_ratio'] = pd.to_numeric(
    (prediction_data[['jockey_owner_1st', 'jockey_owner_2nd', 'jockey_owner_3rd']].sum(axis=1)
     / prediction_data['jockey_owner_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_trainer_placement_ratio'] = pd.to_numeric(
    (prediction_data[['jockey_trainer_1st', 'jockey_trainer_2nd', 'jockey_trainer_3rd']].sum(axis=1)
     / prediction_data['jockey_trainer_rides'].replace(0, 1)),
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

prediction_data['jockey_course_win_%'] = pd.to_numeric(
    (prediction_data['jockey_course_1st'] / prediction_data['jockey_course_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_distance_win_%'] = pd.to_numeric(
    (prediction_data['jockey_distance_1st'] / prediction_data['jockey_distance_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_owner_win_%'] = pd.to_numeric(
    (prediction_data['jockey_owner_1st'] / prediction_data['jockey_owner_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_trainer_win_%'] = pd.to_numeric(
    (prediction_data['jockey_trainer_1st'] / prediction_data['jockey_trainer_rides'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['jockey_win_%'] = pd.to_numeric(
    prediction_data[jockey_win_cols].mean(axis=1, skipna=True),
    errors='coerce'
).fillna(0)

prediction_data['final_placement_%'] = pd.to_numeric(
    prediction_data[placement_ratio_cols].mean(axis=1, skipna=True),
    errors='coerce'
).fillna(0)

prediction_data['consistency_score'] = pd.to_numeric(
    (prediction_data[['horse_distance_1st', 'horse_distance_2nd']].sum(axis=1)
     / prediction_data['horse_distance_runs'].replace(0, 1)),
    errors='coerce'
).fillna(0)

# 5. Merge historical means from merged_test_data into prediction_data via horse_id.
prediction_data = prediction_data.merge(historical_means, on='horse_id', how='left')
for col in [f"{col}_hist_mean" for col in hist_cols]:
    prediction_data[col].fillna(1, inplace=True)

# 6. Compute derived features using historical means:
prediction_data['last_run_relative'] = pd.to_numeric(
    prediction_data['last_run'] / prediction_data['last_run_hist_mean'], errors='coerce'
).fillna(0)

prediction_data['weight_relative'] = pd.to_numeric(
    prediction_data['lbs'] / prediction_data['lbs_hist_mean'], errors='coerce'
).fillna(0)

prediction_data['relative_weight_impact'] = pd.to_numeric(
    (prediction_data['weight_relative'] - 1.0) * prediction_data['horse_placement_ratio'],
    errors='coerce'
).fillna(0)

prediction_data['rpr_relative'] = pd.to_numeric(
    prediction_data['rpr'] / prediction_data['rpr_hist_mean'], errors='coerce'
).fillna(0)

prediction_data['rpr_place_interaction'] = pd.to_numeric(
    prediction_data['rpr_relative'] * prediction_data['final_placement_%'], errors='coerce'
).fillna(0)

prediction_data['race_rpr_variance'] = pd.to_numeric(
    prediction_data.groupby('horse_id')['rpr'].shift(1).rolling(5, min_periods=1).var(),
    errors='coerce'
).fillna(0)

prediction_data['odds_deviation'] = pd.to_numeric(
    prediction_data['odds_decimal'] / prediction_data['odds_decimal_hist_mean'], errors='coerce'
).fillna(0)

prediction_data['field_strength'] = pd.to_numeric(
    prediction_data['horse_distance_win_%_hist_mean'], errors='coerce'
).fillna(0)

prediction_data['trainer_jockey_synergy'] = pd.to_numeric(
    prediction_data['jockey_course_rides'] * (prediction_data['trainer_14_days_percent'] / 100),
    errors='coerce'
).fillna(0)

prediction_data['race_pace_interaction'] = pd.to_numeric(
    prediction_data['rpr'] * (prediction_data['distance_f'] / prediction_data['field_size'].replace(0, 1)),
    errors='coerce'
).fillna(0)

prediction_data['draw_advantage'] = pd.to_numeric(
    prediction_data['draw'] / prediction_data['field_size'].replace(0, 1),
    errors='coerce'
).fillna(0)

prediction_data['has_draw'] = pd.to_numeric(
    (prediction_data['draw'] > 0).astype(int),
    errors='coerce'
).fillna(0)

# 7. Ensure numeric columns for 'distance_f' and 'age', then round 'distance_f' for horse_age_distance
prediction_data['distance_f'] = pd.to_numeric(prediction_data['distance_f'], errors='coerce').fillna(0)
prediction_data['age'] = pd.to_numeric(prediction_data['age'], errors='coerce').fillna(0)
prediction_data['distance_f_int'] = prediction_data['distance_f'].round(0).astype(int)
prediction_data['horse_age_distance'] = prediction_data['age'] * prediction_data['distance_f_int']

# 8. Compute ranking and field-relative features
prediction_data['horse_rpr_rank_in_race'] = pd.to_numeric(
    prediction_data.groupby('race_id')['rpr'].rank(ascending=False, method='min'),
    errors='coerce'
).fillna(0)

prediction_data['horse_odds_rank_in_race'] = pd.to_numeric(
    prediction_data.groupby('race_id')['odds_decimal'].rank(ascending=True, method='min'),
    errors='coerce'
).fillna(0)

prediction_data['horse_rpr_relative_to_field_mean'] = pd.to_numeric(
    prediction_data.groupby('race_id')['rpr'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    ),
    errors='coerce'
).fillna(0)

prediction_data['pace_pressure_index'] = pd.to_numeric(
    prediction_data.groupby('race_id')['ts'].transform(lambda x: (x >= x.quantile(0.75)).sum()),
    errors='coerce'
).fillna(0)

# 9. Sort by horse_id and date for cumulative calculations and compute avg_historical_class

# 10. Compute suitability features:
# Jockey distance suitability
prediction_data['jockey_distance_suitability'] = prediction_data['jockey_distance_win_%'] / \
    prediction_data.groupby('jockey_id')['jockey_distance_win_%'].transform('mean')
prediction_data['jockey_distance_suitability'] = prediction_data['jockey_distance_suitability'].fillna(1)

# Jockey type suitability
prediction_data['jockey_type_suitability'] = prediction_data['jockey_win_%'] / \
    prediction_data.groupby(['jockey_id', 'type'])['jockey_win_%'].transform('mean')
prediction_data['jockey_type_suitability'] = prediction_data['jockey_type_suitability'].fillna(1)

# Horse distance suitability
prediction_data['horse_distance_suitability'] = prediction_data['horse_distance_win_%'] / \
    prediction_data.groupby('horse_id')['horse_distance_win_%'].transform('mean')
prediction_data['horse_distance_suitability'] = prediction_data['horse_distance_suitability'].fillna(1)

# Horse type suitability
prediction_data['horse_type_suitability'] = prediction_data['horse_distance_win_%'] / \
    prediction_data.groupby(['horse_id', 'type'])['horse_distance_win_%'].transform('mean')
prediction_data['horse_type_suitability'] = prediction_data['horse_type_suitability'].fillna(1)

prediction_data['date'] = pd.to_datetime(prediction_data['date'], errors='coerce')
prediction_data = prediction_data.sort_values(['horse_id', 'date'])

prediction_data['avg_historical_class'] = prediction_data.groupby('horse_id')['race_class'].transform(
    lambda x: x.shift(1).expanding().mean()
)
prediction_data['avg_historical_class'] = pd.to_numeric(prediction_data['avg_historical_class'], errors='coerce').fillna(0)

prediction_data['race_class_drop_impact'] = pd.to_numeric(
    (prediction_data['avg_historical_class'] - prediction_data['race_class']) * prediction_data['horse_distance_win_%'],
    errors='coerce'
).fillna(0)

prediction_data['jockey_course_familiarity_boost'] = pd.to_numeric(prediction_data['jockey_course_rides'].apply(lambda x: x * 1.1 if x > 50 else x), errors='coerce').fillna(0)
prediction_data['track_course_bias'] = pd.to_numeric(np.abs(prediction_data['draw'] - prediction_data.groupby(['course_id', 'distance_f', 'going'])['draw'].transform('mean')), errors='coerce').fillna(0)
# Final check: print sample outputs

prediction_data['rpr_lag_1'] = pd.to_numeric(prediction_data.groupby('horse_id')['rpr'].shift(1), errors='coerce').fillna(0)
prediction_data['rpr_lag_2'] = pd.to_numeric(prediction_data.groupby('horse_id')['rpr'].shift(2), errors='coerce').fillna(0)
prediction_data['rpr_lag_3'] = pd.to_numeric(prediction_data.groupby('horse_id')['rpr'].shift(3), errors='coerce').fillna(0)
prediction_data['horse_rpr_trend'] = pd.to_numeric((prediction_data['rpr'] - prediction_data['rpr_lag_3']) / 3, errors='coerce').fillna(0)

prediction_data = prediction_data.drop(columns=['rpr_lag_1', 'rpr_lag_2', 'rpr_lag_3', 'avg_historical_class'], errors='ignore')

prediction_data['rpr_ema'] = prediction_data.sort_values('date').groupby('horse_id')['rpr']\
    .transform(lambda x: x.ewm(span=3, adjust=False).mean())
    
prediction_data['jockey_horse_interaction'] = prediction_data['jockey_win_%'] * prediction_data['horse_distance_win_%']

# Market vs Model disagreement: difference between odds rank and RPR rank within each race
prediction_data['horse_rpr_rank_in_race'] = prediction_data.groupby('race_id')['rpr'].rank(ascending=False, method='min')
prediction_data['horse_odds_rank_in_race'] = prediction_data.groupby('race_id')['odds_decimal'].rank(ascending=True, method='min')
prediction_data['market_model_disagreement'] = prediction_data['horse_odds_rank_in_race'] - prediction_data['horse_rpr_rank_in_race']
prediction_data['race_par_score'] = prediction_data.groupby(['course', 'distance_f', 'race_class'])['rpr'].transform('mean')
prediction_data['max_rpr_in_race'] = prediction_data.groupby('race_id')['rpr'].transform('max')
prediction_data['rpr_diff_to_top'] = prediction_data['max_rpr_in_race'] - prediction_data['rpr']
prediction_data['rpr_rank'] = prediction_data.groupby('race_id')['rpr'].rank(method='min')
prediction_data['top_competitor_rpr'] = pd.to_numeric(prediction_data['rpr_hist_mean'], errors='coerce').fillna(0)
prediction_data['top_competitor_margin'] = pd.to_numeric(prediction_data['rpr'] - prediction_data['top_competitor_rpr'], errors='coerce').fillna(0)

prediction_data['relative_rpr'] = prediction_data.groupby('race_id')['rpr'].transform(lambda x: x - x.mean())

prediction_data['avg_class_last_5'] = prediction_data.sort_values('date').groupby('horse_id')['race_class']\
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
# Calculate Class Differential
prediction_data['class_diff'] = prediction_data['race_class'] - prediction_data['avg_class_last_5']
# Drop the temporary avg_class_last_5 column
prediction_data.drop(columns=['avg_class_last_5'], inplace=True)
# Ensure class_diff is numeric and handle NaNs
prediction_data['class_diff'] = pd.to_numeric(prediction_data['class_diff'], errors='coerce').fillna(0)

winning_distances = prediction_data[prediction_data['horse_distance_1st'] > 0].groupby('horse_id')['distance_f']\
    .mean().reset_index().rename(columns={'distance_f': 'avg_winning_distance'})
# Merge the average winning distance back into the main DataFrame
prediction_data = prediction_data.merge(winning_distances, on='horse_id', how='left')
# Fill NaN avg_winning_distance with the current distance_f (neutral assumption for horses with no wins)
prediction_data['avg_winning_distance'] = prediction_data['avg_winning_distance'].fillna(prediction_data['distance_f'])
# Compute Distance Suitability
prediction_data['distance_fit'] = prediction_data['horse_distance_win_%'] * \
    (1 - abs(prediction_data['distance_f'] - prediction_data['avg_winning_distance']) / prediction_data['distance_f'].replace(0, 1))
# Drop the temporary avg_winning_distance column
prediction_data.drop(columns=['avg_winning_distance'], inplace=True)
# Ensure distance_fit is numeric and handle NaNs or infinities
prediction_data['distance_fit'] = pd.to_numeric(prediction_data['distance_fit'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

going_surface_wins = prediction_data.groupby(['horse_id', 'going', 'surface'])['horse_distance_1st']\
    .mean().reset_index().rename(columns={'horse_distance_1st': 'going_surface_win_%'})
# Merge this back into the main DataFrame based on current going and surface
prediction_data = prediction_data.merge(going_surface_wins, on=['horse_id', 'going', 'surface'], how='left')
# Fill NaNs with the horse's overall horse_distance_win_% as a fallback
prediction_data['going_surface_win_%'] = prediction_data['going_surface_win_%'].fillna(prediction_data['horse_distance_win_%'])
# Ensure going_surface_win_% is numeric and handle any remaining NaNs
prediction_data['going_surface_win_%'] = pd.to_numeric(prediction_data['going_surface_win_%'], errors='coerce').fillna(0)

prediction_data['course_win_%'] = pd.to_numeric(
    prediction_data['jockey_course_1st'] / prediction_data['jockey_course_total_rides'].replace(0, 1),
    errors='coerce'
).fillna(0)

# --- Add New Feature: Relative Performance Index ---
# Compute combined rpr + ts for each horse
prediction_data['rpr_ts_combined'] = prediction_data['rpr'] + prediction_data['ts']
# Calculate the race-specific average of rpr + ts
race_avg_rpr_ts = prediction_data.groupby('race_id')['rpr_ts_combined'].mean()
# Map the race average back to the DataFrame
prediction_data['avg_rpr_ts'] = prediction_data['race_id'].map(race_avg_rpr_ts)
# Compute Relative Performance Index
prediction_data['rpi'] = prediction_data['rpr_ts_combined'] / prediction_data['avg_rpr_ts'].replace(0, 1)
# Drop temporary columns
prediction_data.drop(columns=['rpr_ts_combined', 'avg_rpr_ts'], inplace=True)
# Ensure rpi is numeric and handle NaNs or infinities
prediction_data['rpi'] = pd.to_numeric(prediction_data['rpi'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

avg_lbs = prediction_data['lbs'].mean()
# Compute Weight-Adjusted RPR
prediction_data['rpr_adjusted'] = prediction_data['rpr'] / (prediction_data['lbs'] / avg_lbs)
# Ensure rpr_adjusted is numeric and handle NaNs or infinities
prediction_data['rpr_adjusted'] = pd.to_numeric(prediction_data['rpr_adjusted'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

from sklearn.preprocessing import StandardScaler

# Copy your prediction_data so you don't modify the original
data = prediction_data.copy()
id_cols = ['race_id', 'date', 'off_time', 'field_size']
outcome_cols = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner', 'target']
# Define the column lists:
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

categorical_cols = ['course', 'horse']
id_cols = ['race_id', 'date', 'off_time', 'field_size']
outcome_cols = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner', 'target']

# Load pre-fitted label encoders and scaler
with open(ENCODERS_FILE, 'rb') as f:
    label_encoders = pickle.load(f)
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)

# Process categorical columns using pre-fitted label encoders
for col in categorical_cols:
    if col in prediction_data.columns:
        le = label_encoders[col]
        # Handle unseen labels by mapping to a default value (e.g., -1)
        prediction_data[col] = prediction_data[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        print(f"Transformed {col} with pre-fitted LabelEncoder")

# Ensure numeric columns are numeric and handle inf values
for col in numeric_cols:
    if col in prediction_data.columns:
        prediction_data[col] = pd.to_numeric(prediction_data[col], errors='coerce').fillna(0)
        prediction_data[col] = prediction_data[col].replace([np.inf, -np.inf], 0)

# Normalize numeric columns using pre-fitted scaler
features_to_scale = [col for col in numeric_cols if col in prediction_data.columns]
prediction_data[features_to_scale] = scaler.transform(prediction_data[features_to_scale])

# Drop unnecessary columns (same as preprocessing script)
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
prediction_data.drop(columns=[col for col in columns_to_drop if col in prediction_data.columns], inplace=True)

# Sort by date and race_id (consistent with preprocessing script)
prediction_data = prediction_data.sort_values(['date', 'race_id'])

# Final check: print sample outputs and shape
print(prediction_data.head())
print("Shape of final data:", prediction_data.shape)

# Optionally save the processed prediction data
output_file = os.path.join(DATA_DIR, 'processed_daily_racecard.csv')
prediction_data.to_csv(output_file, index=False, encoding='utf-8')
print(f"Saved processed prediction data to {output_file}")

# Final check: print sample outputs and shape
print(prediction_data.head())
print("Shape of final data:", prediction_data.shape)
