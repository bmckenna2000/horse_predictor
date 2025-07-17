import pandas as pd
import os
from datetime import datetime
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Directory setup
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TODAY = datetime.today()
PREDICTION_FILE = os.path.join(DATA_DIR, "processed_daily_racecard.csv")
MODEL_FILE = os.path.join(DATA_DIR, "horse_race_winner_model.keras")  # Updated model file
ENCODERS_FILE = os.path.join(DATA_DIR, "label_encoders.pkl")
SCALER_FILE = os.path.join(DATA_DIR, "feature_scaler.pkl")
FEATURES_FILE = os.path.join(DATA_DIR, "predictive_features.pkl")
PICKS_FILE = os.path.join(DATA_DIR, f"winner_picks_{TODAY.strftime('%Y-%m-%d')}.csv")

# Define columns to exclude
outcome_cols = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner', 'target']
id_cols = ['race_id', 'date', 'off_time', 'field_size', 'field_size_unscaled']

def load_model_and_encoders():
    def focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.75):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(alpha * weight * cross_entropy)
    
    model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'focal_loss_fn': focal_loss_fn})
    print(f"✅ Loaded Keras model from {MODEL_FILE}, expecting {model.input_shape[1]} features")
    
    with open(ENCODERS_FILE, 'rb') as f:
        label_encoders = pickle.load(f)
    print("✅ Loaded label encoders")
    
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Loaded feature scaler")
    
    with open(FEATURES_FILE, 'rb') as f:
        predictive_features = pickle.load(f)
    print(f"✅ Loaded predictive features: {len(predictive_features)} features")
    
    return model, label_encoders, scaler, predictive_features

def preprocess_data(data, label_encoders, scaler, predictive_features):
    """Preprocess data with scaling and consistent encoding"""
    if 'field_size_unscaled' not in data.columns:
        field_sizes = data.groupby('race_id')['horse'].nunique().reset_index()
        field_sizes.columns = ['race_id', 'field_size_unscaled']
        data = data.merge(field_sizes, on='race_id', how='left')
    
    # Debug odds columns
    print(f"All columns in data: {data.columns.tolist()}")
    odds_raw = (data['raw_odds_decimal'].copy() if 'raw_odds_decimal' in data.columns 
                else data['odds_decimal'].copy() if 'odds_decimal' in data.columns 
                else pd.Series(np.nan, index=data.index))
    print(f"Raw odds sample (first 5): {odds_raw.head().tolist()}")
    
    data = data.drop(columns=[col for col in outcome_cols if col in data.columns], errors='ignore')
    
    missing_cols = [col for col in predictive_features if col not in data.columns]
    if missing_cols:
        print(f"❌ Missing features in processed_daily_racecard.csv: {missing_cols}. Cannot proceed.")
        return None, None, None
    
    extra_cols = [col for col in data.columns if col not in predictive_features + id_cols + ['field_size_unscaled']]
    if extra_cols:
        print(f"⚠️ Extra columns in data not used by model: {extra_cols}")
    
    X = data[predictive_features].copy()
    
    for col in ['horse', 'course']:
        if col in label_encoders:
            encoder = label_encoders[col]
            X[col] = X[col].apply(lambda x: x if x in encoder.classes_ else -1)
            X[col] = X[col].map(lambda x: encoder.transform([x])[0] if x != -1 else -1)
    
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"⚠️ NaNs in {nan_cols}. Filling with median.")
        X[nan_cols] = X[nan_cols].fillna(X[nan_cols].median())
    
    X_scaled = scaler.transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"✅ Preprocessed data shape: {X.shape}")
    print(f"Sample data (first 5 rows):\n{X.head()}")
    return X, data, odds_raw

def make_predictions(model, X, data, odds_raw, batch_size=64):
    scores = model.predict(X.values, batch_size=batch_size, verbose=0).flatten()
    scores = tf.clip_by_value(scores, 0.01, 0.99).numpy()
    data['predicted_prob'] = scores
    data['odds_decimal'] = odds_raw
    print(f"✅ Predictions made. Min prob: {min(scores):.6f}, Max prob: {max(scores):.6f}")
    print(f"Sample probs and odds (first 5):\n{data[['predicted_prob', 'odds_decimal']].head().to_string(index=False)}")
    return data

def select_top_picks(data, group_col='race_id', field_size_col='field_size_unscaled'):
    top_picks_list = []
    all_races = data[group_col].nunique()
    print(f"Total races in data: {all_races}")
    for race_id, group in data.groupby(group_col):
        field_size = int(group[field_size_col].iloc[0])
        print(f"Race {race_id}, Field size: {field_size}, Selecting top winner")
        predicted_winner = group.nlargest(1, 'predicted_prob')  # Select only the top horse
        top_picks_list.append(predicted_winner)
    top_picks = pd.concat(top_picks_list).reset_index(drop=True)
    # Relaxed filter to preserve races with unknown horses
    filtered_count = len(top_picks[(top_picks['horse'] == -1) & (top_picks['predicted_prob'] > 0.9)])
    top_picks = top_picks[~((top_picks['horse'] == -1) & (top_picks['predicted_prob'] > 0.9))]
    print(f"Filtered out {filtered_count} Unknown Horse entries with prob > 0.9")
    print(f"✅ Selected top picks, total: {len(top_picks)} across {top_picks[group_col].nunique()} races")
    return top_picks

def decode_names(data, label_encoders):
    """Decode horse and course names, handling unseen labels (-1)"""
    if 'horse' in label_encoders:
        horse_encoder = label_encoders['horse']
        data['horse_decoded'] = data['horse'].apply(
            lambda x: "Unknown Horse" if x == -1 else horse_encoder.inverse_transform([int(x)])[0]
        )
        print("✅ Decoded horse names with handling for unseen labels")
    else:
        print("❌ No horse encoder found")
        data['horse_decoded'] = data['horse']
    
    if 'course' in label_encoders:
        course_encoder = label_encoders['course']
        data['course_decoded'] = data['course'].apply(
            lambda x: "Unknown Course" if x == -1 else course_encoder.inverse_transform([int(x)])[0]
        )
        print("✅ Decoded course names with handling for unseen labels")
    else:
        print("❌ No course encoder found")
        data['course_decoded'] = data['course']
    
    return data

def save_picks_to_csv(data):
    output_cols = ['course_decoded', 'off_time', 'horse_decoded', 'predicted_prob', 'odds_decimal']  # Aligned with training script
    picks_data = data[output_cols].copy()
    
    picks_data['off_time_sort'] = pd.to_datetime(picks_data['off_time'], format='%H:%M', errors='coerce').dt.time
    picks_data = picks_data.sort_values(['course_decoded', 'off_time_sort', 'predicted_prob'], ascending=[True, True, False])
    picks_data = picks_data.drop(columns=['off_time_sort'])
    
    picks_data.to_csv(PICKS_FILE, index=False)
    print(f"✅ Saved picks to {PICKS_FILE}")
    print("Sample output grouped by course:\n")
    for course, group in picks_data.groupby('course_decoded'):
        print(f"Course: {course}")
        print(group[['off_time', 'horse_decoded', 'predicted_prob', 'odds_decimal']].to_string(index=False))
        print("\n")

def main():
    print(f"📡 Starting prediction process for {TODAY.strftime('%Y-%m-%d')}...")
    
    try:
        data = pd.read_csv(PREDICTION_FILE, low_memory=False)
        print(f"✅ Loaded data from {PREDICTION_FILE}, rows: {len(data)}, columns: {data.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    model, label_encoders, scaler, predictive_features = load_model_and_encoders()
    
    X, processed_data, odds_raw = preprocess_data(data, label_encoders, scaler, predictive_features)
    if X is None:
        return
    
    if X.shape[1] != model.input_shape[1]:
        print(f"❌ Feature mismatch! Model expects {model.input_shape[1]} features, got {X.shape[1]}")
        print(f"Expected features (based on training): {predictive_features}")
        print(f"Actual features provided: {X.columns.tolist()}")
        print("Please ensure the input data matches the features used in training.")
        return
    
    predicted_data = make_predictions(model, X, processed_data, odds_raw)
    
    top_picks = select_top_picks(predicted_data)
    
    decoded_picks = decode_names(top_picks, label_encoders)
    
    save_picks_to_csv(decoded_picks)
    
    print("✅ Prediction process completed")

if __name__ == "__main__":
    main()