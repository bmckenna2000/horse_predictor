import pandas as pd
import os
import logging
import pickle
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("evaluate.log", mode='w')]
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RECENT_HALF_FILE = os.path.join(DATA_DIR, "recent_half_race_data.csv")
PICKS_FILE = os.path.join(DATA_DIR, "race_picks.csv")
ENCODERS_FILE = os.path.join(DATA_DIR, "label_encoders.pkl")

def load_data():
    """Load recent half data and picks."""
    recent_df = pd.read_csv(RECENT_HALF_FILE)
    picks_df = pd.read_csv(PICKS_FILE)
    with open(ENCODERS_FILE, 'rb') as f:
        label_encoders = pickle.load(f)
    logging.info(f"Loaded recent half data: {recent_df.shape}, picks: {picks_df.shape}")
    return recent_df, picks_df, label_encoders

def evaluate_predictions(recent_df, picks_df, label_encoders):
    """Compare picks to actual outcomes (winner, top_2, top_3, top_4, target, is_winner)."""
    # Decode horse names in recent_df
    if 'horse' in label_encoders:
        recent_df['horse_decoded'] = label_encoders['horse'].inverse_transform(recent_df['horse'].astype(int))
    
    # Merge picks with recent data to get actual outcomes
    outcome_cols = ['winner', 'top_2', 'top_3', 'top_4', 'target', 'is_winner']
    eval_df = pd.merge(
        picks_df,
        recent_df[['race_id', 'horse_decoded'] + outcome_cols],
        on=['race_id', 'horse_decoded'],
        how='left'
    )
    
    # Log sample
    logging.info(f"Evaluation sample (first 5 rows):")
    logging.info(eval_df[['race_id', 'horse_decoded', 'predicted_prob'] + outcome_cols].head().to_string())
    
    # Calculate accuracy metrics
    total_picks = len(eval_df)
    
    for pos in outcome_cols:
        correct = (eval_df[pos] == 1).sum()
        accuracy = correct / total_picks if total_picks > 0 else 0
        logging.info(f"Accuracy ({pos}=1): {accuracy:.4f} ({correct}/{total_picks})")
    
    # Precision@N evaluation (using target)
    prec_n_results = []
    for race_id, group in eval_df.groupby('race_id'):
        field_size = group['field_size'].iloc[0]
        n = 1 if field_size <= 4 else 2 if field_size <= 7 else 3 if field_size <= 15 else 4
        actual_placed = group['target'].sum()
        correct_in_top_n = group['target'].head(n).sum()
        prec_n = correct_in_top_n / min(n, actual_placed) if actual_placed > 0 else 0
        prec_n_results.append(prec_n)
    
    prec_n = np.mean(prec_n_results) if prec_n_results else 0
    logging.info(f"Precision@N on recent half: {prec_n:.4f}")
    
    # Save evaluation
    eval_df.to_csv(os.path.join(DATA_DIR, "evaluated_recent_half.csv"), index=False)
    logging.info(f"Saved evaluation to evaluated_recent_half.csv")

if __name__ == "__main__":
    recent_df, picks_df, label_encoders = load_data()
    evaluate_predictions(recent_df, picks_df, label_encoders)