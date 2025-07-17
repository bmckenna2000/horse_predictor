import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

keras = tf.keras
models = tf.keras.models
layers = tf.keras.layers
optimizers = tf.keras.optimizers

# Custom Tee class (unchanged)
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("retrain_metrics.log", mode='a', encoding='utf-8')
    ]
)

# Redirect stdout
text_file = "retrain_terminal_output.txt"
txt_file = open(text_file, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, txt_file)

# Setup Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
daily_results_file = os.path.join(DATA_DIR, "preprocessed_retrain_data.csv")
encoders_file = os.path.join(DATA_DIR, 'label_encoders_retrain.pkl')
features_file = os.path.join(DATA_DIR, 'predictive_features.pkl')
scaler_file = os.path.join(DATA_DIR, 'feature_scaler.pkl')
model_file = os.path.join(DATA_DIR, 'horse_race_classification_model_recent.keras')

# Define Focal Loss
def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true + tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(alpha * weight * cross_entropy)
    return focal_loss_fn

# Load Artifacts
print("Loading model and preprocessing artifacts...")
with open(encoders_file, 'rb') as f:
    label_encoders = pickle.load(f)
horse_encoder = label_encoders['horse']
course_encoder = label_encoders.get('course')
with open(features_file, 'rb') as f:
    predictive_features = pickle.load(f)
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)
best_model = models.load_model(model_file, custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.75)})

# Load Preprocessed Retrain Data
print(f"Loading preprocessed retrain data from: {daily_results_file}")
daily_df = pd.read_csv(daily_results_file)

# Check for required columns
required_cols = ['date', 'race_id', 'horse', 'target'] + predictive_features
missing_cols = [col for col in required_cols if col not in daily_df.columns]
if missing_cols:
    logging.error(f"❌ Missing required columns in {daily_results_file}: {missing_cols}")
    sys.exit(1)

daily_df = daily_df.drop_duplicates()
print(f"Rows in retrain data: {len(daily_df)}")

# Convert date to datetime
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Preprocess Daily Results
print("Preprocessing retrain data...")
field_sizes = daily_df.groupby('race_id')['horse'].nunique().reset_index()
field_sizes.columns = ['race_id', 'field_size_unscaled']
daily_df = daily_df.merge(field_sizes, on='race_id', how='left')

daily_df = daily_df.dropna(subset=predictive_features + ['target'])
print(f"Rows after preprocessing: {len(daily_df)}")

# Check if DataFrame is empty
if daily_df.empty:
    logging.error("❌ No valid data remains after preprocessing. Check input data or encoder compatibility.")
    sys.exit(1)

# Define columns
outcome_cols = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner', 'target']
id_cols = ['race_id', 'date', 'off_time', 'field_size', 'field_size_unscaled']

# Prepare Features and Target
X_daily = daily_df[predictive_features].copy()
y_daily = daily_df['target']
odds_raw = daily_df['odds_decimal'].copy() if 'odds_decimal' in daily_df.columns else None

# Scale Features
X_daily_scaled = scaler.transform(X_daily)
X_daily = pd.DataFrame(X_daily_scaled, columns=X_daily.columns)

# Validate preprocessing
assert not X_daily.isna().any().any(), "X_daily contains NaNs!"
logging.info("Preprocessing checks passed.")

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_daily.values, y_daily)).batch(32)

# Recompile Model
best_model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks for Retraining
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logging.info(
            f"Epoch {epoch+1} - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
        )
    )
]

# Evaluate Before Retraining
print("Evaluating model on retrain data before retraining...")
y_pred_proba = best_model.predict(X_daily, batch_size=32).flatten()
calibrator = LogisticRegression()
calibrator.fit(y_pred_proba.reshape(-1, 1), y_daily)
y_pred_proba_calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
fpr, tpr, thresholds = roc_curve(y_daily, y_pred_proba_calibrated)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Calibrated optimal threshold (pre-retraining): {optimal_threshold:.4f}")
y_pred = (y_pred_proba_calibrated > optimal_threshold).astype(int)
print(f"Pre-retraining ROC-AUC (calibrated): {roc_auc_score(y_daily, y_pred_proba_calibrated):.4f}")
print(f"Pre-retraining F1-Score (optimal threshold): {f1_score(y_daily, y_pred):.4f}")
print(f"Pre-retraining Precision: {precision_score(y_daily, y_pred):.4f}")
print(f"Pre-retraining Recall: {recall_score(y_daily, y_pred):.4f}")
print(f"Pre-retraining Confusion Matrix:\n{confusion_matrix(y_daily, y_pred)}")
# Evaluate at manual threshold (0.3)
y_pred_manual = (y_pred_proba_calibrated > 0.3).astype(int)
print(f"Pre-retraining F1-Score (threshold 0.3): {f1_score(y_daily, y_pred_manual):.4f}")
print(f"Pre-retraining Precision (threshold 0.3): {precision_score(y_daily, y_pred_manual):.4f}")
print(f"Pre-retraining Recall (threshold 0.3): {recall_score(y_daily, y_pred_manual):.4f}")
logging.info(f"Pre-retraining ROC-AUC (calibrated): {roc_auc_score(y_daily, y_pred_proba_calibrated):.4f}")
logging.info(f"Pre-retraining F1-Score (optimal threshold): {f1_score(y_daily, y_pred):.4f}")
logging.info(f"Pre-retraining Precision: {precision_score(y_daily, y_pred):.4f}")
logging.info(f"Pre-retraining Recall: {recall_score(y_daily, y_pred):.4f}")
logging.info(f"Pre-retraining Confusion Matrix:\n{confusion_matrix(y_daily, y_pred)}")

# Structured Output Before Retraining
if odds_raw is not None:
    odds_unscaled = scaler.inverse_transform(daily_df[predictive_features])[:, predictive_features.index('odds_decimal')]
    structured_df = pd.DataFrame({
        'course_encoded': daily_df['course'] if 'course' in daily_df.columns else 'unknown',
        'off_time': daily_df['off_time'],
        'horse_encoded': daily_df['horse'],
        'predicted_probability': y_pred_proba_calibrated,
        'odds_decimal': odds_unscaled,
        'actual_outcome': y_daily
    })
    structured_df['course'] = course_encoder.inverse_transform(structured_df['course_encoded']) if 'course' in daily_df.columns else 'unknown'
    structured_df['horse'] = horse_encoder.inverse_transform(structured_df['horse_encoded'])
    structured_df = structured_df[['course', 'off_time', 'horse', 'predicted_probability', 'odds_decimal', 'actual_outcome']]
    structured_df = structured_df.sort_values(by=['course', 'off_time', 'predicted_probability'], ascending=[True, True, False])
    print("\nPre-retraining Structured Prediction Output:")
    print(structured_df.to_string(index=False))
    logging.info("\nPre-retraining Structured Prediction Output:\n" + structured_df.to_string(index=False))

# Retrain the Model
print("Starting retraining on retrain data...")
history = best_model.fit(
    train_dataset,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)
logging.info("Retraining completed.")
print("Retraining completed - proceeding to post-retraining evaluation...")

# Evaluate After Retraining
print("Evaluating model on retrain data after retraining...")
y_pred_proba = best_model.predict(X_daily, batch_size=32).flatten()
calibrator = LogisticRegression()
calibrator.fit(y_pred_proba.reshape(-1, 1), y_daily)
y_pred_proba_calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
fpr, tpr, thresholds = roc_curve(y_daily, y_pred_proba_calibrated)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Calibrated optimal threshold (post-retraining): {optimal_threshold:.4f}")
y_pred = (y_pred_proba_calibrated > optimal_threshold).astype(int)
print(f"Post-retraining ROC-AUC (calibrated): {roc_auc_score(y_daily, y_pred_proba_calibrated):.4f}")
print(f"Post-retraining F1-Score (optimal threshold): {f1_score(y_daily, y_pred):.4f}")
print(f"Post-retraining Precision: {precision_score(y_daily, y_pred):.4f}")
print(f"Post-retraining Recall: {recall_score(y_daily, y_pred):.4f}")
print(f"Post-retraining Confusion Matrix:\n{confusion_matrix(y_daily, y_pred)}")
# Evaluate at manual threshold (0.3)
y_pred_manual = (y_pred_proba_calibrated > 0.3).astype(int)
print(f"Post-retraining F1-Score (threshold 0.3): {f1_score(y_daily, y_pred_manual):.4f}")
print(f"Post-retraining Precision (threshold 0.3): {precision_score(y_daily, y_pred_manual):.4f}")
print(f"Post-retraining Recall (threshold 0.3): {recall_score(y_daily, y_pred_manual):.4f}")
logging.info(f"Post-retraining ROC-AUC (calibrated): {roc_auc_score(y_daily, y_pred_proba_calibrated):.4f}")
logging.info(f"Post-retraining F1-Score (optimal threshold): {f1_score(y_daily, y_pred):.4f}")
logging.info(f"Post-retraining Precision: {precision_score(y_daily, y_pred):.4f}")
logging.info(f"Post-retraining Recall: {recall_score(y_daily, y_pred):.4f}")
logging.info(f"Post-retraining Confusion Matrix:\n{confusion_matrix(y_daily, y_pred)}")

if odds_raw is not None:
    structured_df = pd.DataFrame({
        'course_encoded': daily_df['course'] if 'course' in daily_df.columns else 'unknown',
        'off_time': daily_df['off_time'],
        'horse_encoded': daily_df['horse'],
        'predicted_probability': y_pred_proba_calibrated,
        'odds_decimal': odds_raw,  # Use raw odds directly
        'actual_outcome': y_daily
    })
else:
    # Debug scaler index if odds_raw isn’t available
    odds_index = predictive_features.index('odds_decimal')
    odds_unscaled = scaler.inverse_transform(daily_df[predictive_features])[:, odds_index]
    structured_df = pd.DataFrame({
        'course_encoded': daily_df['course'] if 'course' in daily_df.columns else 'unknown',
        'off_time': daily_df['off_time'],
        'horse_encoded': daily_df['horse'],
        'predicted_probability': y_pred_proba_calibrated,
        'odds_decimal': odds_unscaled,
        'actual_outcome': y_daily
    })
    print(f"Debug: odds_unscaled sample: {odds_unscaled[:5]}")  # Check values

# Save Updated Model
model_save_path = os.path.join(DATA_DIR, 'horse_race_classification_model_recent.keras')
best_model.save(model_save_path, save_format='tf')
logging.info(f"Updated model saved to: {model_save_path}")
print(f"Updated model saved to: {model_save_path}")

# Close output
txt_file.close()
sys.stdout = sys.__stdout__
print("Retraining script completed successfully.")
logging.info("Retraining script completed successfully.")