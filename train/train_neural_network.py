import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import pickle
import shutil
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from imblearn.over_sampling import SMOTE
from keras_tuner import RandomSearch

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
        logging.FileHandler("train_recent_metrics.log", mode='w', encoding='utf-8')
    ]
)

# Redirect stdout
text_file = "terminal_output.txt"
txt_file = open(text_file, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, txt_file)

# Setup and Load Training Data
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
training_file = os.path.join(DATA_DIR, 'preprocessed_training_data.csv')
encoders_file = os.path.join(DATA_DIR, 'label_encoders.pkl')
features_file = os.path.join(DATA_DIR, 'predictive_features.pkl')
print(f"Loading training data from: {training_file}")
training_df = pd.read_csv(training_file)
print(f"Rows before deduplication: {len(training_df)}")
training_df = training_df.drop_duplicates()

# Add field_size_unscaled
field_sizes = training_df.groupby('race_id')['horse'].nunique().reset_index()
field_sizes.columns = ['race_id', 'field_size_unscaled']
training_df = training_df.merge(field_sizes, on='race_id', how='left')

# Class distribution and compute class weights
class_counts = training_df['target'].value_counts()
positive_cases = class_counts.get(1, 0)
negative_cases = class_counts.get(0, 0)
imbalance_ratio = negative_cases / positive_cases if positive_cases > 0 else float('inf')
# Compute class weights for binary cross-entropy
weight_for_0 = 1.0
weight_for_1 = min(imbalance_ratio, 10.0)  # Cap to stabilize training
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Class distribution - Positive (1): {positive_cases}, Negative (0): {negative_cases}")
print(f"Class weights - Negative: {weight_for_0:.2f}, Positive: {weight_for_1:.2f}")
logging.info(f"Class distribution - Positive: {positive_cases}, Negative: {negative_cases}, Ratio: {imbalance_ratio:.2f}")
logging.info(f"Class weights - Negative: {weight_for_0:.2f}, Positive: {weight_for_1:.2f}")

# Validate data
if training_df.empty:
    logging.error("❌ The training dataframe is EMPTY. Check your data loading step!")
    sys.exit(1)

# Ensure date is datetime
training_df['date'] = pd.to_datetime(training_df['date'])
print("--- Debug Marker 0.7: After Date Conversion ---")

# Load label encoders
with open(encoders_file, 'rb') as f:
    label_encoders = pickle.load(f)
horse_encoder = label_encoders['horse']
course_encoder = label_encoders.get('course')
print("--- Debug Marker 0.8: After Encoder Load ---")

# Define columns
outcome_cols = ['position', 'winner', 'top_2', 'top_3', 'top_4', 'is_winner', 'target']
id_cols = ['race_id', 'date', 'off_time', 'field_size', 'field_size_unscaled']
predictive_features = [col for col in training_df.columns if col not in (id_cols + outcome_cols)]
print(f"Number of predictive features: {len(predictive_features)}")
print(f"Predictive features: {predictive_features}")

# Save predictive features for prediction script
with open(features_file, 'wb') as f:
    pickle.dump(predictive_features, f)
logging.info(f"Predictive features saved to: {features_file}")

# Stratified Group Split
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(training_df, groups=training_df['race_id']):
    train_split = training_df.iloc[train_idx].reset_index(drop=True)
    test_split = training_df.iloc[test_idx].reset_index(drop=True)
    break  # Use one fold

X_train = train_split[predictive_features].copy()
X_test = test_split[predictive_features].copy()
y_train = train_split['target']
y_test = test_split['target']

# Store raw odds before scaling
odds_raw = test_split['odds_decimal'].copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
with open(os.path.join(DATA_DIR, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
logging.info("Feature scaler saved.")

# Validate preprocessing
assert not X_train.isna().any().any(), "X_train contains NaNs!"
assert not X_test.isna().any().any(), "X_test contains NaNs!"
logging.info("NaN checks passed.")

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train)).shuffle(1000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test)).batch(64)

# Define Deeper Model with Multiple Skip Connections
def build_classification_model(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    # Layer 1: 512 neurons (wide input layer)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)  # Increased dropout for early layer
    x_skip1 = x  # Store for first skip connection
    # Layer 2: 256 neurons
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    # Skip connection: Project x_skip1 if dimensions differ
    if x_skip1.shape[-1] != x.shape[-1]:
        x_skip1 = layers.Dense(x.shape[-1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x_skip1)
    x = layers.Add()([x, x_skip1])  # First residual connection
    x_skip2 = x  # Store for second skip connection
    # Layer 3: 128 neurons
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    # Skip connection: Project x_skip2 if dimensions differ
    if x_skip2.shape[-1] != x.shape[-1]:
        x_skip2 = layers.Dense(x.shape[-1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x_skip2)
    x = layers.Add()([x, x_skip2])  # Second residual connection
    # Layer 4: 64 neurons
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Layer 5: 32 neurons
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# Hyperparameter Tuning with Keras Tuner
def build_tuned_model(hp):
    model = build_classification_model(input_dim=len(X_train.columns))
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log'),
            weight_decay=hp.Float('wd', 1e-4, 1e-2, sampling='log')
        ),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# Clear previous tuner directory to force fresh tuning
tuner_dir = os.path.join(os.getcwd(), 'tuner_dir', 'horse_race')
if os.path.exists(tuner_dir):
    shutil.rmtree(tuner_dir)
logging.info("Starting hyperparameter tuning...")

tuner = RandomSearch(
    build_tuned_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='horse_race'
)
tuner.search(train_dataset, epochs=50, validation_data=test_dataset, verbose=1)
logging.info("Hyperparameter tuning completed.")

best_model = tuner.get_best_models(num_models=1)[0]

# Custom Precision@N Metric (unchanged)
def precision_at_n(df, model, group_col='race_id', target_col='target', field_size_col='field_size_unscaled', batch_size=64):
    df = df.reset_index(drop=True)
    columns_to_drop = [col for col in [group_col, target_col, 'date', 'off_time'] + outcome_cols if col in df.columns]
    X_all = df.drop(columns=columns_to_drop)
    X_all = X_all[X_train.columns].values
    scores = model.predict(X_all, batch_size=batch_size, verbose=0).flatten()
    scores = tf.clip_by_value(scores, 0.01, 0.99).numpy()
    df['pred_score'] = scores
    results = []
    for race_id, group in df.groupby(group_col):
        group = group.reset_index(drop=True)
        field_size = int(group[field_size_col].iloc[0])
        n = 1 if field_size <= 4 else 2 if field_size <= 7 else 3 if field_size <= 15 else 4
        predicted_top_n_idx = group.nlargest(n, 'pred_score').index.tolist()
        actual_top_n_idx = np.where(group[target_col] == 1)[0].tolist()
        if not actual_top_n_idx:
            results.append(0.0)
        else:
            intersection_count = len(set(predicted_top_n_idx) & set(actual_top_n_idx))
            results.append(intersection_count / min(n, len(actual_top_n_idx)))
    return np.mean(results) if results else 0

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logging.info(
            f"Epoch {epoch+1} - Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}, "
            f"Accuracy: {logs['accuracy']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}"
        )
    )
]

# Train the best model
logging.info("Starting final training with best model...")
history = best_model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)
logging.info("Final training completed.")
print("Final training completed - proceeding to evaluation...")

# Evaluation with Adjusted Threshold and Structured Output
print("Starting evaluation...")
try:
    test_df = pd.concat([X_test, y_test, test_split['race_id'], test_split['field_size_unscaled']], axis=1).reset_index(drop=True)
    print("Calculating Precision@N...")
    prec_n = precision_at_n(test_df, best_model)
    print(f"Precision@N on test set: {prec_n:.4f}")
    logging.info(f"Precision@N on test set: {prec_n:.4f}")

    print("Predicting probabilities...")
    y_pred_proba = best_model.predict(X_test, batch_size=64).flatten()
    y_pred = (y_pred_proba > 0.7).astype(int)
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score (threshold 0.7): {f1_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix (threshold 0.7):\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    logging.info(f"F1-Score (threshold 0.7): {f1_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion Matrix (threshold 0.7):\n{confusion_matrix(y_test, y_pred)}")

    # Structured Output: course (decoded), off_time, horse (decoded), predicted probability, raw odds
    print("Generating structured output...")
    structured_df = pd.DataFrame({
        'course_encoded': test_split['course'],
        'off_time': test_split['off_time'],
        'horse_encoded': test_split['horse'],
        'predicted_probability': y_pred_proba,
        'odds_decimal': odds_raw
    })

    # Decode course and horse
    structured_df['course'] = course_encoder.inverse_transform(structured_df['course_encoded'])
    structured_df['horse'] = horse_encoder.inverse_transform(structured_df['horse_encoded'])

    # Drop encoded columns and sort for readability
    structured_df = structured_df[['course', 'off_time', 'horse', 'predicted_probability', 'odds_decimal']]
    structured_df = structured_df.sort_values(by=['course', 'off_time', 'predicted_probability'], ascending=[True, True, False])

    # Print and log structured output
    print("\nStructured Prediction Output:")
    print(structured_df.to_string(index=False))
    logging.info("\nStructured Prediction Output:\n" + structured_df.to_string(index=False))

except Exception as e:
    print(f"Error during evaluation: {str(e)}")
    logging.error(f"Error during evaluation: {str(e)}")
    raise

# Plot Training History
print("Generating training history plot...")
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.savefig(os.path.join(DATA_DIR, 'training_loss.png'))
plt.close()
logging.info("Training history plot saved.")

# Save Model
model_save_path = os.path.join(DATA_DIR, 'horse_race_classification_model_recent.keras')
best_model.save(model_save_path, save_format='tf')
logging.info(f"Model saved to: {model_save_path}")
print(f"Model saved to: {model_save_path}")

# Close output
txt_file.close()
sys.stdout = sys.__stdout__
print("Script completed successfully.")
logging.info("Script completed successfully.")