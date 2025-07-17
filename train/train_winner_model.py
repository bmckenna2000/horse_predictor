import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import pickle
import shutil
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from keras_tuner import RandomSearch

keras = tf.keras
models = tf.keras.models
layers = tf.keras.layers
optimizers = tf.keras.optimizers

# Custom Tee class
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
        logging.FileHandler("train_softmax_metrics.log", mode='w', encoding='utf-8')
    ]
)

# Redirect stdout
text_file = "terminal_output_softmax.txt"
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

# Check field size distribution
field_size_counts = training_df['field_size_unscaled'].value_counts().sort_index()
print("Field size distribution:")
print(field_size_counts)
logging.info("Field size distribution:\n" + field_size_counts.to_string())

# Determine maximum field size (capped for memory efficiency)
max_horses = min(20, int(training_df['field_size_unscaled'].max()))
print(f"Maximum number of horses per race (capped): {max_horses}")

# Validate position column
if 'position' not in training_df.columns:
    logging.error("❌ 'position' column missing from training data!")
    sys.exit(1)
if training_df['position'].isna().any():
    logging.warning(f"Found {training_df['position'].isna().sum()} NaN values in 'position'; filling with 0.")
    training_df['position'] = training_df['position'].fillna(0)

# Prepare data for multi-class classification
def prepare_race_data(df, predictive_features, max_horses):
    race_data = []
    race_labels = []
    race_ids = []
    race_field_sizes = []
    race_horse_ids = []
    race_courses = []
    race_off_times = []
    race_odds = []
    problem_races = []
    
    for race_id, group in df.groupby('race_id'):
        try:
            group = group.sort_values('horse').reset_index(drop=True)
            field_size = len(group)
            if field_size > max_horses:
                logging.warning(f"Race {race_id} has {field_size} horses, exceeding max_horses ({max_horses}); skipping.")
                problem_races.append(race_id)
                continue
            
            if group[predictive_features].isna().any().any():
                logging.warning(f"Race {race_id} has NaN values in predictive features; skipping.")
                problem_races.append(race_id)
                continue
            
            X = group[predictive_features].values
            X_padded = np.zeros((max_horses, len(predictive_features)))
            X_padded[:field_size] = X
            
            y = np.zeros(max_horses)
            winner_idx = group[group['position'] == 1].index
            
            if len(winner_idx) != 1:
                logging.warning(f"Race {race_id} has {len(winner_idx)} winners (based on position == 1); skipping.")
                problem_races.append(race_id)
                continue
            y[winner_idx[0]] = 1
            
            race_data.append(X_padded)
            race_labels.append(y)
            race_ids.append(race_id)
            race_field_sizes.append(field_size)
            race_horse_ids.append(group['horse'].values)
            race_courses.append(group['course'].values[0])
            race_off_times.append(group['off_time'].values[0])
            race_odds.append(group['odds_decimal'].values)
        
        except Exception as e:
            logging.error(f"Error processing race {race_id}: {str(e)}")
            problem_races.append(race_id)
            continue
    
    # Save problematic races for inspection
    if problem_races:
        with open(os.path.join(DATA_DIR, 'problem_races.txt'), 'w') as f:
            f.write('\n'.join(map(str, problem_races)))
        logging.info(f"Saved {len(problem_races)} problematic race IDs to problem_races.txt")
    
    if not race_data:
        logging.error("❌ No valid races after processing. Check data for errors.")
        sys.exit(1)
    
    return (np.array(race_data), np.array(race_labels), race_ids, race_field_sizes,
            race_horse_ids, race_courses, race_off_times, race_odds)

# Split data by race_id
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(training_df, groups=training_df['race_id']):
    train_split = training_df.iloc[train_idx].reset_index(drop=True)
    test_split = training_df.iloc[test_idx].reset_index(drop=True)
    break

# Prepare train and test data
try:
    X_train_races, y_train_races, train_race_ids, train_field_sizes, train_horse_ids, _, _, _ = prepare_race_data(
        train_split, predictive_features, max_horses)
    X_test_races, y_test_races, test_race_ids, test_field_sizes, test_horse_ids, test_courses, test_off_times, test_odds = prepare_race_data(
        test_split, predictive_features, max_horses)
except Exception as e:
    logging.error(f"Error preparing data: {str(e)}")
    sys.exit(1)

# Scale features
try:
    scaler = StandardScaler()
    X_train_flat = X_train_races.reshape(-1, len(predictive_features))
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_train_races_scaled = X_train_flat_scaled.reshape(X_train_races.shape)
    X_test_flat = X_test_races.reshape(-1, len(predictive_features))
    X_test_flat_scaled = scaler.transform(X_test_flat)
    X_test_races_scaled = X_test_flat_scaled.reshape(X_test_races.shape)
except Exception as e:
    logging.error(f"Error scaling features: {str(e)}")
    sys.exit(1)

with open(os.path.join(DATA_DIR, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
logging.info("Feature scaler saved.")

# Validate preprocessing
assert not np.isnan(X_train_races_scaled).any(), "X_train contains NaNs!"
assert not np.isnan(X_test_races_scaled).any(), "X_test contains NaNs!"
logging.info("NaN checks passed.")

# Convert to tf.data.Dataset
try:
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_races_scaled, y_train_races)).shuffle(1000).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test_races_scaled, y_test_races)).batch(64)
except Exception as e:
    logging.error(f"Error creating datasets: {str(e)}")
    sys.exit(1)

# Define Model for Multi-Class Classification
def build_classification_model(input_dim, output_dim):
    inputs = layers.Input(shape=(max_horses, input_dim))
    x = layers.TimeDistributed(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    x_skip1 = x
    x = layers.TimeDistributed(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    if x_skip1.shape[-1] != x.shape[-1]:
        x_skip1 = layers.TimeDistributed(layers.Dense(x.shape[-1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(x_skip1)
    x = layers.Add()([x, x_skip1])
    x_skip2 = x
    x = layers.TimeDistributed(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    if x_skip2.shape[-1] != x.shape[-1]:
        x_skip2 = layers.TimeDistributed(layers.Dense(x.shape[-1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(x_skip2)
    x = layers.Add()([x, x_skip2])
    x = layers.TimeDistributed(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)
    x = layers.Reshape((max_horses,))(x)
    outputs = layers.Softmax(axis=-1)(x)
    return models.Model(inputs, outputs)

# Hyperparameter Tuning
def build_tuned_model(hp):
    model = build_classification_model(
        input_dim=len(predictive_features),
        output_dim=max_horses
    )
    model.compile(
        optimizer=optimizers.AdamW(
            learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log'),
            weight_decay=hp.Float('wd', 1e-4, 1e-2, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    return model

# Clear previous tuner directory
tuner_dir = os.path.join(os.getcwd(), 'tuner_dir', 'horse_race_softmax')
if os.path.exists(tuner_dir):
    shutil.rmtree(tuner_dir)
logging.info("Starting hyperparameter tuning...")

try:
    tuner = RandomSearch(
        build_tuned_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='horse_race_softmax'
    )
    tuner.search(train_dataset, epochs=1, validation_data=test_dataset, verbose=1)
    logging.info("Hyperparameter tuning completed.")
except Exception as e:
    logging.error(f"Error during hyperparameter tuning: {str(e)}")
    sys.exit(1)

best_model = tuner.get_best_models(num_models=1)[0]

# Custom Precision@N Metric for Softmax (Winner Focus)
def precision_at_n_softmax(model, X_races, race_ids, field_sizes, horse_ids, y_true):
    results = []
    try:
        y_pred = model.predict(X_races, batch_size=64, verbose=0)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return 0
    
    for i, (race_id, field_size, horse_ids_race, y_true_race) in enumerate(
            zip(race_ids, field_sizes, horse_ids, y_true)):
        try:
            field_size = int(field_size)
            n = 1  # Focus on top-1 (winner)
            pred_probs = y_pred[i][:field_size]
            predicted_top_n_idx = np.argsort(pred_probs)[-n:]
            actual_top_n_idx = np.where(y_true_race[:field_size] == 1)[0]
            if not actual_top_n_idx.size:
                results.append(0.0)
                logging.warning(f"Race {race_id} has no winners in test set; check data consistency.")
            else:
                intersection_count = len(set(predicted_top_n_idx) & set(actual_top_n_idx))
                results.append(intersection_count / min(n, len(actual_top_n_idx)))
        except Exception as e:
            logging.error(f"Error evaluating race {race_id}: {str(e)}")
            continue
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
try:
    history = best_model.fit(
        train_dataset,
        epochs=1,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    logging.info("Final training completed.")
except Exception as e:
    logging.error(f"Error during training: {str(e)}")
    sys.exit(1)

print("Final training completed - proceeding to evaluation...")

# Evaluation
print("Starting evaluation...")
try:
    print("Calculating Precision@N...")
    prec_n = precision_at_n_softmax(
        best_model, X_test_races_scaled, test_race_ids, test_field_sizes, test_horse_ids, y_test_races)
    print(f"Precision@N (Top-1) on test set: {prec_n:.4f}")
    logging.info(f"Precision@N (Top-1) on test set: {prec_n:.4f}")

    print("Predicting probabilities...")
    y_pred_proba = best_model.predict(X_test_races_scaled, batch_size=64)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_races, axis=1)
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")

    # Multi-class F1-score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-Score (weighted): {f1:.4f}")
    logging.info(f"F1-Score (weighted): {f1:.4f}")

    # Structured Output
    print("Generating structured output...")
    structured_data = []
    for i, (race_id, course, off_time, field_size, horse_ids_race, odds_race, pred_probs) in enumerate(
            zip(test_race_ids, test_courses, test_off_times, test_field_sizes, test_horse_ids, test_odds, y_pred_proba)):
        field_size = int(field_size)
        for j in range(field_size):
            structured_data.append({
                'course_encoded': course,
                'off_time': off_time,
                'horse_encoded': horse_ids_race[j],
                'predicted_probability': pred_probs[j],
                'odds_decimal': odds_race[j] if j < len(odds_race) else np.nan
            })
    
    structured_df = pd.DataFrame(structured_data)
    structured_df['course'] = course_encoder.inverse_transform(structured_df['course_encoded'])
    structured_df['horse'] = horse_encoder.inverse_transform(structured_df['horse_encoded'])
    structured_df = structured_df[['course', 'off_time', 'horse', 'predicted_probability', 'odds_decimal']]
    structured_df = structured_df.sort_values(
        by=['course', 'off_time', 'predicted_probability'], ascending=[True, True, False])

    print("\nStructured Prediction Output:")
    print(structured_df.to_string(index=False))
    logging.info("\nStructured Prediction Output:\n" + structured_df.to_string(index=False))

except Exception as e:
    logging.error(f"Error during evaluation: {str(e)}")
    raise

# Plot Training History
try:
    print("Generating training history plot...")
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, 'training_loss_softmax.png'))
    plt.close()
    logging.info("Training history plot saved.")
except Exception as e:
    logging.error(f"Error generating plot: {str(e)}")

# Save Model
model_save_path = os.path.join(DATA_DIR, 'horse_race_softmax_model.keras')
try:
    best_model.save(model_save_path, save_format='tf')
    logging.info(f"Model saved to: {model_save_path}")
    print(f"Model saved to: {model_save_path}")
except Exception as e:
    logging.error(f"Error saving model: {str(e)}")

# Close output
txt_file.close()
sys.stdout = sys.__stdout__
print("Script completed successfully.")
logging.info("Script completed successfully.")