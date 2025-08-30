import pandas as pd
import numpy as np
import tensorflow as tf
import time
import json
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import TextVectorization, StringLookup, Embedding, GlobalAveragePooling1D, Dense, Input, concatenate
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess data
try:
    train_df = pd.read_csv('test/crowdflower-weather-twitter/crowdflower-weather-twitter/train.csv.zip')
    test_df = pd.read_csv('test/crowdflower-weather-twitter/crowdflower-weather-twitter/test.csv.zip')
except FileNotFoundError:
    print("Ensure 'train.csv.zip' and 'test.csv.zip' from the CrowdFlower Weather Twitter competition are in the correct directory.")
    exit()


# Process targets
sentiment_cols = ['s1','s2','s3','s4']
when_cols = ['w1','w2','w3','w4']
kind_cols = ['k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15']

for col in sentiment_cols + when_cols + kind_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)

# Normalize sentiment and when scores to act as distributions
train_df[sentiment_cols] = train_df[sentiment_cols].div(train_df[sentiment_cols].sum(axis=1), axis=0).fillna(0)
train_df[when_cols] = train_df[when_cols].div(train_df[when_cols].sum(axis=1), axis=0).fillna(0)

# Prepare inputs
train_text = train_df['tweet'].astype(str).fillna('')
train_state = train_df['state'].astype(str).fillna('')
train_location = train_df['location'].astype(str).fillna('')

test_text = test_df['tweet'].astype(str).fillna('')
test_state = test_df['state'].astype(str).fillna('')
test_location = test_df['location'].astype(str).fillna('')

# Build model components
text_input = Input(shape=(), dtype=tf.string, name='text')
state_input = Input(shape=(), dtype=tf.string, name='state')
location_input = Input(shape=(), dtype=tf.string, name='location')

text_vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
text_vectorizer.adapt(train_text)
text_features = text_vectorizer(text_input)
text_embed = Embedding(10001, 128)(text_features)
text_pool = GlobalAveragePooling1D()(text_embed)

state_lookup = StringLookup()
state_lookup.adapt(train_state)
state_indices = state_lookup(state_input)
state_embed = Embedding(state_lookup.vocabulary_size()+1, 8)(state_indices)
state_vec = layers.Reshape((-1,))(state_embed)

location_lookup = StringLookup()
location_lookup.adapt(train_location)
loc_indices = location_lookup(location_input)
loc_embed = Embedding(location_lookup.vocabulary_size()+1, 8)(loc_indices)
loc_vec = layers.Reshape((-1,))(loc_embed)

# Define losses and the new metrics dictionary
loss_dict = {
    'sentiment': 'categorical_crossentropy',
    'when': 'categorical_crossentropy',
    'kind': 'binary_crossentropy'
}

# --- KEY CHANGE: Define metrics for each output ---
# This ensures RMSE is calculated and recorded for each head of the model.
metrics_dict = {
    'sentiment': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
    'when': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
    'kind': tf.keras.metrics.RootMeanSquaredError(name='rmse')
}

# Keras Tuner integration
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Hyperparameters
        layers_count = hp.Int('layers', 2, 8)
        units = hp.Int('units', 64, 512, step=64)
        dropout = hp.Float('dropout', 0.1, 0.5, step=0.1)
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        
        # Architecture construction
        combined = concatenate([text_pool, state_vec, loc_vec])
        x = combined
        for _ in range(layers_count):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
        
        # Output layers
        sentiment_output = Dense(4, activation='softmax', name='sentiment')(x)
        when_output = Dense(4, activation='softmax', name='when')(x)
        kind_output = Dense(15, activation='sigmoid', name='kind')(x)
        
        model = Model(
            inputs=[text_input, state_input, location_input],
            outputs=[sentiment_output, when_output, kind_output]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss_dict,
            # --- KEY CHANGE: Use the metrics dictionary here ---
            metrics=metrics_dict
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        bs = hp.Choice('batch_size', [64, 128, 256, 512])
        return model.fit(*args, batch_size=bs, **kwargs)

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='tweet_scoring_tuner'
)

# Prepare training data
X_train_text = train_text.values
X_train_state = train_state.values
X_train_loc = train_location.values
y_train = {
    'sentiment': train_df[sentiment_cols].values,
    'when': train_df[when_cols].values,
    'kind': train_df[kind_cols].values
}

# The tuner will now track val_sentiment_rmse, val_when_rmse, etc.
tuner.search(
    [X_train_text, X_train_state, X_train_loc], y_train,
    epochs=50, # Set a reasonable number of epochs for the search
    validation_split=0.2,
    callbacks=[early_stopping] # Checkpoint is not needed during search
)

print("\n--- Training the best model ---")
best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

# Train the final model with the best hyperparameters
start_time = time.time()
history = model.fit(
    [X_train_text, X_train_state, X_train_loc], y_train,
    epochs=100, # Train for more epochs
    batch_size=best_hps.get('batch_size'),
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint], # Use checkpoint here to save the best version
    verbose=1
)
training_time = time.time() - start_time


# --- KEY CHANGE: Record all RMSE metrics from history ---
# The history object now contains specific RMSE values for each output.
results = {
    'final_training_loss': history.history['loss'][-1],
    'final_validation_loss': history.history['val_loss'][-1],
    'sentiment_train_rmse': history.history['sentiment_rmse'][-1],
    'sentiment_val_rmse': history.history['val_sentiment_rmse'][-1],
    'when_train_rmse': history.history['when_rmse'][-1],
    'when_val_rmse': history.history['val_when_rmse'][-1],
    'kind_train_rmse': history.history['kind_rmse'][-1],
    'kind_val_rmse': history.history['val_kind_rmse'][-1],
    'training_time_seconds': training_time
}


# Save results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print(f"Final Validation Loss: {results['final_validation_loss']:.4f}")
print(f"Final Validation RMSE (Sentiment): {results['sentiment_val_rmse']:.4f}")
print(f"Final Validation RMSE (When): {results['when_val_rmse']:.4f}")
print(f"Final Validation RMSE (Kind): {results['kind_val_rmse']:.4f}")
print("Detailed results with RMSE saved to results.json")


# Predict on the test set
print("\nGenerating predictions on the test set...")
pred_sentiment, pred_when, pred_kind = model.predict(
    [test_text.values, test_state.values, test_location.values])

# Create submission file
submission = pd.DataFrame({'id': test_df['id']})
for i, col in enumerate(sentiment_cols):
    submission[col] = pred_sentiment[:, i]
submission['s5'] = 0.0 # Add the missing s5 column as per original format
for i, col in enumerate(when_cols):
    submission[col] = pred_when[:, i]
for i, col in enumerate(kind_cols):
    submission[col] = pred_kind[:, i]

# Ensure column order matches the sample submission
ordered_cols = ['id'] + sentiment_cols + ['s5'] + when_cols + kind_cols
submission = submission[ordered_cols]
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")
