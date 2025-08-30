<Code>
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import json
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import TextVectorization, StringLookup, Embedding, GlobalAveragePooling1D, Dense, Input, concatenate

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

# Normalize sentiment and when scores
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

# Define original training parameters
original_loss = {
    'sentiment': 'categorical_crossentropy',
    'when': 'categorical_crossentropy',
    'kind': 'binary_crossentropy'
}

original_metrics = {
    'sentiment': 'accuracy',
    'when': 'accuracy',
    'kind': 'accuracy'
}

# Keras Tuner integration
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Hyperparameters
        layers_count = hp.Int('layers', 2, 8)
        units = hp.Int('units', 64, 1024, step=64)
        dropout = hp.Float('dropout', 0.0, 0.5)
        lr = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')
        
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
            optimizer=kt.optimizers.Adam(lr),
            loss=original_loss,
            metrics=original_metrics
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        bs = hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024])
        ep = hp.Int('epochs', 20, 200, step=10)
        return model.fit(*args, batch_size=bs, epochs=ep, **kwargs)

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=50,
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

tuner.search(
    [X_train_text, X_train_state, X_train_loc], y_train,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

# Train model
start_time = time.time()
history = model.fit(
    [X_train_text, X_train_state, X_train_loc], y_train,
    epochs=200,
    batch_size=best_hps.get('batch_size'),
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)
training_time = time.time() - start_time


# Calculate average accuracy across all outputs
avg_train_acc = (history.history['sentiment_accuracy'][-1] + 
                 history.history['when_accuracy'][-1] + 
                 history.history['kind_accuracy'][-1]) / 3

avg_val_acc = (history.history['val_sentiment_accuracy'][-1] +
               history.history['val_when_accuracy'][-1] +
               history.history['val_kind_accuracy'][-1]) / 3

# Prepare results with single metrics
results = {
    'training_accuracy': avg_train_acc,
    'training_loss': history.history['loss'][-1], # Total loss
    'validation_accuracy': avg_val_acc,
    'validation_loss': history.history['val_loss'][-1], # Total validation loss
    'training_time_seconds': training_time
}

# Save results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print(f"Final Validation Loss: {results['validation_loss']:.4f}")
print(f"Final Validation Accuracy (Avg): {results['validation_accuracy']:.4f}")
print("Aggregated results saved to results.json")


# Predict
pred_sentiment, pred_when, pred_kind = model.predict(
    {'text': test_text.values, 'state': test_state.values, 'location': test_location.values})

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    's1': pred_sentiment[:,0], 's2': pred_sentiment[:,1],
    's3': pred_sentiment[:,2], 's4': pred_sentiment[:,3],
    's5': 0.0,
    'w1': pred_when[:,0], 'w2': pred_when[:,1],
    'w3': pred_when[:,2], 'w4': pred_when[:,3]
})

for i in range(15):
    submission[f'k{i+1}'] = pred_kind[:,i]

submission = submission[['id','s1','s2','s3','s4','s5','w1','w2','w3','w4',
                         'k1','k2','k3','k4','k5','k6','k7','k8','k9','k10',
                         'k11','k12','k13','k14','k15']]
submission.to_csv('submission.csv', index=False)
print("Submission file created.")
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/crowdflower-weather-twitter/crowdflower-weather-twitter_kt_solution.py", line 118, in <module>
    tuner = kt.BayesianOptimization(
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/tuners/bayesian.py", line 394, in __init__
    super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/tuner.py", line 122, in __init__
    super().__init__(
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 132, in __init__
    self._populate_initial_space()
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 192, in _populate_initial_space
    self._activate_all_conditions()
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 149, in _activate_all_conditions   
    self.hypermodel.build(hp)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/hypermodel.py", line 120, in _build_wrapper
    return self._build(hp, *args, **kwargs)
  File "/app/test/crowdflower-weather-twitter/crowdflower-weather-twitter_kt_solution.py", line 107, in build
    optimizer=kt.optimizers.Adam(lr),
AttributeError: module 'keras_tuner' has no attribute 'optimizers'
</Error>