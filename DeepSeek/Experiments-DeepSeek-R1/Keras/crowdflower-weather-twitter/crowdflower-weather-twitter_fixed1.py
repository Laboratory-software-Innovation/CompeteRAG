import pandas as pd
import numpy as np
import tensorflow as tf
import time
import json
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import TextVectorization, StringLookup, Embedding, GlobalAveragePooling1D, Dense, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping

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

# Normalize sentiment and when scores (ensure rows sum to 1, or 0 if all are 0)
train_df[sentiment_cols] = train_df[sentiment_cols].div(train_df[sentiment_cols].sum(axis=1), axis=0).fillna(0)
train_df[when_cols] = train_df[when_cols].div(train_df[when_cols].sum(axis=1), axis=0).fillna(0)

# Prepare inputs
train_text = train_df['tweet'].astype(str).fillna('')
train_state = train_df['state'].astype(str).fillna('')
train_location = train_df['location'].astype(str).fillna('')

test_text = test_df['tweet'].astype(str).fillna('')
test_state = test_df['state'].astype(str).fillna('')
test_location = test_df['location'].astype(str).fillna('')

# Build model
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
state_embed = Embedding(state_lookup.vocabulary_size(), 8)(state_indices)
state_vec = layers.Reshape((-1,))(state_embed)

location_lookup = StringLookup()
location_lookup.adapt(train_location)
loc_indices = location_lookup(location_input)
loc_embed = Embedding(location_lookup.vocabulary_size(), 8)(loc_indices)
loc_vec = layers.Reshape((-1,))(loc_embed)

combined = concatenate([text_pool, state_vec, loc_vec])
x = Dense(128, activation='relu')(combined)
x = Dense(64, activation='relu')(x)

sentiment_output = Dense(4, activation='softmax', name='sentiment')(x)
when_output = Dense(4, activation='softmax', name='when')(x)
kind_output = Dense(15, activation='sigmoid', name='kind')(x)

model = Model(inputs=[text_input, state_input, location_input],
              outputs=[sentiment_output, when_output, kind_output])

# FIXED: Provided metrics as a dictionary, mapping each output to its desired metric.
model.compile(optimizer='adam',
              loss={'sentiment': 'categorical_crossentropy',
                    'when': 'categorical_crossentropy',
                    'kind': 'binary_crossentropy'},
              metrics={
                  'sentiment': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                  'when': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                  'kind': tf.keras.metrics.RootMeanSquaredError(name='rmse')
              })

# Train model
start_time = time.time()
history = model.fit(
    {'text': train_text.values, 'state': train_state.values, 'location': train_location.values},
    {'sentiment': train_df[sentiment_cols].values,
     'when': train_df[when_cols].values,
     'kind': train_df[kind_cols].values},
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')],
    verbose=1
)
training_time = time.time() - start_time

# Get metrics from the best epoch and for each output
best_epoch = np.argmin(history.history['val_loss'])

results = {
    'best_epoch': int(best_epoch + 1),
    'training_time_seconds': training_time,
    'final_training_loss': history.history['loss'][best_epoch],
    'final_validation_loss': history.history['val_loss'][best_epoch],
    'sentiment_training_rmse': history.history['sentiment_rmse'][best_epoch],
    'sentiment_validation_rmse': history.history['val_sentiment_rmse'][best_epoch],
    'when_training_rmse': history.history['when_rmse'][best_epoch],
    'when_validation_rmse': history.history['val_when_rmse'][best_epoch],
    'kind_training_rmse': history.history['kind_rmse'][best_epoch],
    'kind_validation_rmse': history.history['val_kind_rmse'][best_epoch],
}


# Save results
with open('results.json', 'w') as f:
    # Use a custom encoder to handle numpy float32 types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    json.dump(results, f, indent=4, cls=NpEncoder)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print(f"Results from best epoch ({results['best_epoch']}) saved to results.json")


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
