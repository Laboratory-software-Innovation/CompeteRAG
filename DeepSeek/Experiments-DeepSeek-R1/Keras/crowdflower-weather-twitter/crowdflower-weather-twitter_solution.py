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
state_embed = Embedding(state_lookup.vocabulary_size()+1, 8)(state_indices)
state_vec = layers.Reshape((-1,))(state_embed)

location_lookup = StringLookup()
location_lookup.adapt(train_location)
loc_indices = location_lookup(location_input)
loc_embed = Embedding(location_lookup.vocabulary_size()+1, 8)(loc_indices)
loc_vec = layers.Reshape((-1,))(loc_embed)

combined = concatenate([text_pool, state_vec, loc_vec])
x = Dense(128, activation='relu')(combined)
x = Dense(64, activation='relu')(x)

sentiment_output = Dense(4, activation='softmax', name='sentiment')(x)
when_output = Dense(4, activation='softmax', name='when')(x)
kind_output = Dense(15, activation='sigmoid', name='kind')(x)

model = Model(inputs=[text_input, state_input, location_input],
              outputs=[sentiment_output, when_output, kind_output])

model.compile(optimizer='adam',
              loss={'sentiment': 'categorical_crossentropy',
                    'when': 'categorical_crossentropy',
                    'kind': 'binary_crossentropy'},
              metrics=['accuracy'])

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
    verbose=1
)
training_time = time.time() - start_time

# Prepare results
results = {
    'training_accuracy': {
        'sentiment': history.history['sentiment_accuracy'][-1],
        'when': history.history['when_accuracy'][-1],
        'kind': history.history['kind_accuracy'][-1]
    },
    'training_loss': {
        'sentiment': history.history['sentiment_loss'][-1],
        'when': history.history['when_loss'][-1],
        'kind': history.history['kind_loss'][-1]
    },
    'validation_accuracy': {
        'sentiment': history.history['val_sentiment_accuracy'][-1],
        'when': history.history['val_when_accuracy'][-1],
        'kind': history.history['val_kind_accuracy'][-1]
    },
    'validation_loss': {
        'sentiment': history.history['val_sentiment_loss'][-1],
        'when': history.history['val_when_loss'][-1],
        'kind': history.history['val_kind_loss'][-1]
    },
    'training_time_seconds': training_time
}

# Save results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print("Results saved to results.json")

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
  File "/app/test/crowdflower-weather-twitter/crowdflower-weather-twitter_fixed.py", line 80, in <module>
    history = model.fit(
  File "/usr/local/lib/python3.10/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.10/dist-packages/keras/src/trainers/compile_utils.py", line 254, in _build_metrics_set
    raise ValueError(
ValueError: For a model with multiple outputs, when providing the `metrics` argument as a list, it should have as many entries as the model has outputs. Received:
metrics=['accuracy']
of length 1 whereas the model has 3 outputs.
</Error>