<Code>
import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load training data
train_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/train.csv')
X = train_df.drop(['row_id', 'target'], axis=1).values
y = train_df['target']

# Encode labels
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)
y_categorical = to_categorical(encoded_y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
start_time = time.time()
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=1024,
                    validation_data=(X_val, y_val),
                    verbose=1)
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds.")

# Load test data
test_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/test.csv')
X_test = scaler.transform(test_df.drop(['row_id'], axis=1).values)

# Generate predictions
probabilities = model.predict(X_test)
predicted_labels = encoder.inverse_transform(np.argmax(probabilities, axis=1))

# Create submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'target': predicted_labels
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")

# --- ADDED BLOCK: Save training results ---
if history and history.history:
    # Get the number of epochs the model actually ran for
    final_epoch = len(history.history['loss']) - 1
    
    # Create a dictionary with the final results
    results = {
        'final_training_loss': history.history['loss'][final_epoch],
        'final_training_accuracy': history.history['accuracy'][final_epoch],
        'final_validation_loss': history.history['val_loss'][final_epoch],
        'final_validation_accuracy': history.history['val_accuracy'][final_epoch],
        'training_time_seconds': training_time
    }

    # Save the dictionary to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Training results saved to results.json")
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022_solution.py", line 13, in <module>
    train_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/train.csv')
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 1898, in _make_engine
    return mapping[engine](f, **self.options)
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/c_parser_wrapper.py", line 93, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 574, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 663, in pandas._libs.parsers.TextReader._get_header
  File "pandas/_libs/parsers.pyx", line 874, in pandas._libs.parsers.TextReader._tokenize_rows
  File "pandas/_libs/parsers.pyx", line 891, in pandas._libs.parsers.TextReader._check_tokenize_status
  File "pandas/_libs/parsers.pyx", line 2053, in pandas._libs.parsers.raise_parser_error
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x99 in position 17: invalid start byte
</Error>