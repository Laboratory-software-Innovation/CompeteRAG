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

# Load training data with correct encoding
train_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/train.csv', encoding='latin-1')
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

# Train model with time tracking
start_time = time.time()
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=1024,
                    validation_data=(X_val, y_val),
                    verbose=1)
training_time = time.time() - start_time

# Load test data with correct encoding
test_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/test.csv', encoding='latin-1')
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

# Save training metrics
results = {
    'training_accuracy': history.history['accuracy'][-1],
    'training_loss': history.history['loss'][-1],
    'validation_accuracy': history.history['val_accuracy'][-1],
    'validation_loss': history.history['val_loss'][-1],
    'training_time_seconds': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022_fixed.py", line 13, in <module>
    train_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/train.csv', encoding='latin-1')
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 626, in _read
    return parser.read(nrows)
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py", line 1923, in read
    ) = self._engine.read(  # type: ignore[attr-defined]
  File "/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/c_parser_wrapper.py", line 234, in read
    chunks = self._reader.read_low_memory(nrows)
  File "pandas/_libs/parsers.pyx", line 838, in pandas._libs.parsers.TextReader.read_low_memory
  File "pandas/_libs/parsers.pyx", line 905, in pandas._libs.parsers.TextReader._read_rows
  File "pandas/_libs/parsers.pyx", line 874, in pandas._libs.parsers.TextReader._tokenize_rows
  File "pandas/_libs/parsers.pyx", line 891, in pandas._libs.parsers.TextReader._check_tokenize_status
  File "pandas/_libs/parsers.pyx", line 2061, in pandas._libs.parsers.raise_parser_error
pandas.errors.ParserError: Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.
</Error>