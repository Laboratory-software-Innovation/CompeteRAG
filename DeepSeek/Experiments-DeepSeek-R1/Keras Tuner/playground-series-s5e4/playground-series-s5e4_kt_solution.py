<Code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time # Added for timing

# Load data
train = pd.read_csv('test/playground-series-s5e4/playground-series-s5e4/train.csv')
test = pd.read_csv('test/playground-series-s5e4/playground-series-s5e4/test.csv')

# Drop irrelevant features
train = train.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1)
test_ids = test['id']
test = test.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1)

# Preprocess datetime features
for df in [train, test]:
    df['Publication_Time'] = pd.to_datetime(df['Publication_Time'], format='%H:%M:%S').dt.hour
    df['Publication_Day'] = df['Publication_Day'].astype('category')

# Separate target
y_train = train.pop('Listening_Time_minutes')

# Define preprocessing
numerical = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
             'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Time']
categorical = ['Genre', 'Publication_Day', 'Episode_Sentiment']

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

cat_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical),
    ('cat', cat_pipeline, categorical)
])

X_train = preprocessor.fit_transform(train).astype('float32')
X_test = preprocessor.transform(test).astype('float32')

# ========== REPLACED MODEL BLOCK START ========== #
import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

n_features = X_train.shape[1]
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', min_value=2, max_value=8, step=1)
        units = hp.Int('units', min_value=64, max_value=1024, step=64)
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=0.01, sampling='log')
        batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256, 512, 1024])
        epochs = hp.Int('epochs', min_value=20, max_value=200, step=10)

        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout)(x)
        x = Dense(1)(x)
        
        model = Model(inputs, x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mean_squared_error',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.get('batch_size'),
            epochs=hp.get('epochs'),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='podcast_tuner'
)

tuner.search(
    X_train, y_train,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)
# =========== REPLACED MODEL BLOCK END =========== #

# Train model
start_time = time.time() # Start timer
history = model.fit(X_train, y_train,
                      epochs=200,
                      batch_size=48,
                      validation_split=0.2,
                      callbacks=[early_stop, checkpoint],
                      verbose=0)
end_time = time.time() # End timer

# ========== ADDED FOR METRICS REPORTING START ========== #

# Calculate training time
training_time = end_time - start_time

# Find the best epoch (since EarlyStopping(restore_best_weights=True) is used)
best_epoch = np.argmin(history.history['val_loss'])

# Retrieve the best scores from the history object
val_loss = history.history['val_loss'][best_epoch]
val_rmse = history.history['val_root_mean_squared_error'][best_epoch]
train_loss = history.history['loss'][best_epoch]
train_rmse = history.history['root_mean_squared_error'][best_epoch]

# Print the results
print("\n--- Model Training & Evaluation Summary ---")
print(f"Total Training Time: {training_time:.2f} seconds")
print(f"Stopped at Epoch: {best_epoch + 1}\n")

print("--- Final Metrics (from best epoch) ---")
print(f"Validation Loss (MSE): {val_loss:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}\n")
print(f"Training Loss (MSE): {train_loss:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print("-----------------------------------------\n")

# Note: "Testing" metrics are not possible as the test.csv file has no labels.
# The validation metrics serve as the performance benchmark on unseen data.

# ========== ADDED FOR METRICS REPORTING END ========== #

# Generate predictions
model.load_weights('best_model.h5')
predictions = model.predict(X_test).flatten()

# Create submission
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': predictions})
submission_df.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/playground-series-s5e4/playground-series-s5e4_kt_solution.py", line 24, in <module>
    df['Publication_Time'] = pd.to_datetime(df['Publication_Time'], format='%H:%M:%S').dt.hour
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/tools/datetimes.py", line 1068, in to_datetime
    cache_array = _maybe_cache(arg, format, cache, convert_listlike)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/tools/datetimes.py", line 249, in _maybe_cache
    cache_dates = convert_listlike(unique_dates, format)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/tools/datetimes.py", line 435, in _convert_listlike_datetimes
    return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/tools/datetimes.py", line 469, in _array_strptime_with_fallback
    result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
  File "pandas/_libs/tslibs/strptime.pyx", line 501, in pandas._libs.tslibs.strptime.array_strptime
  File "pandas/_libs/tslibs/strptime.pyx", line 451, in pandas._libs.tslibs.strptime.array_strptime
  File "pandas/_libs/tslibs/strptime.pyx", line 583, in pandas._libs.tslibs.strptime._parse_with_format
ValueError: time data "Night" doesn't match format "%H:%M:%S", at position 0. You might want to try:
    - passing `format` if your strings have a consistent format;
    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;
    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this.
</Error>