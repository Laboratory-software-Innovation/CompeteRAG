<Code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
train = pd.read_csv('/home/exouser/GitHub/Version 1/REU/test/playground-series-s5e4/playground-series-s5e4/train.csv')
test = pd.read_csv('/home/exouser/GitHub/Version 1/REU/test/playground-series-s5e4/playground-series-s5e4/test.csv')

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

# Build model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train model
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=48,
                    validation_split=0.2,
                    callbacks=[early_stop, checkpoint],
                    verbose=0)

# Generate predictions
model.load_weights('best_model.h5')
predictions = model.predict(X_test).flatten()

# Create submission
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': predictions})
submission_df.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/home/exouser/GitHub/Version 1/REU/test/playground-series-s5e4/playground-series-s5e4_solution.py", line 22, in <module>
    df['Publication_Time'] = pd.to_datetime(df['Publication_Time'], format='%H:%M:%S').dt.hour
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/exouser/.local/lib/python3.12/site-packages/pandas/core/tools/datetimes.py", line 1063, in to_datetime
    cache_array = _maybe_cache(arg, format, cache, convert_listlike)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/exouser/.local/lib/python3.12/site-packages/pandas/core/tools/datetimes.py", line 247, in _maybe_cache
    cache_dates = convert_listlike(unique_dates, format)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/exouser/.local/lib/python3.12/site-packages/pandas/core/tools/datetimes.py", line 433, in _convert_listlike_datetimes
    return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/exouser/.local/lib/python3.12/site-packages/pandas/core/tools/datetimes.py", line 467, in _array_strptime_with_fallback
    result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "strptime.pyx", line 501, in pandas._libs.tslibs.strptime.array_strptime
  File "strptime.pyx", line 451, in pandas._libs.tslibs.strptime.array_strptime
  File "strptime.pyx", line 583, in pandas._libs.tslibs.strptime._parse_with_format
ValueError: time data "Night" doesn't match format "%H:%M:%S", at position 0. You might want to try:
    - passing `format` if your strings have a consistent format;
    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;
    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this.
</Error>
