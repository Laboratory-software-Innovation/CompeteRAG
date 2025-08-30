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
import time
import json

# Load data
train = pd.read_csv('test/playground-series-s5e4/playground-series-s5e4/train.csv')
test = pd.read_csv('test/playground-series-s5e4/playground-series-s5e4/test.csv')

# Drop irrelevant features
train = train.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1)
test_ids = test['id']
test = test.drop(['id', 'Podcast_Name', 'Episode_Title'], axis=1)

# Preprocess categorical features
# Treat Publication_Time as a categorical feature to handle non-time values like "Night"
for df in [train, test]:
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')

# Separate target
y_train = train.pop('Listening_Time_minutes')

# Define preprocessing
numerical = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
             'Guest_Popularity_percentage', 'Number_of_Ads']
# Add Publication_Time to the list of categorical features
categorical = ['Genre', 'Publication_Day', 'Episode_Sentiment', 'Publication_Time']

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

# Build model using the new architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile model with specified optimizer and metrics
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train model with timing
start_time = time.time()
history = model.fit(X_train, y_train,
                      epochs=200,
                      batch_size=48,
                      validation_split=0.2,
                      callbacks=[early_stop, checkpoint],
                      verbose=1)
training_time = time.time() - start_time

# --- Section to save metrics to a file in the requested format ---
# Find the best epoch (since restore_best_weights=True is used)
best_epoch = np.argmin(history.history['val_loss'])

# Create a dictionary with the results from the best epoch
results = {
    "training_rmse": float(round(history.history['root_mean_squared_error'][best_epoch], 4)),
    "training_loss": float(round(history.history['loss'][best_epoch], 4)),
    "validation_rmse": float(round(history.history['val_root_mean_squared_error'][best_epoch], 4)),
    "validation_loss": float(round(history.history['val_loss'][best_epoch], 4)),
    "training_time_seconds": round(training_time, 2)
}

# Save the results dictionary to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
# --- END OF CHANGE ---

# Generate predictions
model.load_weights('best_model.h5')
predictions = model.predict(X_test).flatten()

# Create submission
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' and 'results.json' created successfully.")
