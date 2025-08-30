import pandas as pd
import numpy as np
import time
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# Load and preprocess data
try:
    train = pd.read_csv('test/nyc-taxi-trip-duration/nyc-taxi-trip-duration/train.zip')
except FileNotFoundError:
    print("Ensure 'train.zip' and 'test.zip' from the NYC Taxi Trip Duration competition are in the correct directory.")
    exit()


# Feature engineering
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['hour'] = train['pickup_datetime'].dt.hour
train['weekday'] = train['pickup_datetime'].dt.weekday
train['month'] = train['pickup_datetime'].dt.month
train['haversine'] = haversine(train['pickup_longitude'], train['pickup_latitude'],
                               train['dropoff_longitude'], train['dropoff_latitude'])

# Prepare features and target
features = ['vendor_id', 'passenger_count', 'hour', 'weekday', 'month', 'haversine', 
            'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
categorical_features = ['vendor_id']
numeric_features = [f for f in features if f not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

X = preprocessor.fit_transform(train[features])
y = np.log1p(train['trip_duration'])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Dense(120, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(80, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(0.001), loss='mse', 
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Train model
start_time = time.time()
history = model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=2048,
          verbose=1)
training_time = time.time() - start_time


# Get the metrics from the final epoch
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['rmse'][-1]
final_val_mae = history.history['val_rmse'][-1]

# Create results dictionary
results = {
    'final_training_loss (rmse)': final_train_loss,
    'final_validation_loss (rmse)': final_val_loss,
    'final_training_rmse': final_train_mae,
    'final_validation_rmse': final_val_mae,
    'training_time_seconds': training_time
}

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print(f"Final Validation MAE: {final_val_mae}")
print("Results saved to results.json")



# Generate predictions on test data
try:
    test = pd.read_csv('test/nyc-taxi-trip-duration/nyc-taxi-trip-duration/test.zip')
except FileNotFoundError:
    print("Ensure 'test.zip' from the NYC Taxi Trip Duration competition is in the correct directory.")
    exit()

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['hour'] = test['pickup_datetime'].dt.hour
test['weekday'] = test['pickup_datetime'].dt.weekday
test['month'] = test['pickup_datetime'].dt.month
test['haversine'] = haversine(test['pickup_longitude'], test['pickup_latitude'],
                               test['dropoff_longitude'], test['dropoff_latitude'])

X_test = preprocessor.transform(test[features])
predictions = np.expm1(model.predict(X_test).flatten())

# Create submission
submission = pd.DataFrame({'id': test['id'], 'trip_duration': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")
