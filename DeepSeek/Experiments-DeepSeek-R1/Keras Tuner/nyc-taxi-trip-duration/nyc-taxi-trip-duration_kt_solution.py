
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
import keras_tuner as kt
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

# Build and tune model with Keras Tuner
n_features = X_train.shape[1]
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', 2, 8)
        units = hp.Int('units', 64, 1024, step=64)
        dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')
        
        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout)(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate),
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')] # CHANGED to RMSE
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
            epochs=hp.Int('epochs', 20, 200, step=10),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=False,
    project_name='taxi_duration_tuner'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters()[0]
# model = tuner.hypermodel.build(best_hps)
model = tf.keras.models.load_model('best_model.h5', custom_objects={'rmse': tf.keras.metrics.RootMeanSquaredError()})
# Train model
start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    initial_epoch=68,
    batch_size=best_hps.get('batch_size'),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)
training_time = time.time() - start_time

# Get the metrics from the final epoch
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['rmse'][-1]
final_val_mae = history.history['val_rmse'][-1]

# Create results dictionary
results = {
    'final_training_loss (mse)': final_train_loss,
    'final_validation_loss (mse)': final_val_loss,
    'final_training_rmse': final_train_mae,
    'final_validation_rmse': final_val_mae,
    'training_time_seconds': training_time
}

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
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
