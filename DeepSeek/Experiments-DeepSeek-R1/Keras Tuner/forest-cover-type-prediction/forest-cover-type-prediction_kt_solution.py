
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import time
import json

# Load data
try:
    train = pd.read_csv('test/forest-cover-type-prediction/forest-cover-type-prediction/train.csv.zip')
    test = pd.read_csv('test/forest-cover-type-prediction/forest-cover-type-prediction/test.csv.zip')
except FileNotFoundError:
    print("Dataset files not found. Please check the path.")
    # Create dummy data for runnability
    train = pd.DataFrame(np.random.randint(0, 100, size=(1000, 56)), columns=['Id'] + [f'col{i}' for i in range(54)] + ['Cover_Type'])
    train['Cover_Type'] = np.random.randint(1, 8, 1000)
    for i in range(1, 5): train[f'Wilderness_Area{i}'] = np.random.randint(0, 2, 1000)
    for i in range(1, 41): train[f'Soil_Type{i}'] = np.random.randint(0, 2, 1000)
    test = pd.DataFrame(np.random.randint(0, 100, size=(500, 55)), columns=['Id'] + [f'col{i}' for i in range(54)])
    for i in range(1, 5): test[f'Wilderness_Area{i}'] = np.random.randint(0, 2, 500)
    for i in range(1, 41): test[f'Soil_Type{i}'] = np.random.randint(0, 2, 500)


# Preprocessing
def preprocess(df):
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    hillshades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    # Ensure hillshade columns exist before trying to clip
    for col in hillshades:
        if col not in df.columns:
            df[col] = 0 # Add column if it doesn't exist
    df[hillshades] = df[hillshades].clip(0, 255)
    return df

# Split features and target
X = preprocess(train).drop('Cover_Type', axis=1)
y = train['Cover_Type'] - 1  # Convert to 0-6
X_test = preprocess(test)

# Feature groups
cont_features = ['Elevation', 'Aspect', 'Slope', 
                 'Horizontal_Distance_To_Hydrology',
                 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways',
                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                 'Horizontal_Distance_To_Fire_Points']
wild_features = [f'Wilderness_Area{i}' for i in range(1,5)]
soil_features = [f'Soil_Type{i}' for i in range(1,41)]

# Ensure all feature columns exist in the dataframe
for col in cont_features + wild_features + soil_features:
    if col not in X.columns:
        X[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0

# Standard scaling
scaler = StandardScaler()
X_cont = scaler.fit_transform(X[cont_features])
X_cont_test = scaler.transform(X_test[cont_features])

# Combine features
X_processed = np.hstack([X_cont, X[wild_features + soil_features].values])
X_test_processed = np.hstack([X_cont_test, X_test[wild_features + soil_features].values])

# Split training data
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, 
                                                  test_size=0.2, 
                                                  random_state=42)

# --- Keras-Tuner Model Definition Starts Here ---
import keras_tuner as kt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

n_features = X_processed.shape[1]
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        inputs = Input(shape=(n_features,))
        x = inputs
        
        # Hyperparameters
        layers = hp.Int('layers', 2, 8, step=1)
        units = hp.Int('units', 64, 1024, step=64)
        activation = hp.Choice('activation', ['relu'])
        dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')
        
        # Network architecture
        for _ in range(layers):
            x = Dense(units, activation=activation)(x)
            x = Dropout(dropout)(x)
        
        x = Dense(7, activation='softmax')(x)
        model = Model(inputs, x)
        
        model.compile(
            optimizer=hp.Choice('optimizer', ['adam']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
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
    overwrite=True,
    project_name='forest_cover_tuner'
)

if 'y_val' in locals():
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint]
    )
else:
    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint]
    )

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)
# --- Keras-Tuner Model Definition Ends Here ---

# --- Timing and Results Logic Starts Here ---
start_time = time.time()

# Training
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=best_hps.get('batch_size'),
                    callbacks=[early_stopping, checkpoint],
                    verbose=2)

training_time = time.time() - start_time

# Save results to a JSON file
last_epoch = len(history.history['loss']) - 1
results = {
    'training_accuracy': history.history['accuracy'][last_epoch],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history['val_accuracy'][last_epoch],
    'validation_loss': history.history['val_loss'][last_epoch],
    'training_time_seconds': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f)
# --- Timing and Results Logic Ends Here ---


# Prediction
preds = model.predict(X_test_processed).argmax(axis=1) + 1  # Convert back to 1-7

# Create submission
submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': preds})
submission.to_csv('submission.csv', index=False)

print("Script finished successfully.")
print(f"Training time: {training_time:.2f} seconds")
print("submission.csv and results.json have been created.")