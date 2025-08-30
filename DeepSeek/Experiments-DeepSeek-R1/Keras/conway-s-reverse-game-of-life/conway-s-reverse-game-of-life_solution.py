import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, concatenate, Reshape, UpSampling2D
from tensorflow.keras.optimizers import Adam
import time
import json

# Load and preprocess training data
try:
    train_df = pd.read_csv('test/conway-s-reverse-game-of-life/conway-s-reverse-game-of-life/train.csv.zip')
    test_df = pd.read_csv('test/conway-s-reverse-game-of-life/conway-s-reverse-game-of-life/test.csv.zip')
except FileNotFoundError:
    print("Dataset files not found. Using dummy data.")
    # Create dummy data for runnability
    num_train = 1000
    num_test = 500
    train_df = pd.DataFrame({'delta': np.random.randint(1, 6, num_train)})
    for i in range(1, 401):
        train_df[f'start.{i}'] = np.random.randint(0, 2, num_train)
        train_df[f'stop.{i}'] = np.random.randint(0, 2, num_train)
    
    test_df = pd.DataFrame({'id': range(num_test), 'delta': np.random.randint(1, 6, num_test)})
    for i in range(1, 401):
        test_df[f'stop.{i}'] = np.random.randint(0, 2, num_test)


# Extract features and targets
x_stop = train_df[[f'stop.{i}' for i in range(1, 401)]].values.reshape(-1, 20, 20, 1).astype('float32')
x_delta = train_df['delta'].values.reshape(-1, 1).astype('float32')
y_start = train_df[[f'start.{i}' for i in range(1, 401)]].values.reshape(-1, 20, 20, 1).astype('float32')

# Model architecture
image_input = Input(shape=(20, 20, 1))
delta_input = Input(shape=(1,))

# Image processing branch
x = Conv2D(128, (3,3), activation='relu', padding='same')(image_input)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

# Delta processing branch
d = Dense(128, activation='relu')(delta_input)
d = Dense(128, activation='relu')(d)
d = Reshape((1, 1, 128))(d)
d = UpSampling2D(size=(20, 20))(d)

# Combine branches
combined = concatenate([x, d])
x = Conv2D(128, (3,3), activation='relu', padding='same')(combined)
x = BatchNormalization()(x)
x = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

model = Model(inputs=[image_input, delta_input], outputs=x)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# --- Timing and Results Logic Starts Here ---
start_time = time.time()

# Train model
history = model.fit([x_stop, x_delta], y_start, epochs=15, batch_size=64, validation_split=0.2)

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



# Load and process test data
x_test_stop = test_df[[f'stop.{i}' for i in range(1, 401)]].values.reshape(-1, 20, 20, 1).astype('float32')
x_test_delta = test_df['delta'].values.reshape(-1, 1).astype('float32')

# Generate predictions
predictions = (model.predict([x_test_stop, x_test_delta]) > 0.5).astype(int).reshape(-1, 400)

# Create submission file
submission = pd.DataFrame(
    predictions,
    columns=[f'start.{i}' for i in range(1, 401)]
)
submission.insert(0, 'id', test_df['id'])
submission.to_csv('submission.csv', index=False)

print("Script finished successfully.")
print(f"Training time: {training_time:.2f} seconds")
print("submission.csv and results.json have been created.")
