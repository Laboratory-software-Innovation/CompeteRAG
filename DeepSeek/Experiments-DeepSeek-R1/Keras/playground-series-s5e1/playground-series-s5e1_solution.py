import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load data
train = pd.read_csv('test/playground-series-s5e1/playground-series-s5e1/train.csv')
test = pd.read_csv('test/playground-series-s5e1/playground-series-s5e1/test.csv')

# Preprocess dates
for df in [train, test]:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

# Handle missing target values
train['num_sold'] = train.groupby(['country', 'store', 'product'])['num_sold'].transform(
    lambda x: x.fillna(x.median()))
train['num_sold'] = train['num_sold'].fillna(train['num_sold'].median())

# Encode categorical features
cat_cols = ['country', 'store', 'product']
encoders = {col: LabelEncoder() for col in cat_cols}
for col in cat_cols:
    # Combine train and test for a complete set of labels
    all_labels = pd.concat([train[col], test[col]]).unique()
    encoders[col].fit(all_labels)
    train[f'{col}_enc'] = encoders[col].transform(train[col])
    test[f'{col}_enc'] = encoders[col].transform(test[col])

# Prepare numerical features
num_features = ['year', 'month_sin', 'month_cos', 'dayofweek_sin', 
                'dayofweek_cos', 'is_weekend', 'dayofyear']
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])

# Split training data based on date
train = train.sort_values('date')
cutoff = train['date'].quantile(0.8)
train_data = train[train['date'] < cutoff]
val_data = train[train['date'] >= cutoff]

# Prepare inputs
def get_inputs(df, num_features_df):
    return [
        df['country_enc'].values,
        df['store_enc'].values,
        df['product_enc'].values,
        num_features_df
    ]

X_train = get_inputs(train_data, train_data[num_features].values)
y_train = train_data['num_sold'].values
X_val = get_inputs(val_data, val_data[num_features].values)
y_val = val_data['num_sold'].values

# Build model
country_in = Input(shape=(1,), name='country')
store_in = Input(shape=(1,), name='store')
product_in = Input(shape=(1,), name='product')
num_in = Input(shape=(len(num_features),), name='numerical')

embed_size = 8

c = Flatten()(Embedding(len(encoders['country'].classes_), embed_size)(country_in))
s = Flatten()(Embedding(len(encoders['store'].classes_), embed_size)(store_in))
p = Flatten()(Embedding(len(encoders['product'].classes_), embed_size)(product_in))

concat = Concatenate()([c, s, p, num_in])
x = Dense(256, activation='relu')(concat)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1)(x)

model = Model([country_in, store_in, product_in, num_in], output)
model.compile(optimizer='adam', loss='huber', metrics=[tf.metrics.RootMeanSquaredError(name = 'rmse')]
) # Added mae for tracking

# Train model
es = EarlyStopping(patience=5, restore_best_weights=True)
start_time = time.time()
history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          batch_size=1024,
          callbacks=[es],
          verbose=2)
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds.")

# Generate predictions
test_inputs = [test['country_enc'], test['store_enc'], test['product_enc'], test[num_features].values]
predictions = model.predict(test_inputs).flatten()

# Create submission
submission = pd.DataFrame({'id': test['id'], 'num_sold': np.round(predictions).astype(int)})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")

# --- ADDED BLOCK: Save training results ---
if history and history.history:
    # Get the number of epochs the model actually ran for
    final_epoch = len(history.history['loss']) - 1
    
    # Create a dictionary with the final results
    results = {
        'final_training_loss': history.history['loss'][final_epoch],
        'final_training_rmse': history.history['rmse'][final_epoch],
        'final_validation_loss': history.history['val_loss'][final_epoch],
        'final_validation_rmse': history.history['val_rmse'][final_epoch],
        'training_time_seconds': training_time
    }

    # Save the dictionary to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Training results saved to results.json")
