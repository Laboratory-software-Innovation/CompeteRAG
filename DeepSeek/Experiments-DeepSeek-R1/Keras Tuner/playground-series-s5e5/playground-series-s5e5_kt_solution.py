<Code>
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and prepare data
train_df = pd.read_csv('test/playground-series-s5e5/playground-series-s5e5/train.csv')
test_df = pd.read_csv('test/playground-series-s5e5/playground-series-s5e5/test.csv')
test_ids = test_df['id']

# Feature/target separation
X = train_df.drop(['id', 'Calories'], axis=1)
y = train_df['Calories']
X_test = test_df.drop('id', axis=1)

# Feature engineering
num_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_features = ['Sex']

# Missing value imputation
imputer = SimpleImputer(strategy='median')
X[num_features] = imputer.fit_transform(X[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])

# Categorical encoding
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

# Feature scaling
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Keras Tuner implementation
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
n_features = X_train.shape[1]

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        layers = hp.Int('layers', 2, 8)
        units = hp.Int('units', 64, 1024, step=64)
        dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')
        
        model.add(Dense(units, activation='relu', input_shape=(n_features,)))
        model.add(Dropout(dropout))
        
        for _ in range(layers-1):
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout))
        
        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
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
    project_name='calorie_tuner'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

# Training with early stopping
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=best_hps.get('batch_size'),
    callbacks=[early_stopping, checkpoint],
    verbose=0
)

# Generate predictions
predictions = model.predict(X_test).flatten()

# Create submission file
pd.DataFrame({'id': test_ids, 'Calories': predictions}).to_csv('sample_submission.csv', index=False)
</Code>

<Error>
When submiting I was provided witht the following error:

"Mean Squared Logarithmic Error cannot be used when targets contain negative values."

and also add this to the file:

# Create a dictionary with the final results
    results = {
        'final_training_loss': history.history['loss'][final_epoch],
        'final_training_mae': history.history['mae'][final_epoch],
        'final_validation_loss': history.history['val_loss'][final_epoch],
        'final_validation_mae': history.history['val_mae'][final_epoch],
        'training_time_seconds': training_time
    }
</Error>