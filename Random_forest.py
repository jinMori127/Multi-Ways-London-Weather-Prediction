from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_prediction_vs_actual

# Load and preprocess the data
data = pd.read_csv('london_weather.csv', parse_dates=['date'])
data.fillna({
    "cloud_cover": data["cloud_cover"].mode()[0],
    "global_radiation": data["global_radiation"].mean(),
    "mean_temp": data["mean_temp"].mean()
}, inplace=True)
data.dropna(inplace=True)
data['date'] = pd.to_datetime(data['date'])

# Select relevant features
features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']
data = data[features]

# Calculate the indices for splitting (before creating sequences)
total_len = len(data)
train_size = int(total_len * 0.85)  # First 85% for training

# Split the data
train_data = data[:train_size].values
test_data = data[train_size:].values

# Create sequences after splitting
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())  # Flatten the sequences for RandomForest input
        y.append(data[i + seq_length])  # Predict the next timestep's features
    return np.array(X), np.array(y)

# Sequence length (days)
sequence_length = 7

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Train a multi-output RandomForestRegressor
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Plot predictions vs actual values
plot_prediction_vs_actual(y_pred, y_test, features, save_dir='random_forest_plots')

# Optionally print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
