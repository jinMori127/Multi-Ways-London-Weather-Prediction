from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_prediction_vs_actual


# Load and clean data
data = pd.read_csv('london_weather.csv')
data = data.dropna(axis=0)

# Set the date as input (x)
x = data[['date']]  # Assuming the 'date' column is numerical or encoded as integers

# Set the weather features as targets (y)
features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth']
y = data[features]

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=0)

# Train a multi-output RandomForestRegressor
clf = RandomForestRegressor(random_state=0)
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Evaluate the model for each feature
for i, feature in enumerate(features):
    mse = mean_squared_error(y_test[feature], y_pred[:, i])
    r2 = r2_score(y_test[feature], y_pred[:, i])
    print(f"Feature: {feature}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

# Compare predictions with actual values for the first few samples
comparison = pd.DataFrame(y_pred, columns=features)
comparison['Date'] = x_test.values  # Add the corresponding date to the prediction output
print(comparison.head())

# Feature importance plot for predicting weather features
importances = clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Predicting Weather Features')
plt.show()

# Call the plot function for all features

plot_prediction_vs_actual(y_pred, y_test, features, save_dir='random_forest_plots')
