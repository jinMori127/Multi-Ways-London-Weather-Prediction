
import os 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
    
class WeatherDataset(Dataset):
    def __init__(self, data, label_columns=None):
        self.features = data.drop(label_columns, axis=1)
        self.features['year'] = self.features['date'].dt.year
        self.features['month'] = self.features['date'].dt.month
        self.features['day'] = self.features['date'].dt.day
        self.features.drop('date', axis=1, inplace=True)
        self.labels = data[label_columns].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

   
def load_weather_data(batch_size):
    full_data = pd.read_csv('london_weather.csv', parse_dates=['date'])
    
    # Fill missing values
    full_data.fillna({
        "cloud_cover": full_data["cloud_cover"].mode()[0],
        "global_radiation": full_data["global_radiation"].mean(),
        "mean_temp": full_data["mean_temp"].mean()
    }, inplace=True)
    
    # Drop any remaining rows with NaN values
    full_data.dropna(inplace=True)

    # Specify feature columns for scaling
    features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']

    columns_to_keep = ['date'] + features

    # Filter the DataFrame to keep only the desired columns
    full_data = full_data[columns_to_keep]

    # Create a new DataFrame with original data for date handling
    scaled_data = full_data.copy()
    
    # # Scale the feature data
    # scaler = StandardScaler()
    # scaled_data[features] = scaler.fit_transform(scaled_data[features])

    # Calculate the indices for splitting
    train_size = int(len(scaled_data) * 0.8)      # First 80%
    val_size = int(len(scaled_data) * 0.1)        # Next 10%
    
    # Split the data
    train_data = scaled_data.iloc[:train_size]     # First 80%
    val_data = scaled_data.iloc[train_size:train_size + val_size]  # Next 10%
    test_data = scaled_data.iloc[train_size + val_size:]  # Last 10%


    # Create DataLoaders
    trainloader = DataLoader(WeatherDataset(data=train_data, label_columns=features), batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(WeatherDataset(data=val_data, label_columns=features), batch_size=batch_size, shuffle=False, pin_memory=True)
    testloader = DataLoader(WeatherDataset(data=test_data, label_columns=features), batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Print the date ranges for each dataset
    print(f"Training data date range: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"Validation data date range: {val_data['date'].min()} to {val_data['date'].max()}")
    print(f"Test data date range: {test_data['date'].min()} to {test_data['date'].max()}")

    return trainloader, valloader, testloader, 3 #, scaler  # Return scaler for inverse transformation if needed


# data preberation function 
def load_weather_data_transformer(filepath, batch_size, seq_length):
    # Load and preprocess the data
    data = pd.read_csv(filepath, parse_dates=['date'])
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

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length])
    X, y = np.array(X), np.array(y)

    # Calculate the indices for splitting
    total_len = len(X)
    train_size = int(total_len * 0.8)  # First 80% for training
    val_size = int(total_len * 0.1)    # Next 10% for validation
    test_size = total_len - train_size - val_size  # Remaining 10% for test

    # Split the data based on calculated sizes
    X_train, y_train = X[:train_size], y[:train_size]
    X_eval, y_eval = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    eval_dataset = torch.utils.data.TensorDataset(X_eval_tensor, y_eval_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, evalloader, testloader, scaler, features, len(features), seq_length


###########################################################################################################################
################################# other function ###########################################################################
###########################################################################################################################

def plot_prediction_vs_actual(pred, actual, labels, save_dir = 'transformer_plots' , apply_mean = False, window_size=10):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, label in enumerate(labels):
        plt.figure(figsize=(24, 7))
        string_mean = ''
        if apply_mean :
            string_mean = "_mean"
            # Smoothing (rolling mean)
            pred_values = pd.Series(pred[:, i]).rolling(window=window_size).mean()
            try:
                actual_values = actual[:, i]
            except:
                actual_values = actual[labels[i]].values

            actual_values = pd.Series(actual_values).rolling(window=window_size).mean()

        else :     
            # Use predictions and actual values directly
            pred_values = pred[:, i]
            try:
                actual_values = actual[:, i]
            except:
                actual_values = actual[labels[i]].values

        plt.plot(pred_values, label='Predicted ' + label, color='blue', linewidth=2)
        plt.plot(actual_values, label='Actual ' + label, color='orange', linestyle='dashed', linewidth=2)

        plt.xlabel('Samples')
        plt.ylabel(label)
        plt.legend()
        plt.title(f'{label} - Predicted vs Actual')
        plt.grid(True)

        # Optionally plot only a subset of data
        plt.xlim(0, len(pred_values))
        plot_filename = os.path.join(save_dir, f'{label}{string_mean}_pred_vs_actual.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free up memory


def plot_losses(loss_val=list, loss_train=list, save_dir=""):

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(loss_train) + 1)
    plt.figure(figsize=(24, 6))
    plt.plot(epochs, loss_train, 'b-', label='Training Loss')
    plot_name_str = "Training"
    if loss_val is not None and len(loss_val) > 0:
        plot_name_str += " and Validation"
        plt.plot(epochs, loss_val, 'r-', label='Validation Loss')
    plot_name_str += " Loss"
    
    # Plot formatting
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(plot_name_str)
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_filename = os.path.join(save_dir, plot_name_str)
    plt.savefig(plot_filename)
    plt.close()   


def plot_error(pred, actual, labels, save_dir ):
    subdir = os.path.join(save_dir, "error_plot")
    os.makedirs(subdir, exist_ok=True)

    for i, label in enumerate(labels):
        pred_values = pred[:, i]
        try:
            actual_values = actual[:, i]
        except:
            actual_values = actual[labels[i]].values 

        error = np.abs(actual_values[:50]- pred_values[:50])
        plt.figure(figsize=(10, 5))
        plt.plot(error, marker='o', label='Error (Actual - Predicted)', color='r')
        plt.title(f'Error in {label} Prediction')
        plt.xlabel('index')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)

        plot_filename = os.path.join(subdir, f"error_{label}.png")
        plt.savefig(plot_filename)
        plt.close()  


def plot_metrics(epochs, average_errors, detection_rates):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Detection Rate', color=color)
    ax2.plot(epochs, detection_rates, color=color, label='Detection Rate', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Average Error and Detection Rate Over Epochs')
    plt.grid()
    plt.show()


def calculate_metrics(predictions, actuals, threshold=5.0):  # Change threshold as needed
    # Calculate average error

    # Define extreme value threshold (e.g., top 90th percentile)
    extreme_threshold = np.percentile(actuals.numpy(), 90)

    # Determine detections within the similarity threshold
    detected_extremes = (predictions.numpy() > (extreme_threshold - threshold)).astype(int)
    actual_extremes = (actuals.numpy() > extreme_threshold).astype(int)

    # Calculate detection rate based on detections
    true_positives = np.sum(detected_extremes * actual_extremes)
    false_negatives = np.sum((1 - detected_extremes) * actual_extremes)
    
    if true_positives + false_negatives > 0:
        detection_rate = true_positives / (true_positives + false_negatives)
    else:
        detection_rate = 0.0

    # Calculate direction of error
    direction_of_error = torch.sign(predictions - actuals).numpy()

    return  detection_rate, direction_of_error

