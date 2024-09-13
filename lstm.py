import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import *


# Define general variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"You are using: {device}")

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
    full_data = full_data.dropna()
    train_data, remaining_data = train_test_split(full_data, test_size=0.2, shuffle=True)
    val_data, test_data = train_test_split(remaining_data, test_size=0.5, shuffle=True)
    label_columns = ['cloud_cover','sunshine','global_radiation','max_temp','mean_temp','min_temp','precipitation','pressure', 'snow_depth']

    trainset = WeatherDataset(data=train_data, label_columns=label_columns)
    valset = WeatherDataset(data=val_data, label_columns=label_columns)
    testset = WeatherDataset(data=test_data, label_columns=label_columns)
    input_dim = trainset[0][0].shape[0]

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return trainloader, valloader, testloader, input_dim

class Weather_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(Weather_network, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)  
        attention_weights = self.softmax(attention_weights) 
        attention_weights = attention_weights.permute(0, 2, 1) 
        weighted_output = torch.bmm(attention_weights, lstm_output) 
        return weighted_output.squeeze(1)  

    def forward(self, x):
        lstm_output, (hn, cn) = self.lstm(x)  
        attn_output = self.attention_net(lstm_output)  
        attn_output = self.dropout(attn_output)
        attn_output = self.batch_norm(attn_output)
        out = self.fc(attn_output)  
        return out

class CustomLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss()):
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, outputs, targets):
        base_loss_value = self.base_loss(outputs, targets)
        mse_loss_snow_depth = F.mse_loss(outputs[:, 8], targets[:, 8]) * 100
        mse_loss_sun_shine = F.mse_loss(outputs[:, 1], targets[:, 1])
        mse_loss_preseption = F.mse_loss(outputs[:, 6], targets[:, 6])
        mse_loss_preseption += F.mse_loss(outputs[:, 0], targets[:, 0]) * 20
        loss = base_loss_value + mse_loss_snow_depth + mse_loss_sun_shine + mse_loss_preseption
        return loss



def plot_relative_errors(predictions, actuals):
    relative_errors = np.abs((actuals - predictions) / actuals)
    plt.figure(figsize=(12, 6))
    plt.plot(relative_errors, color='blue', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Relative Error')
    plt.title('Relative Error Plot')
    plt.grid(True)
    plt.show()


def test_model(testloader, model, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    test_loss = test_loss / len(testloader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    return torch.cat(all_outputs), torch.cat(all_labels)

batch_size = 128
trainloader, valloader, testloader, input_dim = load_weather_data(batch_size)

hidden_dim = 1024
output_dim = 9  
num_layers = 4
dropout = 0.05
print(input_dim)

def train(input_dim, hidden_dim, output_dim, num_layers, dropout, trainloader, valloader, testloader):
    weather_net = Weather_network(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout)
    weather_net.to(device)

    criterion = CustomLoss()
    optimizer = optim.Adam(weather_net.parameters(), lr=0.0003)

    num_epochs = 1000
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        weather_net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = weather_net(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)

        weather_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = weather_net(inputs.unsqueeze(1))
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(valloader.dataset)
        valid_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    print("Training finished.")
    plot_losses(loss_val=valid_losses, loss_train=train_losses, save_dir="lstm_attention_plots")

    torch.save(weather_net.state_dict(), 'weather_net.pth')
    predictions, actuals = test_model(testloader, weather_net, criterion, device)

    label_columns = ['cloud_cover','sunshine','global_radiation','max_temp','mean_temp','min_temp','precipitation','pressure', 'snow_depth']
    
    errors = actuals - predictions
    plot_relative_errors(predictions.numpy(), actuals.numpy())
    plot_prediction_vs_actual(predictions, actuals, label_columns, save_dir='lstm_attention_plots')
    plot_prediction_vs_actual(predictions, actuals, label_columns, save_dir='lstm_attention_plots', apply_mean=True)

train(input_dim, hidden_dim, output_dim, num_layers, dropout, trainloader, valloader, testloader)

# uncommit for test
# weather_net = Weather_network(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout)
# weather_net.load_state_dict(torch.load('weather_net.pth'))
# criterion = CustomLoss()
# predictions, actuals = test_model(testloader, weather_net, criterion, device)
# label_columns = ['cloud_cover','sunshine','global_radiation','max_temp','mean_temp','min_temp','precipitation','pressure', 'snow_depth']
# # plot_prediction_vs_actual(predictions, actuals, label_columns, save_dir='lstm_attention_plots', apply_mean=True)
# plot_error(predictions, actuals,label_columns, save_dir='lstm_attention_plots')