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

class Weather_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5, bidirectional=False):
        super(Weather_network, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=bidirectional)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(output_dim)

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
        out = self.fc(attn_output)
        out = self.batch_norm(out)
        return out


class CustomLoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss()):
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, outputs, targets):
        base_loss_value = self.base_loss(outputs, targets)

        loss = base_loss_value
        return loss


def test_model(testloader, model, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    test_loss = test_loss / len(testloader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    return torch.cat(all_outputs), torch.cat(all_labels)

batch_size = 128
trainloader,evalloader, testloader, scaler, features, input_dim, seq_length = load_weather_data_transformer('london_weather.csv', batch_size, seq_length=7)
hidden_dim = 512
num_layers = 3
dropout = 0.25
print(input_dim)

def train(input_dim, hidden_dim, output_dim, num_layers, dropout, trainloader, valloader, testloader):
    weather_net = Weather_network(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
                                  num_layers=num_layers, dropout=dropout, bidirectional=True)
    weather_net.to(device)

    criterion = CustomLoss()
    optimizer = optim.Adam(weather_net.parameters(), lr=0.0001)

    num_epochs = 50
    train_losses = []
    valid_losses = []
    detection_rates = []

    for epoch in range(num_epochs):
        weather_net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = weather_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)

        weather_net.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = weather_net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        val_loss = val_loss / len(valloader.dataset)
        valid_losses.append(val_loss)

        # Calculate metrics
        predictions = torch.cat(all_outputs)
        actuals = torch.cat(all_labels)


        # Inverse transform predictions and actuals
        predictions_np = predictions.numpy()
        actuals_np = actuals.numpy()

        predictions_inv = scaler.inverse_transform(predictions_np)
        actuals_inv = scaler.inverse_transform(actuals_np)

        # Convert the inverse-transformed data back to PyTorch tensors
        predictions_inv_tensor = torch.tensor(predictions_inv)
        actuals_inv_tensor = torch.tensor(actuals_inv)

        detection_rate, direction_of_error = calculate_metrics(predictions_inv_tensor, actuals_inv_tensor)
        detection_rates.append(detection_rate)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Detection Rate of Extremes: {detection_rate:.4f}")

    print("Training finished.")
    plot_losses(loss_val=valid_losses, loss_train=train_losses, save_dir="lstm_attention_plots")

    torch.save(weather_net.state_dict(), 'weather_net.pth')
    predictions, actuals = test_model(testloader, weather_net, criterion, device)

    # # Reverse scaling
    # predictions_inv = scaler.inverse_transform(predictions)
    # actuals_inv = scaler.inverse_transform(actuals)
    # plot_metrics(num_epochs, 2, detection_rates)
    # Reverse scaling
    predictions_inv = scaler.inverse_transform(predictions)
    actuals_inv = scaler.inverse_transform(actuals)

    label_columns = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']

    # plot_relative_errors(predictions.numpy(), actuals.numpy())
    plot_prediction_vs_actual(predictions_inv, actuals_inv, label_columns, save_dir='lstm_attention_plots')
    plot_prediction_vs_actual(predictions_inv, actuals_inv, label_columns, save_dir='lstm_attention_plots', apply_mean=True)
    plot_error(predictions_inv, actuals_inv, label_columns, save_dir='lstm_attention_plots')

train(input_dim, hidden_dim, input_dim, num_layers, dropout, trainloader, evalloader, testloader)

# uncommit for test
# weather_net = Weather_network(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout)
# weather_net.load_state_dict(torch.load('weather_net.pth'))
# criterion = CustomLoss()
# predictions, actuals = test_model(testloader, weather_net, criterion, device)
# label_columns = ['cloud_cover','sunshine','global_radiation','max_temp','mean_temp','min_temp','precipitation','pressure', 'snow_depth']
# # plot_prediction_vs_actual(predictions, actuals, label_columns, save_dir='lstm_attention_plots', apply_mean=True)
# plot_error(predictions, actuals,label_columns, save_dir='lstm_attention_plots')