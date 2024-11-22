import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


# Define Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, num_heads, transformer_hidden_dim):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.transformer_embedding = nn.Linear(hidden_dim, transformer_hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(transformer_hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)  
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  

        transformer_input = self.transformer_embedding(lstm_out)
        transformer_input = transformer_input.permute(1, 0, 2)  
        transformer_out = self.transformer_encoder(transformer_input)
        out = self.fc(transformer_out[-1, :, :])  
        return out

# Define loss function
class CustomLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss()):
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, outputs, targets):
        base_loss_value = self.base_loss(outputs, targets)
        return base_loss_value

# Define training function for the hybrid model
def train_hybrid_model(trainloader,evalloader, testloader, input_dim, hidden_dim, output_dim, num_layers, dropout, seq_length, num_heads, transformer_hidden_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"You are using: {device}")
    list_train_loss = []
    list_eval_loss = []
    detection_rates = []

    # Initialize model
    model = HybridModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout, num_heads=num_heads, transformer_hidden_dim=transformer_hidden_dim).to(device)

    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()

        # Training
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader.dataset)
        list_train_loss.append(epoch_loss)

        # Evaluation
        model.eval()

        with torch.no_grad():
            test_loss = 0.0
            all_outputs = []
            all_labels = []
            for inputs, labels in evalloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())


            eval_loss = test_loss / len(evalloader.dataset)
            list_eval_loss.append(eval_loss)

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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {eval_loss:.4f} Detection Rate of Extremes: {detection_rate:.4f}")


    # test model
    model.eval()
    with torch.no_grad():
        predictions, actuals = [], []
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions.append(outputs.cpu())
            actuals.append(labels.cpu())
    print("test loss: ", loss.item())
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()

    # Reverse scaling
    predictions_inv = scaler.inverse_transform(predictions)
    actuals_inv = scaler.inverse_transform(actuals)

    torch.save(predictions_inv, 'lstmwithtrans_predictions_inv.pt')

    torch.save(model.state_dict(), "weather_lstm_with_transformer.pth")
    epochs = list(range(1, num_epochs + 1))
    plot_metrics(epochs, 2, detection_rates)
    
    plot_prediction_vs_actual(predictions_inv, actuals_inv, features, save_dir="lstm_with_transformer_plots")
    plot_prediction_vs_actual(predictions_inv, actuals_inv, features, save_dir="lstm_with_transformer_plots", apply_mean=True)
    plot_losses(loss_val=list_eval_loss, loss_train=list_train_loss, save_dir="lstm_with_transformer_plots")

    label_columns =['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']
    plot_error(predictions, actuals,label_columns, save_dir='lstm_with_transformer_plots')


# Parameters
batch_size = 64
seq_length = 7
hidden_dim = 512
num_layers = 3
dropout = 0.2
num_heads = 10
transformer_hidden_dim = 80

trainloader,evalloader, testloader, scaler, features, input_dim, seq_length = load_weather_data_transformer('london_weather.csv', batch_size, seq_length)
output_dim = len(features)
train_hybrid_model(trainloader,evalloader, testloader, input_dim, hidden_dim, output_dim, num_layers, dropout, seq_length, num_heads, transformer_hidden_dim)
