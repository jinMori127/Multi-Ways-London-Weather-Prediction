import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *

# Load data


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, num_heads=9, num_layers=2, hidden_dim=72):
        super(TransformerRegressor, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout= 0.25)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # Apply linear transformation to input
        x = x.permute(1, 0, 2)  # PyTorch Transformer expects (seq_len, batch, hidden_dim)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # Take the output of the last time step
        x = self.fc(x)
        return x


def train_transform(trainloader,evalloader, testloader, scaler, features, input_dim, seq_length , device):
    # Define and train the model
    list_train_loss = []
    list_eval_loss = []
    detection_rates = []


    model = TransformerRegressor(input_dim=input_dim, output_dim=len(features), seq_length=seq_length, num_heads=10, hidden_dim=80).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 40
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

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {eval_loss:.4f}, error :{detection_rate:.4f}")

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

    torch.save(predictions_inv, 'transformer_predictions_inv.pt')

    # Plot predictions vs actual values
    torch.save(model.state_dict(), 'weather_transformer_net.pth')
    epochs = list(range(1, num_epochs + 1))
    plot_metrics(epochs, 2, detection_rates)

    plot_prediction_vs_actual(predictions_inv, actuals_inv, features, save_dir='transformer_plots')
    plot_prediction_vs_actual(predictions_inv, actuals_inv, features, save_dir='transformer_plots', apply_mean=True)

    plot_losses(loss_val=list_eval_loss, loss_train=list_train_loss, save_dir='transformer_plots')
    plot_error(predictions, actuals, labels=['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure'], save_dir='transformer_plots')

batch_size = 64
seq_length = 7

trainloader, evalloader, testloader, scaler, features, input_dim, seq_length = load_weather_data_transformer('london_weather.csv', batch_size, seq_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform(trainloader,evalloader, testloader, scaler, features, input_dim, seq_length, device=device)

print("done training")