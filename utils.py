
import os 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def plot_prediction_vs_actual(pred, actual, labels, save_dir = 'transformer_plots' , apply_mean = False, window_size=10):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, label in enumerate(labels):
        plt.figure(figsize=(24, 7))

        if apply_mean :
            # Smoothing (rolling mean)
            pred_values = pd.Series(pred[:, i]).rolling(window=window_size).mean()
            actual_values = pd.Series(actual[labels[i]].values).rolling(window=window_size).mean()

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
        plot_filename = os.path.join(save_dir, f'{label}_pred_vs_actual.png')
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

def plot_error_histogram(errors, save_dir=""):
    plt.figure(figsize=(12, 6))
    # Ensure errors is a 1D array
    if len(errors.shape) > 1:
        errors = errors.flatten()
    plt.hist(errors, bins=50, alpha=0.75, color='blue')  # Single color for the histogram
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plot_filename = os.path.join(save_dir, "error_disribution")
    plt.savefig(plot_filename)
    plt.close()  

