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




def plot_prediction_vs_actual_new(first_one, second,
                              actuals_inv, labels, save_dir='transformer_plots', 
                              apply_mean=False, window_size=10):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, label in enumerate(labels):
        plt.figure(figsize=(24, 7))
        string_mean = ''
        
        # Apply rolling mean smoothing if requested
        if apply_mean:
            string_mean = "_mean"
            # Apply rolling mean to each type of predictions and actuals
            lstm_values = pd.Series(first_one[:, i]).rolling(window=window_size).mean()
            transformer_values = pd.Series(second[:, i]).rolling(window=window_size).mean()
            actual_values = pd.Series(actuals_inv[:, i]).rolling(window=window_size).mean()
        else:
            # Use predictions and actual values directly
            lstm_values = first_one[:, i]
            transformer_values = second[:, i]
            actual_values = actuals_inv[:, i]

        # Plot each set of predictions along with actual values
        plt.plot(lstm_values, label='LSTM Predicted ' + label, color='blue', linewidth=2)
        plt.plot(transformer_values, label='Transformer Predicted ' + label, color='green', linewidth=2)
        plt.plot(actual_values, label='Actual ' + label, color='orange', linestyle='dashed', linewidth=2)

        plt.xlabel('Samples')
        plt.ylabel(label)
        plt.legend()
        plt.title(f'{label} - Predicted vs Actual')
        plt.grid(True)

        # Set plot limits and save
        plt.xlim(0, len(actual_values))
        plot_filename = os.path.join(save_dir, f'{label}{string_mean}_pred_vs_actual.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free up memory



def plot_prediction_vs_actual_means(first_one, second, third,
                                     actuals_inv, labels, save_dir='transformer_plots'):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate the means of the differences for each label
    means_first = []
    means_second = []
    means_third = []

    for i, label in enumerate(labels):
        # Calculate the differences: first_one - actual, second - actual, third - actual
        first_diff = abs(first_one[:, i] - actuals_inv[:, i])
        second_diff = abs(second[:, i] - actuals_inv[:, i])
        third_diff = abs(third[:, i] - actuals_inv[:, i])
        
        # Calculate the mean of the differences
        means_first.append(np.mean(first_diff))
        means_second.append(np.mean(second_diff))
        means_third.append(np.mean(third_diff))

    # Create a bar plot for the means
    x = np.arange(len(labels))  # label locations
    width = 0.25  # width of the bars

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width, means_first, width, label='lstm  Mean Difference', color='blue')
    bars2 = plt.bar(x, means_second, width, label='transformer Mean Difference', color='red')
    bars3 = plt.bar(x + width, means_third, width, label='both Prediction Mean Difference', color='green')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    plt.xlabel('Labels')
    plt.ylabel('Mean Difference')
    plt.title('Mean Differences from Actuals for Predictions')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y')

    # Save the histogram
    plot_filename = os.path.join(save_dir, 'mean_differences_histogram.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free up memory

    for i, label in enumerate(labels):
        plt.figure(figsize=(8, 6))
        
        # Create a single bar plot for the current label
        plt.bar(['lstm Prediction', 'transformer Prediction', 'both Prediction'], 
                [means_first[i], means_second[i], means_third[i]], 
                color=['blue', 'red', 'green'])

        # Add labels and title for the individual plot
        plt.xlabel('Predictions')
        plt.ylabel('Mean Difference')
        plt.title(f'Mean Differences for {label}')
        plt.grid(axis='y')

        # Save the individual plot for the current label
        individual_plot_filename = os.path.join(save_dir, f'mean_difference_{label}.png')
        plt.savefig(individual_plot_filename)
        plt.close()  # Close the figure to free up memory



lstm_predictions_inv = torch.load('lstm_predictions_inv.pt')
transformer_predictions_inv = torch.load('transformer_predictions_inv.pt')
both_predictions_inv = torch.load('lstmwithtrans_predictions_inv.pt')

actuals_inv = torch.load('actuals_inv.pt')
label_columns = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']
plot_prediction_vs_actual_new(lstm_predictions_inv, transformer_predictions_inv,actuals_inv, 
                          label_columns, save_dir="1_2_plots", apply_mean=True)

plot_prediction_vs_actual_new(lstm_predictions_inv, both_predictions_inv,actuals_inv, 
                          label_columns, save_dir="1_3_plots", apply_mean=True)

plot_prediction_vs_actual_new(transformer_predictions_inv, both_predictions_inv ,actuals_inv, 
                          label_columns, save_dir="2_3_plots", apply_mean=True)


plot_prediction_vs_actual_means(lstm_predictions_inv, transformer_predictions_inv,both_predictions_inv, actuals_inv, 
                          labels=label_columns, save_dir="1_2_3_plots/no_mean/diff")



# plot_prediction_vs_actual_new_111(lstm_predictions_inv, transformer_predictions_inv,actuals_inv, 
#                           label_columns, save_dir="1_2_plots/mean/diff", apply_mean=True)

# plot_prediction_vs_actual_new_111(lstm_predictions_inv, both_predictions_inv,actuals_inv, 
#                           label_columns, save_dir="1_3_plots/mean/diff", apply_mean=True)

# plot_prediction_vs_actual_new_111(transformer_predictions_inv, both_predictions_inv ,actuals_inv, 
#                           label_columns, save_dir="2_3_plots/mean/diff", apply_mean=True)
