# Weather Prediction Using Deep Learning

## Introduction  
This project focuses on predicting weather features in London using three deep learning models:  
1. **LSTM** (Long Short-Term Memory)  
2. **Transformer**  
3. **Hybrid LSTM-Transformer**  

The aim is to forecast the weather for Day 7 based on data from the previous six days.

## Problem Description  
We use the following weather features over six days to make predictions:  
- **Cloud Cover**  
- **Sunshine**  
- **Global Radiation**  
- **Maximum Temperature**  
- **Mean Temperature**  
- **Minimum Temperature**  
- **Pressure**  

The task is to predict these features for **Day 7** using the data from the previous six days.

## Models  
### 1. LSTM (Long Short-Term Memory)  
LSTM is a type of Recurrent Neural Network (RNN) that is effective in learning from sequential data. We use this model to capture temporal dependencies in weather data and make predictions for the 7th day.

### 2. Transformer  
The Transformer model, which utilizes self-attention mechanisms, excels at parallel processing and handling long-range dependencies. It is known for its smaller error range compared to RNN-based models.

### 3. Hybrid LSTM-Transformer  
This model combines the strengths of both LSTM and Transformer. It integrates the sequential processing power of LSTM with the attention-based architecture of the Transformer to improve prediction accuracy.

## Dataset  
The dataset consists of weather data from London over a period of time. For each day, the following features are recorded:  
- **Cloud Cover**  
- **Sunshine**  
- **Global Radiation**  
- **Maximum Temperature**  
- **Mean Temperature**  
- **Minimum Temperature**  
- **Pressure**  

The goal is to use the data from the past 6 days to predict these features for Day 7.

## Evaluation  
We evaluate the models based on their prediction accuracy and error ranges. The mean absolute error (MAE) is used to calculate the difference between the predicted and actual values for each feature. The models are compared in terms of the number of correct predictions and the size of the error ranges.

## Results  


### Visualizations  


## Acknowledgments  
This project was developed as part of the BSc Computer Science program at the University of Haifa. Special thanks to the Department of Computer Science for their support, especially Dr. Dan Rosenbaum.


## How to Run  
### 1. Clone the Repository  
```bash
git clone https://github.com/your-username/project-name.git
cd project-name




