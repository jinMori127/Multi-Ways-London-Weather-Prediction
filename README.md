# Weather Prediction Using Deep Learning

## Introduction  
This project focuses on predicting weather features in London using three deep learning models:  
1. **LSTM** (Long Short-Term Memory)  
2. **Transformer**  
3. **Hybrid LSTM-Transformer**  

The aim is to forecast the weather for Day 7 based on data from the previous six days.

![image](https://github.com/user-attachments/assets/37bc4ac0-1ffe-4ccd-87c1-0058385ae89c)

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

**LSTM Network Description:**

Lstm layer in this we set the batch_first=True  
We also used the attention mechanism size and at the end we apply softmax to the output of it, also we apply dropout to avoid overfitting.  
Fully connected layer takes input and return output with the wanted size.  

**Parameters:**  
- batch_size = 128  
- hidden_dim = 512  
- output_dim = 9  
- lstm_num_layers = 3  
- dropout = 0.05  
We used Adam optimizer with learning rate=0.0001.

**Loss Function:**  
  We use nn.L1Loss()
  
**Note:** We figure out after a lot of tries giving a smaller learning rate enable the network to predict better.  
- num_epochs = 100  

### 2. Transformer  
The Transformer model, which utilizes self-attention mechanisms, excels at parallel processing and handling long-range dependencies. It is known for its smaller error range compared to RNN-based models.

**Transformer Network Description:** 
We define `TransformerEncoderLayer`, which is the fundamental building block of the transformer architecture. In this, we set `batch_first=True` and `dropout=0.25`. The `TransformerEncoder` stacks multiple encoder layers together, and in this case, we use 2 layers. A fully connected layer takes input of size `hidden_dim` and returns the desired output.  

**Parameters:**  
- Batch_size = 64  
- hidden_dim = 80  
- num_heads = 10  

We used Adam optimizer with `learning_rate=0.001`.  
- Num_epochs = 40  

### 3. Hybrid LSTM-Transformer  
This model combines the strengths of both LSTM and Transformer. It integrates the sequential processing power of LSTM with the attention-based architecture of the Transformer to improve prediction accuracy.

**Hybrid LSTM-Transformer:** 
  At this model we combined the two articture from the lstm that we built and the transformer.
  
**Parameters:**
  - lr=0.00001 
  - num_epochs = 100
  - The other parameter is the same from the lstm and transformer we used before.

**Loss Function:**
    The loss function used nn.MSELoss()
    

## Dataset Preparation/Description:
We took the London Weather Data from: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data

The dataset consists of weather data from London over a period of time. For each day, the following features are recorded:  
- **Cloud Cover**  
- **Sunshine**  
- **Global Radiation**  
- **Maximum Temperature**  
- **Mean Temperature**  
- **Minimum Temperature**  
- **Pressure**  

**Dataset Preparation :** We used a specific approach for data preparation to handle missing values effectively:  
- For each empty cell in the **cloud_cover** column, we filled it with the most frequent value of that column.  
- For empty cells in the **global_radiation** or **mean_temp** columns, we filled them with the mean value of their respective columns.  
- If any column was entirely missing, it was dropped from the dataset.  
This preprocessing ensured the dataset was complete and ready for use in model training and evaluation.
- After handling missing values, the data was scaled using StandardScaler to standardize the features.
This scaling ensures that all features have a mean of 0 and a standard deviation of 1, which improves the performance of the models during training.

**Data division:** train: 80%, validation: 10%, test: 10%

## Evaluation  
We evaluate the models based on their prediction accuracy and error ranges. The mean absolute error (MAE) is used to calculate the difference between the predicted and actual values for each feature. The models are compared in terms of the number of correct predictions and the size of the error ranges.

## Results  

### LSTM Network results:
![image](https://github.com/user-attachments/assets/82c53ee3-e433-430a-86a4-b68cdd77de06)

#### Prediction Visualization  
We evaluate the model's performance on the test set by visualizing predictions for several outputs.  
In each plot:  
- Points represent the **ground truth** and **predicted values**. The closer these points are to each other, the better the results indicate model performance.  
- The **blue line** represents the **predicted values**, while the **orange line** represents the **ground truth values**.  

These visualizations help to clearly assess how well the model captures the underlying patterns in the data.

![image](https://github.com/user-attachments/assets/b58218e0-7eae-4e18-932b-a7e2c1af8010)

![image](https://github.com/user-attachments/assets/3f6e30db-161f-44c1-9b6a-bfd345f010be)

![image](https://github.com/user-attachments/assets/5f37a07e-d8e4-4166-8d98-b35331750eba)

There is more plots for full look at the results of the other feature you can take a look at the lstm_attention_plots folder.
At this folder there is also the same plots but we done mean to the to make the plot smoother (plots names: <feature_name>_mean_xx )

#### Error Visualization 
  For more clarification we will show some plots for the error we get actual vs predicted  
  By plotting the error of 50 samples of the predicted and then: 
  Error = np.abs(predict-acual) so we can see how much accurate is our network: 

  ![image](https://github.com/user-attachments/assets/9a572ec5-df93-482e-b8d7-991648909c0f)

  As we can see, we were able to predict the cloud cover within a range of 0 to 4 of the 
  actual cover. 


  ![image](https://github.com/user-attachments/assets/1f85e670-ea87-4bbe-9cf7-4048835c989b)
  
  As we can see, we were able to predict the global radiation within a range of 0 to 100 of the 
  actual. 


  ![image](https://github.com/user-attachments/assets/f25a6f18-8a22-40bd-b660-af24b2bff1ff)

  As we can see, we were able to predict the maximum temperature within a range of 0 to 
  5 degrees of the actual temperature. 


-----------------------------------------------------------------------------------------------------------------------------------------

### Transformer Network results:
![image](https://github.com/user-attachments/assets/ac0299a1-3bcd-4e71-a7a4-5f01cc3a5ee1)

#### Prediction Visualization  

  ![image](https://github.com/user-attachments/assets/b53295ea-94b7-4deb-b0ff-fd3b189fb293)
  
  ![image](https://github.com/user-attachments/assets/903e1c79-e8ae-4620-82b2-fdfbda7976ad)
  
  ![image](https://github.com/user-attachments/assets/76ce2216-ac6c-48d5-b705-ceb9739f6dfb)
  
  There is more plots for full look at the results of the other feature you can take a look at the transformer_plots folder.
  At this folder there is also the same plots but we done mean to the to make the plot smoother (plots names: <feature_name>_mean_xx )


#### Error Visualization 

  ![image](https://github.com/user-attachments/assets/97db7bbd-fc7f-495b-8aee-68359ac3fd06)
  
  As we can see, we were able to predict the cloud cover within a range of 0 to 2 of the 
  actual cover. 

  ![image](https://github.com/user-attachments/assets/95c75783-50c3-4882-8455-7f15a90d62d1)
  
  As we can see, we were able to predict the global radiation within a range of 0 to 1 of the 
  actual (as we can see there is a pretty good improvement here from the lstm). 


  ![image](https://github.com/user-attachments/assets/b2b13b23-72c7-44f4-aaa6-345785d46ce9)
  
  As we can see, we were able to predict the maximum temperature within a range of 0 to 
  1 degrees of the actual temperature. 


  to take a full look at the error plots visit the folder transformer_plots/error_plot.


-----------------------------------------------------------------------------------------------------------------------------------------


### Lstm with Transformer Network results:

![image](https://github.com/user-attachments/assets/34575267-b457-4d2f-a2c3-5aeca8df07b0)

#### Prediction Visualization  

  ![image](https://github.com/user-attachments/assets/cbec83df-f32f-4b52-a1da-ce6e4dc94392)
  
  ![image](https://github.com/user-attachments/assets/50332aba-b401-4aa5-8ce1-3b980e0c3945)
  
  ![image](https://github.com/user-attachments/assets/f1be57ea-4bdf-458d-9d6f-dfdbb37281fc)

  There is more plots for full look at the results of the other feature you can take a look at the lstm_with_transformer_plots folder.
  At this folder there is also the same plots but we done mean to the to make the plot smoother (plots names: <feature_name>_mean_xx )


#### Error Visualization 

  ![image](https://github.com/user-attachments/assets/f9a1aee7-673b-4218-8540-b669756458c4)

  As we can see, we were able to predict the cloud cover within a range of 0 to 2 of the 
  actual cover. 

  ![image](https://github.com/user-attachments/assets/e13b21bc-bac5-4e23-a2b2-40f3496e49a4)

  As we can see, we were able to predict the global radiation within a range of 0 to 1 of the 
  actual (as we can see there is a pretty good improvement here from the lstm). 

  ![image](https://github.com/user-attachments/assets/707b6327-1cd5-4ee7-8d63-b12ddfed0ad6)
  As we can see, we were able to predict the maximum temperature within a range of 0 to 
  0.8 degrees of the actual temperature (which is a improvment from the past two models). 

  to take a full look at the error plots visit the folder lstm_with_transformer_plots/error_plot.


## Acknowledgments  
This project was developed as part of the BSc Computer Science program at the University of Haifa. Special thanks to the Department of Computer Science for their support, especially Dr. Dan Rosenbaum.


## How to Run  
### 1. Clone the Repository  
```bash
git clone https://github.com/your-username/project-name.git
cd project-name




