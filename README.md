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
We used Adam optimizer with learning rate=0.00001.

**Loss Function:**  
  We use nn.MSE()
  
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

We used Adam optimizer with `learning_rate=0.00001`.  
- Num_epochs = 100

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

## Results  

### LSTM Network results:
![image](https://github.com/user-attachments/assets/3f6f3d59-be4c-4886-b09e-32dfa2365d84)

#### Prediction Visualization  
We evaluate the model's performance on the test set by visualizing predictions for several outputs.  
In each plot:  
- Points represent the **ground truth** and **predicted values**. The closer these points are to each other, the better the results indicate model performance.  
- The **blue line** represents the **predicted values**, while the **orange line** represents the **ground truth values**.  

These visualizations help to clearly assess how well the model captures the underlying patterns in the data.

![image](https://github.com/user-attachments/assets/06fecfba-de13-4608-ba06-5f464ff9a73c)

![image](https://github.com/user-attachments/assets/4d74d6d8-b5a8-47cf-a4f9-cbc1cce4fa6f)

![image](https://github.com/user-attachments/assets/ffd0fab7-0d60-4fda-8c54-726fd5429c24)

There is more plots for full look at the results of the other feature you can take a look at the [LSTM Attention Plots](./lstm_attention_plots/)
 folder.
At this folder there is also the same plots but we done mean to the to make the plot smoother (plots names: <feature_name>_mean_xx )





#### Error Visualization 
  For more clarification we will show some plots for the error we get actual vs predicted  
  By plotting the error of 50 samples of the predicted and then: 
  Error = np.abs(predict-acual) so we can see how much accurate is our network: 

  ![image](https://github.com/user-attachments/assets/61155747-62ca-452d-b0c9-ba290f1c2461)

  As we can see, we were able to predict the cloud cover within a range of 0 to 4 of the 
  actual cover. 


  ![image](https://github.com/user-attachments/assets/61d58c49-782f-4434-a097-9478fac4c107)
  
  As we can see, we were able to predict the global radiation within a range of 0 to 100 of the 
  actual. 


  ![image](https://github.com/user-attachments/assets/9c71027a-3107-4608-b2d0-7ece769974c4)

  As we can see, we were able to predict the maximum temperature within a range of 0 to 
  5 degrees of the actual temperature. 


-----------------------------------------------------------------------------------------------------------------------------------------

### Transformer Network results:
![image](https://github.com/user-attachments/assets/88a3095f-a268-47e6-b1d2-16dc58274019)

#### Prediction Visualization  

  ![image](https://github.com/user-attachments/assets/ba0b6e55-edda-4b44-8f5e-4977e06d1f60)
  
  ![image](https://github.com/user-attachments/assets/5f312022-bd60-4c3a-a6d4-9dfcbe08542f)
  
  ![image](https://github.com/user-attachments/assets/4aa82128-b96e-483a-9ca7-5efe2d53b332)
  
  There is more plots for full look at the results of the other feature you can take a look at the [Transformer Plots](./transformer_plots/)
 folder.
  At this folder there is also the same plots but we done mean to the to make the plot smoother (plots names: <feature_name>_mean_xx )


#### Error Visualization 

  ![image](https://github.com/user-attachments/assets/d7196dd1-81dc-461d-8ad7-e2ce27e647c8)
  
  As we can see, we were able to predict the cloud cover within a range of 0 to 2 of the 
  actual cover. 

  ![image](https://github.com/user-attachments/assets/c3ae2dde-34e7-4b51-82e3-7da04af8ba7f)
  
  As we can see, we were able to predict the global radiation within a range of 0 to 1 of the 
  actual (as we can see there is a pretty good improvement here from the lstm). 


  ![image](https://github.com/user-attachments/assets/03d4d0de-68b6-443e-8be7-bd0a73f740af)
  
  As we can see, we were able to predict the maximum temperature within a range of 0 to 
  1 degrees of the actual temperature. 


  to take a full look at the error plots visit the folder transformer_plots/error_plot.


-----------------------------------------------------------------------------------------------------------------------------------------


### Lstm with Transformer Network results:

![image](https://github.com/user-attachments/assets/34575267-b457-4d2f-a2c3-5aeca8df07b0)

#### Prediction Visualization  

  ![image](https://github.com/user-attachments/assets/0cfdef44-6694-4f06-aa70-78544b515fd2)
  
  ![image](https://github.com/user-attachments/assets/3f0b4cb2-7232-47f6-9164-ea5a123e60d9)
  
  ![image](https://github.com/user-attachments/assets/2893de03-2145-427a-8a8c-fa77bd071b6d)

  There is more plots for full look at the results of the other feature you can take a look at the [LSTM with Transformer Plots](./lstm_with_transformer_plots/)
 folder.
  At this folder there is also the same plots but we done mean to the to make the plot smoother (plots names: <feature_name>_mean_xx )


#### Error Visualization 

  ![image](https://github.com/user-attachments/assets/c2679308-d446-4b4f-b99f-0fc113d089bb)

  As we can see, we were able to predict the cloud cover within a range of 0 to 2 of the 
  actual cover. 

  ![image](https://github.com/user-attachments/assets/b56e542f-6278-46b3-8860-eabb1a91cbe0)

  As we can see, we were able to predict the global radiation within a range of 0 to 1 of the 
  actual (as we can see there is a pretty good improvement here from the lstm). 

  ![image](https://github.com/user-attachments/assets/38158fc8-7b42-458b-985e-0d356306be1b)
  As we can see, we were able to predict the maximum temperature within a range of 0 to 
  0.8 degrees of the actual temperature (which is a improvment from the past two models). 

  to take a full look at the error plots visit the folder lstm_with_transformer_plots/error_plot.


## Models comparison 
  After long looking on the comparison ways we find out that the best thing to do is:
  to calculate the mean of the error for each way and put them in histogram 
  We did that by mean(abs(pred – actual )) And we get this results :
  
  ### cloud cover: 
  ![image](https://github.com/user-attachments/assets/19cfcb8a-c4fb-4dae-8ca9-325a858fd999)

  In the cloud cover we got that the third way (lstm with transformer) is the best because it has the smallest error mean 1.58.

  ### global radiation:
  ![image](https://github.com/user-attachments/assets/b9446c15-b0ff-47dd-b720-4bc24e54c26e)

  we got also here advantage for the third way.

  ### max temp:
  ![image](https://github.com/user-attachments/assets/d7a3c890-c62a-4405-9113-2567d7f47808)

  In this case, the third way is worse by a slight difference.

  In the other plots we get that the third way is the best u can take a look for the full results on 
  1_2_3_plots\no_mean\diff folder
  

  #### Here is a plot for all of the features:

  ![image](https://github.com/user-attachments/assets/2bba376f-52de-4b11-99f9-ff8c56965b77)

  
## Acknowledgments  
This project was developed as part of the BSc Computer Science program at the University of Haifa. Special thanks to the Department of Computer Science for their support, especially Dr. Dan Rosenbaum.


## How to Run  
### 1. Clone the Repository  
```bash
git clone https://github.com/your-username/project-name.git
cd project-name




