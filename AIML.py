import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os
import yfinance as yf
from datetime import date
def aiml(ticker,data):
    d=pd.DataFrame(data) # creates a data set using pandas
    d.reset_index(inplace=True) # Reset the index of the DataFrame
    data = d.filter(["Close"]) # Filter the DataFrame to keep only the "Close" column
    final_value=data['Close'][len(data)-1] # Extract the last value from the "Close" column
    dataset = data.values  # Convert the DataFrame to a NumPy array
    training_data_len = math.ceil(len(dataset) * .8) # Calculate the length of the training data (80% of the dataset)
    scaler = MinMaxScaler(feature_range=(0,1)) # Initialize the MinMaxScaler with the feature range (0, 1)
    scaler_data = scaler.fit_transform(dataset) # Fit the scaler to the data and transform the dataset

    length_scaler=len(scaler_data) # Finds the length of the scaler_data set
    train_data = scaler_data[0:training_data_len,:] # Create the training dataset

    # Initialize the feature set and target set
    x_train = []
    y_train = []
    # Populate the feature set and target set
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0]) # Append the previous 60 data points to x_train
        y_train.append(train_data[i,0]) # Append the current data point to y_train 
    x_train, y_train = np.array(x_train), np.array(y_train) # Convert the lists to NumPy arrays

    # Reshape x_train to the format expected by LSTM [samples, time steps, features]
    x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))  # Add dropout with 20% dropout rate
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error') # Compile the model
    model.fit(x_train,y_train,batch_size = 32,epochs = 10) # Train the model
    test_data = scaler_data[training_data_len - 60:] # Prepare the test data
    x_test=[] # Initialize the test feature set
    y_test = dataset[training_data_len:, :] # Set the test target set to the actual values from the original dataset
    
    # Populate the test feature set
    for i in range(60,len(test_data)): 
        x_test.append(test_data[i-60:i,0]) # Append the previous 60 data points to x_test

    x_test = np.array(x_test) # Convert the list to a NumPy array

    # Reshape x_test to the format expected by LSTM [samples, time steps, features]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction = model.predict(x_test) # Make predictions on the test data
    # Inverse transform the predictions to the original scale
    prediction = scaler.inverse_transform(prediction) 
    mse = np.mean((prediction- y_test)**2) # Calculate the Mean Squared Error (MSE)
    rmse=np.sqrt(mse) # Calculate the Root Mean Squared Error (RMSE)
    print("RMSE :",rmse) 
    print("MSE :",mse)
    
    # Define the accuracy_percentage function
    def accuracy_percentage(y_true, y_pred, threshold):
        correct_predictions = np.sum(np.abs(y_true - y_pred) / y_true <= threshold)
        total_predictions = len(y_true)
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy
    threshold = 0.05  # Example threshold (5%)
    accuracy = accuracy_percentage(y_test, prediction, threshold)
    print("Accuracy Percentage:", accuracy, "%")

    """ train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['prediction'] = prediction 
    print(valid)
    quote = d
    new_df=valid.filter(['prediction'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    print("Next Day predition :",pred_price) """

    # Prepare for future predictions
    length_test=len(test_data)
    fut_inp = test_data[length_test-100:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)

    #Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()
    print(tmp_inp)
    
    # Predict the next 365 days
    lst_output=[]
    n_steps=100
    i=0
    while(i<366):
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    print(lst_output)
    ds_new = scaler_data.tolist() # Convert the scaler data to a list
    ds_new.extend(lst_output) # Extend the list with the newly predicted future values
    #Creating final data for plotting
    final_graph = scaler.inverse_transform(ds_new).tolist() # Convert the combined list back to the original scale
    print("final_graph\n",final_graph)

    val=[]
    # Flattening final_graph and appending each element to val
    for i in final_graph:
        val.append(*i)
    print(val)
    a=d[["Date","Close"]] # Create a new Data set using Date and Close column
    extension = pd.date_range(start=str(date.today()), periods=365, normalize=True) # Get date range till 365 days
    a['Date'] = a['Date'].dt.date # Remove time component
    additional_dates = pd.DataFrame({'Date': extension.date})  # Adds the data to the Date column
    additional_dates["Close"]=val[length_scaler+1:] # Adds the data to the Close column
    df = pd.concat([a, additional_dates], ignore_index=True) # Concat the two data sets
    # Takes the final_value and subtracts with the new final value from Close column to check if it is gain or loss
    pr=df["Close"][len(df)-1]-final_value  
    if pr>0:
        color="green"
        info="Gain"
    else:
        color="red"
        info="Loss"
    from pprint import pprint
    print(df) # prints the new dataframe
    return df,accuracy,mse,rmse,color,pr,info # Pass the data to the app.py