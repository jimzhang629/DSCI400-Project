
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def calculate_f_regression(df, target='NY.GDP.MKTP.CD', k=10):
    """
    This function performs feature selection by calculating the f_regression scores between indicators in
    df data frame and the target variable "target" and select the "k" best features.

    df (dataframe) -- the data frame that includes all indicators as columns
    target (string) -- an indicator that must be a column in df
    k (int) -- the number of best indicators for SelectKBest
    Returns -- a data frame that includes the indicators and their f_regression scores indexed by the original indices
             in df
    """
    # Get the data and drop the target feature
    idx = get_idx("NY.GDP.MKTP.KD", df)
    print(df)
    print(idx)
    df = df.drop(['True'], axis=1) #remove indicator names
    #df = df.drop(['Series'], axis=1) #remove indicator names
    X = df.transpose()
    X = X.drop(idx, 1)
    y = np.asarray(df.loc[idx])
    # Fit the model and save columns and scores in data frames
    bestfeatures = SelectKBest(score_func=f_regression, k=k)
    fit = bestfeatures.fit(X, y)
    scores = pd.DataFrame(fit.scores_)
    indicators = pd.DataFrame(X.columns)
    # concatenate the two data frames and label them
    indicatorsScores = pd.concat([indicators, scores], axis=1)
    indicatorsScores.columns = ['Indicator', 'Score']
    return indicatorsScores.nlargest(k, "Score")



#reading the csv file into pandas data frame
data = pd.read_csv("/work/DSCI400-Project/src/cached_data/norm_ALL_DB_COL_data_100_threshold.csv")

def get_idx(target_ind, data):
    '''
    Gets target indicator index.
    target_ind (string) -- target indicator
    data (Pandas dataframe) -- dataframe of indicators
    '''
    rows = data.shape[0]
    for i in range(rows):
        if data['True'][i] == target_ind:
            return i
    return 0

def pearson_correlation_feature_selection(data, target_ind, corr_threshold = 0.8):
    '''
    Generates the pearson correlation matrix between a target indicator and all other indicators for a country.
    Then, remove the indicators that fail to meet a pre-determined correlation threshold.

    data (Pandas dataframe) -- dataframe of interest
    target_ind (string) -- WB code for indicator of interest
    corr_threshold (float) -- A correlation threshold that an indicator must meet with the target indicator in order to not be removed
    Returns -- a dataframe of the features that meet the correlation threshold with the target indicator
    '''
    
    if abs(corr_threshold) > 1.0:
        print("ERROR: Correlation threshold must be between -1 and 1")
        return None

    target_idx = get_idx(target_ind, data)
    data = data.drop(['True'], axis=1) #remove indicator names
    transpose_data = data.transpose()
    df = transpose_data
    cor = df.corr()[target_idx]

    abs_cor = abs(cor) #Absolute correlation with target variable
    relevant_features = abs_cor[abs_cor>corr_threshold]  #Select the indicators that meet the threshold
    return relevant_features

def plot_target_indicator(target_idx, target_ind, data):
    '''
    Plots 2D line graph of the target indicator. X value is year, Y value is target indicator value at that year.
    target_ind (string) -- target indicator of interest
    data (Pandas dataframe) -- Your data

    Returns: Plots of target indicator
    '''

    data.iloc[:, target_idx].plot(figsize=(25,10))

    plot_acf(data.iloc[:,target_idx])
    plt.title(target_ind)

    plt.show()

def split_data(dataset, target_idx, training_data_proportion):
    '''
    Splits a dataframe into training and test data

    dataset (Pandas dataframe) -- your data
    training_data_proportion (float) -- proportion of data to use as training data

    Returns: Dataset split into training and test data
    '''
    #divide the data into train and test data
    train_size = int(len(dataset) * training_data_proportion)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    #index the data into dependent and independent variables
    train_X, train_y = np.concatenate((train[:, :target_idx], train[:,target_idx + 1:]), axis = 1), train[:, target_idx]
    test_X, test_y = np.concatenate((test[:, :target_idx], test[:,target_idx + 1:]), axis = 1), test[:, target_idx]

    # Convert data into suitable dimension for using it as input in LSTM network
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return train_X, train_y, test_X, test_y

def post_process_PC(df, target_ind, corr_threshold):
    '''
    Post processing function for Pearson Correlation feature selection.
    
    df (dataframe) -- the data frame that includes all indicators as columns
    target_ind (string) -- Target indicator of interest
    corr_threshold (float) -- A correlation threshold that an indicator must meet with the target indicator in order to not be removed
    
    Returns -- the indices of indicators that meet the correlation threshold with the target indicator
    '''
    lst = list(pearson_correlation_feature_selection(df, target_ind, corr_threshold).index.values)
    ind = df.loc[lst]
    ind.index._data = np.array(range(len(lst)))
    return ind

def post_process_FR(df, target_ind, k):
    '''
    Performs post processing for f-regression feature selection.
    
    df (dataframe) -- the data frame that includes all indicators as columns
    target_ind (string) -- Target indicator of interest
    k (int) -- Number of k-best scored indicators to keep.
    
    Returns -- indices of k-best f-score indicators with target indicators 
    
    '''
    lst = list(calculate_f_regression(df,target = target_ind, k = k).index.values)
    ind = df.loc[lst]
    ind.index._data = np.array(range(len(lst)))
    return ind

def LSTM_predictions(target_ind, data, epochs=200, batch_size=72, corr_threshold=0.95):
    '''
    Get LSTM predictions for target indicator

    target_ind (string) -- target indicator (string)
    data (Pandas dataframe) -- dataframe of indicators
    epochs (int) -- number of training epochs
    batch_size (int) -- how much data in each batch
    corr_threshold (float) -- Pearson correlation threshold that an indicator must meet with target indicator in order to be considered in analysis.
    Returns: Future predicted values of target indicator from LSTM model
    '''

    data2 = post_process_FR(data, target_ind, corr_threshold)
    target_idx = get_idx(target_ind, data2)
    data2 = data2.drop(['True'], axis=1) #remove indicator names
    transpose_data = data2.transpose()
    df = transpose_data

    # We will use it further to show the graph of multi step prediction
    df_train = df.iloc[0:int(0.8*transpose_data.shape[0])] #select first 80% of the rows (years)
    df_test = df.iloc[int(0.8*transpose_data.shape[0]):] #select latter 20% of the rows (years)

    # plotting the line graph of target indicator and ACF graph
    plot_target_indicator(target_idx, target_ind, df)

    #Scale the data
    dataset = transpose_data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Split data into training and test data
    train_X, train_y, test_X, test_y = split_data(dataset, target_idx ,0.8);

    # Train the model
    dropout = 0.2
    neurons = 100
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(Dropout(dropout))
    model.add(LSTM(neurons))

    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, train_y, epochs=100, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('optimal_LSTM_2layer_loss_{}dropout_{}neurons_{}corrthreshold.png'.format(dropout, neurons, corr_threshold))
    plt.show()

    # Prediction on training and testing data
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)

    # Converting from three dimension to two dimension
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_train_predict = concatenate((train_predict, train_X), axis=1)
    inv_test_predict = concatenate((test_predict, test_X), axis=1)

    # Transforming to original scale
    inv_train_predict = scaler.inverse_transform(inv_train_predict)
    inv_test_predict = scaler.inverse_transform(inv_test_predict)

    # Predicted values on training data
    inv_train_predict = inv_train_predict[:,0]

    # Predicted values on testing data
    inv_test_predict = inv_test_predict[:,0]

    # Scaling back the original train labels
    train_y = train_y.reshape((len(train_y), 1))
    inv_train_y = concatenate((train_y, train_X), axis=1)
    inv_train_y = scaler.inverse_transform(inv_train_y)
    inv_train_y = inv_train_y[:,0]

    # Scaling back the original test labels
    test_y = test_y.reshape((len(test_y), 1))
    inv_test_y = concatenate((test_y, test_X), axis=1)
    inv_test_y = scaler.inverse_transform(inv_test_y)
    inv_test_y = inv_test_y[:,0]

    # Calculating rmse on train data
    rmse_train = sqrt(mean_squared_error(inv_train_y, inv_train_predict))
    print('Train RMSE: %.3f' % rmse_train)

    # Calculating rmse on test data
    rmse_test = sqrt(mean_squared_error(inv_test_y, inv_test_predict))
    print('Test RMSE: %.3f' % rmse_test)

    # Plotting the graph of test actual vs predicted
    inv_test_y = inv_test_y.reshape(-1,1)
    inv_test_y.shape

    t = np.arange(0,int(0.2*transpose_data.shape[0])+1,1)

    plt.plot(t,inv_test_y, color='g', label='actual')
    plt.plot(t,inv_test_predict, color='b', label="predicted")
    plt.title("Actual vs. Predicted Values")
    plt.ylabel('Value')
    plt.xlabel('Prediction Step')
    plt.legend(loc='best')
    plt.savefig('optimal_LSTM_2layer_{}dropout_{}neurons_{}corrthreshold_actualvspredicted.png'.format(dropout, neurons, corr_threshold))
    plt.show()

    # Plotting the graph to show multi step prediction
    plt.figure(figsize=(25, 10))
    plt.plot(df_train.index, inv_train_predict, color='r', label="training_set_actual")
    plt.plot(df_test.index, inv_test_predict, color='b',label="test_set_predicted")
    plt.plot(df_test.index, inv_test_y, color='g', label='test_set_actual')
    plt.legend(loc='best', fontsize='xx-large')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("Actual Training Set Values vs. Predicted Test Set Values")
    plt.ylabel("Value")
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig('optimal_LSTM_2layer_{}dropout_{}neurons_{}corrthreshold_actualtrainingvspredictedtest.png'.format(dropout, neurons, corr_threshold))
    plt.show()

    return inv_test_predict

LSTM_predictions("NY.GDP.MKTP.KD", data, epochs=100, corr_threshold = 200)
