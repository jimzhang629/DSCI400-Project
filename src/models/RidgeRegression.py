import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

def ridge_regression(df, a=1.0):
    '''
    Separates data from 'df' into independent values and dependent (GDP) values.
    Runs ridge regression on this data.
    Returns set of scores of each independent indicator.

    df -- dataframe of values
    a -- alpha value for ridge regression
    '''
    target_idx = df.index.values.tolist().index('NY.GDP.MKTP.CD')

    df = df.drop(df.columns[0], axis=1).transpose()
    df_values = df.values

    # X = df.drop(['NY.GDP.MKTP.CD'], axis=0).transpose().values
    # y = df.loc['NY.GDP.MKTP.CD'].values
    # print(df.transpose().head)

    train_size = int(len(df_values) * 0.80)
    test_size = len(df_values) - train_size
    train, test = df_values[0:train_size,:], df_values[train_size:len(df_values),:]

    train_X, train_y = np.concatenate((train[:, :target_idx], train[:,target_idx + 1:]), axis=1), train[:, target_idx]
    test_X, test_y = np.concatenate((test[:, :target_idx], test[:,target_idx + 1:]), axis=1), test[:, target_idx]

    ridge_mod = Ridge(alpha=a)
    ridge_mod.fit(train_X, train_y)

    y_hat = ridge_mod.predict(test_X)
    # print(y_hat)
    # print(test_y)

    return similarity_btwn_arrs(test_y, y_hat)

def similarity_btwn_arrs(expected_arr, experimental_arr):
    '''
    Calculates percentage difference between two arrays of numbers. Returns the
    mean of all percent differences in the array.

    expected_arr -- array of expected values
    experimental_arr -- array of calculated, predicted values
    '''
    sim = []

    for i in range(len(expected_arr)):
        sim.append(float(expected_arr[i] - experimental_arr[i]) * 100 / expected_arr[i])

    return mean(acc)

if __name__ == "__main__":
    '''
    The Main Function of this file, where execution starts.
    '''
    df = wrangle('ALL_DB_COL_data_100_threshold.csv')
    similarity_score = ridge_regression(df)
    print(similarity_score)