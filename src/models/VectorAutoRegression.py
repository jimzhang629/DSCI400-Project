import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests
from src.modules.DataWrangling import wrangle
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error as mse


def granger_threshold(df, indicator, lag, thresh, test='ssr_ftest'):
    '''
    Computes the Granger causation for the indicator in the first column of df
    with all other indicators.

    df (DataFrame) -- The input data.
    indicator (String) -- The indicator being compared with all other indicators.
    lag (Integer) -- The maximum amount of Granger lag permitted
    thresh (Integer) -- The highest p-value to reject the null hypothesis
    test (String) -- The test to use to determine Granger causuality

    return: A reduced dataframe that only has indicators that Granger cause
    the main indicator, and an integer representing how many causal indicators
    there are
    '''

    # go through every indicator
    indicators = list(df)
    reduced_df = df.loc[:, indicator] # create candidate df
    for comparison_indicator in indicators:
        comparison = df.loc[:, indicator]  # create comparison matrix
        if comparison_indicator != indicator: # only check causality for
            # indicators that are not the same
            comparison = pd.concat([comparison,
                                    df.loc[:, comparison_indicator]], axis=1)

            try:  # grangercausalitytests fails with constant cols
                granger = grangercausalitytests(comparison, lag, verbose=False)

                # find how many tests had a p-value <= thresh
                granger_counter = 0
                for k, v in granger.items():
                    p_score = v[0][test][1]
                    if p_score < thresh:
                        granger_counter += 1

                # The comparison indicator Granger causes the target indicator
                if granger_counter == lag:
                    reduced_df = pd.concat([reduced_df,
                                            df.loc[:, comparison_indicator]],
                                           axis=1)

            except:
                print('Comparison Indicator Constant')

    return reduced_df, len(reduced_df.columns)


def most_causal_indicator(df, lag, thresh, test='ssr_ftest'):
    '''
    Determines the indicator that has the most causal indicators

    df (DataFrame) -- The input data.
    lag (Integer) -- The maximum amount of Granger lag permitted
    thresh (Integer) -- The highest p-value to reject the null hypothesis
    test (String) -- The test to use to determine Granger causaulity

    return: The name of the indicator with the most causal indicators
    '''
    largest_df = 0
    best_indicator = None
    indicators = list(df)

    # go through every indicator
    for main_indicator in indicators:
        _, rows = granger_threshold(df, main_indicator, lag, thresh, test)
        if rows > largest_df:
            largest_df = rows
            best_indicator = main_indicator
    return best_indicator


def adf_test(ts, signif=0.05):
    '''
    Tests if the time series is stationary

    ts (Series) -- The input time series
    signif (Float) -- The highest threshold to determine stationarity

    return: Whether the time series is stationary
    '''

    # run ADF test to determine stationarity
    dftest = adfuller(ts, regression='ct')
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags',
                                        '# Observations'])
    for key, value in dftest[4].items():
        adf['Critical Value (%s)' % key] = value

    p = adf['p-value']
    if p <= signif: # the time series is stationary
        return True
    else:
        return False


def df_adf_test(df):
    '''
    Determines if all the time series in df is stationary

    df (DataFrame) - The input data containing multiple time series

    return: Whether all time series in the dataframe is stationary
    '''
    indicators = list(df)

    # check stationarity for all indicators
    for indicator in indicators:
        if not adf_test(df[indicator]):
            return False
    return True


def first_diff(ts):
    '''
    Calculates the first difference of a time series

    ts (Series) -- The input time series

    return: The first difference of the time series
    '''
    return ts.diff().dropna()


def make_stationary(df):
    '''
    Makes all time series in the input dataframe stationary by calculating
    nth differences

    df (DataFrame) -- The input dataframe

    return: The stationary outupt dataframe and how many differences were
    required
    '''
    counter = 0
    stationary_df = df.copy(deep=True)
    stationary = df_adf_test(df)

    # keep applying differences until all series are stationary
    while not stationary:
        indicators = list(stationary_df)
        for indicator in indicators:
            differenced_series = first_diff(stationary_df[indicator])
            stationary_df[indicator] = differenced_series
        counter += 1
        stationary_df = stationary_df.iloc[1:]  # remove new NaN row
        stationary = df_adf_test(stationary_df)

    return stationary_df, counter


def pred_model(data, num_yr):
    '''
    Runs a prediction VAR model on the input data

    DataFrame data (DataFrame) -- The input data
    num_yr (Integer) -- The number of years to forecast

    return: A forecasting vector for all indicators
    '''
    model = VAR(data)
    fit = model.fit()
    prediction = fit.forecast(fit.y, steps=num_yr)
    return prediction


def plot_ind(orig, pred, indicator, val):
    '''
    Plots the original data, the predicted forecasting, and the actual
    future results.

    orig (DataFrame) -- The original df with past and future data
    pred (Series) -- The prediction vector for the target indicator
    indicator (String) -- The name of the target indicator
    val (Series) -- The actual future values of the indicator
    '''

    # construct the future vectors
    orig_ind = orig.iloc[:, 0]
    pred_ind = pred

    # plot the prediction and iinput data
    plt.plot(range(len(orig_ind), len(orig_ind) + len(pred_ind)), pred_ind,
             label='Prediction')
    plt.plot(range(len(orig_ind)), orig_ind, label='Input Data')
    plt.xlabel("Number of Years")
    plt.ylabel("Normalized Indicator Value")
    plt.title("Predicting Future Values for Indicator " + indicator)

    # plot the predicted future
    if len(val) > 0:
        plt.plot(range(len(orig_ind), len(orig_ind) + len(pred_ind)),
                 val.iloc[:, 0],
                 label='Real Values for Predicted Years')
    plt.legend()


def forecast_VAR(df, indicator, granger_lag=5):
    '''
    The end-to-end function to run the VAR model and plot results

    df (Dataframe) -- The dataframe of the input data
    String indicator (String) -- The target indicator to forecast
    granger_lag (Integer) -- The maximum amount of lag allowed by Granger
    causality

    @return: The actual and predicted future vectors
    '''

    # only use indicators that granger cause the target
    df = df.T
    granger_df, rows = granger_threshold(df, indicator, granger_lag, 0.05)

    # todo: implement stationarity check
    # stationary, diff_num = make_stationary(granger_df)

    # separate the data into training and testing
    all_years = granger_df.index
    num_training = math.ceil(len(all_years) * 0.8)
    num_testing = len(all_years) - num_training
    training_years = all_years[-num_testing:]

    # create the VAR model and plot results
    testing_set = granger_df.drop(training_years, 0)
    pred_test = pred_model(testing_set, num_testing)[:, 0]
    pred_test = pd.Series(pred_test, index=training_years)
    val1 = granger_df.loc[training_years]
    plot_ind(testing_set, pred_test, indicator, val1)
    plt.savefig('Figures/' + indicator + '_VAR.png', dpi=200)

    return pred_test, val1.iloc[:, 0]

def forecast_VAR_filename(filename, indicator, granger_lag=5):
    '''
    The end-to-end function to run the VAR model and plot results.
    Helper function that uses filepath name instead of dataframe.

    filepath (String) -- The filepath of the input data
    String indicator (String) -- The target indicator to forecast
    granger_lag (Integer) -- The maximum amount of lag allowed by Granger
    causality
    
    @return: The actual and predicted future vectors
    '''
    df = wrangle(filepath)
    return forecast_VAR(df, indicator, granger_lag)


def calc_all_mse(file):
  """
  calculates the error of var model for every possible target indicator in the file
  @param file: filepath of csv containing un-wrangled dataset
  @return: a dictionary containing the mean-squared-error for each different target indicator after wrangling
  """

  all_data = wrangle(file)
  all_inds = all_data.index
  err = {}
  count = 0
  for ind in all_inds:
      act,pred = forecast_VAR_filename('ALL_DB_COL_data_100_threshold.csv', ind)
      #filters out cases where VAR cannot be run
      #since granger causality leaves only 1 indicator left
      if len(act) != len(pred):
        err[ind] = 99999999
      else:
        err[ind] = mse(act,pred)
      #print iteration number
      print('Indicator' + str(count))
      count+=1
  return err

predicted, actual = forecast_VAR_filename('../cached_data/ALL_DB_COL_data_100_threshold.csv', 'SH.DTH.MORT')
