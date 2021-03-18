import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.modules.DataWrangling import wrangle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Following https://towardsdatascience.com/
# best-bulletproof-python-feature-selection-methods-every-data-scientist-should-know-7c1027a833c6

# Import data
data = wrangle('../cached_data/ALL_DB_COL_data_100_threshold.csv')

# Remove the feature with all 0 values.
data = data.drop('SE.PRM.DURS')


def calculate_f_regression(df, target='NY.GDP.MKTP.CD', k=10):
    """
    This function performs feature selection by calculating the f_regression scores between indicators in
    df data frame and the target variable "target" and select the "k" best features.

    :param df: the data frame that includes all indicators as columns
    :param target: an indicator that must be a column in df
    :param k: the number of best indicators for SelectKBest
    :return: a data frame that includes the indicators and their f_regression scores indexed by the original indices
             in df
    """
    # Get the data and drop the target feature
    X = df.transpose()
    X = X.drop(target, 1)
    y = np.asarray(df.loc[target])
    # Fit the model and save columns and scores in data frames
    bestfeatures = SelectKBest(score_func=f_regression, k=k)
    fit = bestfeatures.fit(X, y)
    scores = pd.DataFrame(fit.scores_)
    indicators = pd.DataFrame(X.columns)
    # concatenate the two data frames and label them
    indicatorsScores = pd.concat([indicators, scores], axis=1)
    indicatorsScores.columns = ['Indicator', 'Score']
    return indicatorsScores


def display_f_regression(df, target='NY.GDP.MKTP.CD', k=10, prnt=True, plot=False):
    """
    This function displays the output of calculate_f_regression either by printing/plotting the indicators.

    :param df: the data frame that includes all indicators as columns
    :param target: an indicator that must be a column in df
    :param k: the number of best indicators for SelectKBest
    :param prnt: a flag indicating whether to print indicators
    :param plot: a flag indicating whether to plot indicators
    :return: nothing
    """
    # Perform f_regression calculation
    scores = calculate_f_regression(df, target=target, k=k)
    # Print top scores if indicated
    if prnt:
        print(scores.nlargest(k, 'Score'))
    # Plot top scores if indicated
    if plot:
        scores.nlargest(k, 'Score').plot(kind='barh',
                                         title="Indices of {} top f_regression scores for {} indicator.".format(k,
                                                                                                                target))
        plt.show()
    return


# display_f_regression(data)
