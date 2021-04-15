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
    X = df.drop(['NY.GDP.MKTP.CD'], axis=0).values
    y = df.loc['NY.GDP.MKTP.CD'].transpose().values

    # print(X.shape)
    # print(y.shape)

    ridge_mod = Ridge(alpha=a)
    cross_val = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    scores = cross_val_score(ridge_mod, X, y, scoring='neg_mean_absolute_error', cv=cross_val, n_jobs=-1)
    scores = absolute(scores)

    return scores

if __name__ == "__main__":
    '''
    The Main Function of this file, where execution starts.
    '''
    df = wrangle('ALL_DB_COL_data_100_threshold.csv')
    scores = ridge_regression(df)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))