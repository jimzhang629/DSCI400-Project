import pandas as pd

from statsmodels.tsa.stattools import grangercausalitytests
from src.modules.DataWrangling import wrangle
from statsmodels.tsa.stattools import adfuller

def granger_threshold(df, indicator, lag, thresh, test='ssr_ftest'):
    """

    @param df:
    @param indicator:
    @param lag:
    @param thresh:
    @param test:
    @return:
    """

    # go through every indicator
    indicators = list(df)
    reduced_df = df.loc[:, indicator] # create candidate df
    for comparison_indicator in indicators:
        comparison = df.loc[:, indicator]  # create comparison matrix
        if comparison_indicator != indicator: # only check causality for indicators that are not the same
            comparison = pd.concat([comparison, df.loc[:, comparison_indicator]], axis=1)

            try:  # grangercausalitytests fails with constant cols
                granger = grangercausalitytests(comparison, lag)

                # find how many tests had a p-value <= thresh
                granger_counter = 0
                for k, v in granger.items():
                    p_score = v[0][test][1]
                    if p_score < thresh:
                        granger_counter += 1

                if granger_counter == lag:
                    reduced_df = pd.concat([reduced_df, df.loc[:, comparison_indicator]], axis=1)

            except:
                print('Comparison Indicator Constant')

    return reduced_df, len(reduced_df.index)


def most_causal_indicator(df, lag, thresh, test='ssr_ftest'):
    """

    @param df:
    @param lag:
    @param thresh:
    @param test:
    @return:
    """
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


df = wrangle('../cached_data/ALL_DB_COL_data_100_threshold.csv')
df = df.T
# granger_df, rows = granger_threshold(df, 'SH.DTH.IMRT', 5, 0.05)
best_indic = most_causal_indicator(df, 5, 0.05)

dummy = 1