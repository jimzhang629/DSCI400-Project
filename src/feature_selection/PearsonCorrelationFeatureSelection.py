#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install wbgapi')





import csv
import numpy as np
import pandas as pd
import re
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


def get_indicator_codes():
    '''
    (Helper function) Get list of sets of indicator codes. Each set of indicators
    belong to the same db source.
    '''
    all_codes = []
    db_nums = set(row['id'] for row in wb.source.list())

    for db_iter in db_nums:
        try:
            indicator_set = set(row['id'] for row in wb.series.list(db=db_iter))
            all_codes.append(indicator_set)
        except:
            continue # ignore db's that throw errors with invalid formats

    return all_codes

def get_iso_code(country):
    '''
    (Helper function) Get iso code for country.
    country (string) -- country to get iso code for
    '''
    iso_code = wb.economy.coder(country)

    if iso_code is None:
        print("ERROR: get_iso_code could not resolve country name.")
        return None
    else:
        return iso_code

def get_specific_indicator_code(keyword):
    '''
    (Helper function) Get specific codes for indicators related to a keyword
    keyword (string) -- keyword to get codes for
    '''

    indicator_codes = wb.series.info(q=keyword)

    if indicator_codes is None:
        print("ERROR: indicator_codes could not resolve keyword argument.")
        return None
    else:
        return indicator_codes

def get_indicators_for_country(country, min_year=None, max_year=None):
    '''
    Get dataframe of indicators and respective values for specific country (and
    possibly for specific year time range).
    country (string) -- country specified to get indicators for
    min_year (4-digit int) -- start year of data coverage consideration (only
                              used if max_year also available)
    max_year (4-digit int) -- end year of data coverage consideration, this
                              is exclusive (e.g. max_year=2019 will only
                              retrieve data up to 2018) (only used if max_year
                              also available)
    '''
    all_indicators = get_indicator_codes()
    country_iso_code = get_iso_code(country)

    if country_iso_code is None:
        return pd.DataFrame({'NOTHING' : []}) # no error print needed as one will be printed in get_iso_code

    ret_dfs = [] # list of all indicator df's to be later combined

    for indicator_set in all_indicators:

        try:
            if min_year is not None and max_year is not None:
                ind_df = wb.data.DataFrame(indicator_set, country_iso_code,                     time=range(min_year, max_year, 1), labels=True)
            else:
                ind_df = wb.data.DataFrame(indicator_set, country_iso_code, labels=True)

            ret_dfs.append(ind_df)

        except:
            continue # ignore indicator df's that have fundamental formatting issues

    return pd.concat(ret_dfs, axis=0)

def filter_indicators_by_coverage(ind_df, threshold=0.0):
    '''
    Get set of indicators filtered by data coverage. Indicators that
    have data coverage equal to or above 'threshold' will be included.
    ind_df -- dataframe of indicators and their respective values
    threshold (float) -- minimum data coverage amount
    '''
    df_cols = list(ind_df.columns)
    year_regex = re.compile("^YR[0-9][0-9][0-9][0-9]")
    years_list = [col for col in df_cols if re.search(r"^YR[0-9][0-9][0-9][0-9]", col)]

    ind_df['COUNT_NAN'] = ind_df[years_list].isnull().sum(axis=1)
    ind_df['COUNT_TOT'] = len(years_list)
    ind_df['COVERAGE_PERCENT'] = (ind_df['COUNT_TOT'] - ind_df['COUNT_NAN']) / ind_df['COUNT_TOT']

    return ind_df.loc[ind_df['COVERAGE_PERCENT'] >= threshold]

if __name__ == "__main__":
    '''
    The Main Function of this file, where execution starts.
    '''
    ind_df = get_indicators_for_country('Colombia', 1980, 2011)
    filtered_df = filter_indicators_by_coverage(ind_df, 1.0)
    filtered_df.to_csv('ALL_DB_COL_data_100_threshold.csv', index=True, header=True, index_label=True)


def indicator_dataframe(country, start_year, end_year, coverage_threshold=0.9):
    '''
    country (string) -- The country of interest.
    start_year (int) -- The earliest year of interest.
    end_year (int) -- The latest year of interest.
    coverage_threshold -- The required indicator coverage threshold. For example, if it is 0.9, then there must exist data for 90% of the selected years.
    '''
    fetched_ind = get_indicators_for_country(country, start_year, end_year)
    filtered_ind = filter_indicators_by_coverage(fetched_ind, coverage_threshold)
    country_code = get_iso_code(country)
    df = wb.data.DataFrame(list(filtered_ind.index), country_code, time=range(start_year, end_year), skipBlanks=True, columns='series')

    return df

def pearson_correlation_feature_selection(country, target_indicator_code, start_year, end_year, coverage_threshold = 0.9, corr_threshold = 0.8):
    '''
    Generates the pearson correlation matrix between a target indicator and all other indicators for a country.
    Then, remove the indicators that fail to meet a pre-determined correlation threshold.

    country (string) -- The country of interest.
    target_indicator_code (string) -- The specific indicator code for the target indicator.
    start_year (int) -- The earliest year of interest.
    end_year (int) -- The latest year of interest.
    coverage_threshold (float) -- The required indicator coverage threshold. For example, if it is 0.9, then there must exist data for 90% of the selected years.
    corr_threshold (float) -- A correlation threshold that an indicator must meet with the target indicator in order to not be removed.

    Returns -- a dataframe of the features that meet the correlation threshold with the target indicator.
    '''
    if abs(corr_threshold) > 1.0:
        print("ERROR: Correlation threshold must be between -1 and 1")
        return None

    df = indicator_dataframe(country, start_year, end_year, coverage_threshold)

    cor = df.corr()[target_indicator_code]
    #plt.figure(figsize=(12,10)) #Plot the correlation matrix
    #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()

    abs_cor = abs(cor) #Absolute correlation with target variable
    relevant_features = abs_cor[abs_cor>corr_threshold]  #Select the indicators that meet the threshold
    return relevant_features


pearson_correlation_feature_selection('Colombia', 'SP.POP.TOTL', 2011, 2013, 1, 0.9)
