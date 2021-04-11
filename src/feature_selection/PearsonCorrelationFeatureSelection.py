#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install wbgapi')

import csv
import numpy as np
import pandas as pd
import re
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns
import DataCoverage as dc
import modules.DataIngestion as di

# get_ipython().run_line_magic('matplotlib', 'inline')

def indicator_dataframe(country, start_year, end_year, coverage_threshold=0.9):
    '''
    country (string) -- The country of interest.
    start_year (int) -- The earliest year of interest.
    end_year (int) -- The latest year of interest.
    coverage_threshold -- The required indicator coverage threshold. For example, if it is 0.9, then there must exist data for 90% of the selected years.
    '''
    fetched_ind = dc.get_indicators_for_country(country, start_year, end_year)
    filtered_ind = di.filter_indicators_by_coverage(fetched_ind, coverage_threshold)
    country_code = dc.get_iso_code(country)
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
