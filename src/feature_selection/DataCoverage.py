import csv
import re
from modules.DataIngestion import get_indicators_for_country

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