import csv
import re
import numpy as np
import wbgapi as wb

def get_indicator_codes():
    '''
    (Helper function) Get set of code/id's for all indicators.
    '''
    return set(row['id'] for row in wb.series.list(db=2)) # 2 specifies WDI

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
    Get dataframe of indicators for specific country (and possibly for
    specific year time range).

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

    if min_year is not None and max_year is not None:
        ind_df = wb.data.DataFrame(all_indicators, country_iso_code, \
            time=range(min_year, max_year, 1), labels=True)
    else:
        ind_df = wb.data.DataFrame(all_indicators, country_iso_code, labels=True)

    return ind_df

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

    ind_df['COUNT_NAN'] = test_df[years_list].isnull().sum(axis=1)
    ind_df['COUNT_TOT'] = len(years_list)
    ind_df['COVERAGE_PERCENT'] = (ind_df['COUNT_TOT'] - ind_df['COUNT_NAN']) / ind_df['COUNT_TOT']

    return ind_df.loc[ind_df['COVERAGE_PERCENT'] >= threshold]
    
if __name__ == "__main__":
    '''
    The Main Function of this file, where execution starts.
    '''
    ind_df = get_indicators_for_country('Colombia', 1980, 2011)
    filtered_df = filter_indicators_by_coverage(ind_df, 1.0)
    filtered_df.to_csv('COL_df_100_threshold.csv', index=True, header=True, index_label=True)