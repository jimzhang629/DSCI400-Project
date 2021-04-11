import csv
import re
import pandas as pd
import wbgapi as wb

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
                ind_df = wb.data.DataFrame(indicator_set, country_iso_code, \
                    time=range(min_year, max_year, 1), labels=True)
            else:
                ind_df = wb.data.DataFrame(indicator_set, country_iso_code, labels=True)

            ret_dfs.append(ind_df)

        except:
            continue # ignore indicator df's that have fundamental formatting issues

    return pd.concat(ret_dfs, axis=0)