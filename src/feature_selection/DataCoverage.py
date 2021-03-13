import csv
import numpy as np
import pandas as pd
import world_bank_data as wb

def get_indicator_codes():
    '''
    Get list of code/id's for all indicators.
    '''
    all_indicators = wb.get_indicators()
    return list(all_indicators.index)

def get_indicators_for_country(country, threshold=0.0):
    '''
    Get list of indicators with data for specified country.

    country (string) -- returned indicators should have data available for this
                        country (e.g. "Colombia", "Norway")
    threshold (float) -- returned indicators should have a data coverage for
                         the specified country equal to or above this threshold
    '''
    indicator_codes = get_indicator_codes()
    keep_indicators = []

    try:
        country_data = wb.search_countries(country)
    except: # country name is not valid in dataset
        print("ERROR: get_indicators_for_country could not resolve country name.\n")
        return

    for indicator in indicator_codes:

        try:
            df = wb.get_series(indicator)
        except ValueError:
            continue # ignore deleted or archived indicators

        try:
            country_df = df.loc[country]
        except KeyError:
            continue # no relevant country data in this indicator, so ignore
        
        num_na = country_df.isna().sum()
        num_rows = len(country_df)
        coverage_percentage = (num_rows - num_na) / num_rows
        #print(coverage_percentage)

        if coverage_percentage >= threshold:
            keep_indicators.append(indicator)

    return keep_indicators

def export_array(arr, filename):
    '''
    Exports array of values into a CSV file.

    arr (array) -- array of values to export
    filename (string) -- name of file to export to, including ".csv"
    '''
    # reshape file for export formatting
    len_arr = len(arr)
    reshaped_arr = np.array(arr).reshape((len_arr, 1))
    print(reshaped_arr)

    file = open(filename, 'w+', newline ='')

    with file:     
        write = csv.writer(file) 
        write.writerows(reshaped_arr)
    file.close()


if __name__ == "__main__":
    '''
    The Main Function of this file, where execution starts.
    '''
    col_indicators = get_indicators_for_country('Colombia', 0.90)
    export_array(col_indicators, 'col_data.csv')