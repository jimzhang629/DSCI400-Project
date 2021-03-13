import csv
import numpy as np
import wbgapi as wb

def get_indicator_codes():
    '''
    (Helper function) Get list of code/id's for all indicators.
    '''
    return [row['id'] for row in wb.series.list(db=2)] # 2 specifies WDI

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
    Get set (generator) of indicators for specific country.
    This is a generator function, as it returns a generator object.

    country (string) -- country specified to get indicators for
    min_year (4-digit int) -- start year of data coverage consideration (only
                              used if max_year also available)
    max_year (4-digit int) -- end year of data coverage consideration (only
                              used if max_year also available)   
    '''
    joined_indicator_codes = ';'.join(get_indicator_codes()) # joined for refetch fxn
    country_iso_code = get_iso_code(country)

    if country_iso_code is None:
        return [] # no error print needed as one will be printed in get_iso_code

    if min_year is not None and max_year is not None:
        ind_generator = wb.refetch('sources/{source}/series/{series}/country/{economy}', \
            ['series', 'economy'], source=2, series=joined_indicator_codes, \
            economy=country_iso_code, time=range(min_year, max_year, 1))
    else:
       ind_generator = wb.refetch('sources/{source}/series/{series}/country/{economy}', \
            ['series', 'economy'], source=2, series=joined_indicator_codes, \
            economy=country_iso_code) 

    return ind_generator

def filter_indicators_by_coverage(ind_generator, threshold=0.0):
    '''
    Get set of indicators filtered by data coverage. Indicators that
    have data coverage equal to or above 'threshold' will be included.

    Deciding against converting the generator into a dataframe and just iterating
    due to immense size of data; it will probably be more efficient iterating
    than manipulating the large dataframe for each unique indicator.

    ind_generator (generator) -- list of indicators and their respective values
    threshold (float) -- minimum data coverage amount                         
    '''
    filtered_ind = set()
    prev_ind_code = None
    num_nan = 0
    num_total = 0

    for row in ind_generator:
        curr_ind_code = row['variable'][0]['id']
        curr_value = row['value']

        # calculate stats for prev indicator
        if curr_ind_code != prev_ind_code and prev_ind_code is not None:

            coverage_percentage = (num_total - num_nan) / num_total
            if coverage_percentage >= threshold:
                filtered_ind.add(prev_ind_code)

            num_nan = 0
            num_total = 0

        if curr_value is None:
            num_nan += 1
        num_total += 1
        prev_ind_code = curr_ind_code

    return list(filtered_ind)

def export_array(arr, filename):
    '''
    Exports array of values into a CSV file.

    arr (array) -- array of values to export
    filename (string) -- name of file to export to, including ".csv"
    '''
    # reshape file for export formatting
    len_arr = len(arr)

    if len_arr == 0:
        print("ERROR: export_array cannot convert empty data.")
        return

    reshaped_arr = np.array(arr).reshape((len_arr, 1))

    file = open(filename, 'w+', newline ='')

    with file:     
        write = csv.writer(file) 
        write.writerows(reshaped_arr)
    file.close()

    
if __name__ == "__main__":
    '''
    The Main Function of this file, where execution starts.
    '''
    fetched_ind = get_indicators_for_country('Colombia', 1980, 2010)
    filtered_ind = filter_indicators_by_coverage(fetched_ind, 0.90)
    print(len(filtered_ind))
    export_array(filtered_ind, 'NEW_col_data_90_threshold.csv')