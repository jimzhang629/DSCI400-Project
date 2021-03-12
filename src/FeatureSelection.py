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

    country -- returned indicators should have data available for this country
    threshold -- returned indicators should have a data coverage for the
                 specified country equal to or above this threshold
    '''
    indicator_codes = get_indicator_codes()
    #country_data = wb.search_countries(country)
    keep_indicators = []

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
        print(coverage_percentage)

        if coverage_percentage >= threshold:
            keep_indicators.append(indicator)

    return keep_indicators

print(get_indicators_for_country('Colombia', 0.85))