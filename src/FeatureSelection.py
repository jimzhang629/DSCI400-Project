import world_bank_data as wb

def get_indicator_codes():
    all_indicators = wb.get_indicators(topic=3, source=2)
    return list(all_indicators.index)

def get_indicators_for_country(country, threshold=0.0):
    indicator_codes = get_indicator_codes()
    #country_data = wb.search_countries(country)
    keep_indicators = []

    for indicator in indicator_codes:
        df = wb.get_series(indicator)
        country_df = df.loc[country] # only country-specific data
        num_na = country_df.isna().sum()
        num_rows = len(country_df)
        coverage_percentage = (num_rows - num_na) / num_rows
        print(coverage_percentage)

        if coverage_percentage >= threshold: # filter out indicators
            keep_indicators.append(indicator)

    return keep_indicators

print(get_indicators_for_country('Colombia', 0.85))

# # get data from API
# all_indicators = wb.get_indicators()
# all_indicators_codes = list(all_indicators.index)
# #all_countries = wb.get_countries()

# # Colombia country data
# co_country = wb.search_countries('Colombia')

# # indicator data for Colombia
# coverage_threshold = 85.0
# keep_indicators = []

# for indicator in all_indicators_codes:
#     df = wb.get_series(indicator)
#     colombia_df = df.loc['Colombia'] # only Colombia data
#     num_na = colombia_df.isna().sum()
#     num_rows = len(colombia_df)
#     coverage_percentage = (num_rows - num_na) / num_rows
#     print(coverage_percentage)

#     if coverage_percentage >= coverage_threshold: # filter out indicators
#         keep_indicators.append(indicator)

# print(keep_indicators)

# pop_df = wb.get_series('SP.POP.TOTL')
# colombia_pop_df = pop_df.loc['Colombia']
# num_na = colombia_pop_df.isna().sum()
# num_rows = len(colombia_pop_df)
# coverage_percentage = (num_rows - num_na) / (num_rows)