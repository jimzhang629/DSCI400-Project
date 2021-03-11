import world_bank_data as wb

# get data from API
all_indicators = wb.get_indicators()
all_indicators_codes = list(all_indicators.index)
#all_countries = wb.get_countries()

# Colombia country data
co_country = wb.search_countries('Colombia')

# indicator data for Colombia
coverage_threshold = 85.0
keep_indicators = []

for indicator in all_indicators_codes:
    df = wb.get_series(indicator)
    colombia_df = df.loc['Colombia'] # only Colombia data
    num_na = colombia_df.isna().sum()
    num_rows = len(colombia_df)
    coverage_percentage = (num_rows - num_na) / num_rows
    print(coverage_percentage)

    if coverage_percentage >= coverage_threshold: # filter out indicators
        keep_indicators.append(indicator)

print(keep_indicators)

# pop_df = wb.get_series('SP.POP.TOTL')
# colombia_pop_df = pop_df.loc['Colombia']
# num_na = colombia_pop_df.isna().sum()
# num_rows = len(colombia_pop_df)
# coverage_percentage = (num_rows - num_na) / (num_rows)