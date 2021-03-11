import world_bank_data as wb

# get data from API
#all_indicators = wb.get_indicators()
#all_countries = wb.get_countries()

# Colombia country data
co_country = wb.search_countries('Colombia')

# indicator data for Colombia
#indicator_names = 
pop_df = wb.get_series('SP.POP.TOTL')
colombia_pop_df = pop_df.loc['Colombia']
num_na = colombia_pop_df.isna().sum()
num_rows = len(colombia_pop_df)
coverage_percentage = (num_rows - num_na) / (num_rows)