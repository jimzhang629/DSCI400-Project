import feature_selection.DataCoverage as dc
import feature_selection.F_Regression as fr
from src.modules.DataWrangling import csv2df, normalize

# get data from API
# input: country, years
# output: dataframe of values and indicators
ind_df = dc.get_indicators_for_country('Colombia', 1980, 2011)
filtered_df = dc.filter_indicators_by_coverage(ind_df, 1.0)

# optionally save df as csv for easier access to data later on 
filtered_df.to_csv('ALL_DB_COL_data_100_threshold.csv', index=True, header=True, index_label=True)
got_filtered_df = csv2df('All_DB_COL_data_100_threshold.csv')

# fix up df for use
got_filtered_df = normalize(got_filtered_df)
got_filtered_df = got_filtered_df.transpose()

# featue selection (e.g. f regression)
indicator_scores = fr.f_regression(got_filtered_df)