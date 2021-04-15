import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def csv2df(filepath, t=False):
    """
    Returns a dataframe of a cached csv file. The index is the indicator name.
    Rows are countries and cols are years.
    @param filepath: The filepath of the csv file.
    @param t: Transpose result if True, default False.
    @return: A Dataframe object representing the data in the csv file.
    """

    # convert the csv to DatFrame object
    df = pd.read_csv(filepath, index_col=0) # 0 col is indicator name
    res = df.copy(deep=True)

    # remove all cols that are not a year
    cols = df.columns
    for col in cols:
        if not col.startswith('YR'):
            res = res.drop(labels=col, axis=1)

    # convert all YRXXXX indices to just XXXX
    cols = res.columns
    new_cols = []
    for col in cols:
        new_cols.append(col[2:])
    res.columns = new_cols

    # transpose if required
    if t:
        res = res.transpose()

    return res

def normalize(data):
    """
    Normalizes the DataFrame object so all series values are between 0 and 1
    @param data: The DataFrame object to normalize
    @return: A normalized version of data
    """

    # create scaling object
    data = data.transpose()
    scaler = MinMaxScaler()

    # fit the data to the scale and normalize
    scaler.fit(data)
    norm_data = pd.DataFrame(scaler.transform(data), index=data.index,
                             columns=data.columns)

    return norm_data.transpose()


def remove_duplicates(df):
    """
    Removes the duplicate rows of the DataFrame matrix.
    @param df: The input DataFrame matrix.
    @return: An output DataFrame matrix with no duplicated rows.
    """
    return df.drop_duplicates()


def wrangle(filepath, norm=True):
    """
    Imports the CSV file, removes duplicates, and normalizes all the data.
    @param filepath: The filepath of the CSV file.
    @return: A DataFrame file with all normalized and cleaned data.
    """
    df = csv2df(filepath)
    if norm:
        df = normalize(df)
    return remove_duplicates(df)