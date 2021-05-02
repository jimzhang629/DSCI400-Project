# Modeling Country Indicators

## Project Description
The goal of this project is to visualize and predict relationships between world development indicators in the World Bank. Our objectives are as follows:
1. Visualize relationships between various civil indicators
2. Train a network that predicts various metrics of a country using other metrics as input data
    1. Train network for a single country
    2. Train new networks for other countries
3. Use these networks to predict the future metrics of each country
4. (if time permits) Cluster data by regions or specific metrics to identify trends

## Usage Instructions
TBD

## Dependencies
For a list of necessary packages and package versions, see requirements.txt.

To download these dependencies locally, run
```
$pip install -r /path/to/requirements.txt
```

## Dataset
Data used for this project can be obtained through this API: https://pypi.org/project/wbgapi/

Below is a snippet of sample code to create and populate a data directory locally:
```
import wbgapi as wb

df = wb.data.DataFrame('SP.POP.TOTL', 'COL', db=2, time=range(2000, 2020))
```
The code snippet above creates a local dataframe of values from the SP.POP.TOTL indicator for the country of Colombia (whose ISO code is COL), from the WDI database (signified by db=2), within the range of years from 2000 to 2020.

