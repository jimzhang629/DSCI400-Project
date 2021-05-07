# Modeling Country Indicators

## Project Description
The goal of this project is to visualize and predict relationships between world development indicators in the World Bank. Our objectives are as follows:
1. Visualize relationships between various civil indicators
2. Train a network that predicts various metrics of a country using other metrics as input data
3. Use these networks to predict the future metrics of each country

## Usage Instructions
1. Clone this repository by running the following on your terminal
```
$git clone https://github.com/jimzhang629/DSCI400-Project.git
```
2. Download the required dependencies by running the following while in the cloned repository
```
$pip install -r requirements.txt
```
3. To run LSTM.py and generate predictions for a target indicator, import LSTM.py and call LSTM_predictions. The function requires a target indicator (string) and your data (dataframe) as input.
4. To run VectorAutoRegression.py and generate predictions for a target indicator, import VectorAutoRegression.py and call forecast_VAR_filename.  The function requires a filepath (string) to the csv containing the data, as well as a target indicator (string). The function also has an optional paramater, granger_lag (integer), which is 5 by default. It does not require the data to have been wrangled yet, as the module automatically does so.  This function outputs the forecasted series for the target indicator.

## Dependencies
For a list of necessary packages and package versions, see requirements.txt.

To download these dependencies locally, run the following on your terminal
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

