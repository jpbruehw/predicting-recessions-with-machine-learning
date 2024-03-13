# import starting packages
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import probplot
from sklearn.preprocessing import MinMaxScaler
#from SECRET import API_KEY

# set up access
#fred = Fred(api_key=API_KEY)
fred = Fred(api_key='ENTER KEY')

# get series for the NBER recession indicator
nber_recession_data = fred.get_series('USREC')

# get the ten year treasury yield
treasury_10_yr = fred.get_series('DGS10').resample('M').mean()
treasury_10_yr /= 100

# the FRED API's 3-month t-bill rate only goes back to the 80s
# due to this, we can call the data from the yfinance API
# the ticker that represents the 3-month t-bill is ^IRX
# unfortunately, monthly data only goes back to the 80s too
# the daily data goes back much further so we will use that
# then take the average of each month
t_bill_ticker = '^IRX'
# earliest available start date is 1960-01-04
t_bill_start_date = '1960-01-04'
# set end date
# get the current date from the datetime
# https://docs.python.org/3/library/datetime.html#datetime.datetime.now
end_date_yf = datetime.now()
# perform some basic formatting
# date needs to be a string
# and a very specific format
formatted_end_date = end_date_yf.strftime('%Y-%m-%d')
# set the interval to daily
t_bill_interval = '1d'
# make api call
daily_t_bill = yf.download(t_bill_ticker, start=t_bill_start_date, end=end_date_yf, interval=t_bill_interval)
# extract the closing data
t_bill_closing = daily_t_bill['Close']
# group by month and get the average for each month
t_bill_monthly_data = t_bill_closing.resample('M').mean()
# convert data from float to %
t_bill_monthly_data /= 100

# check if data is retrieved successfully
if treasury_10_yr is not None and t_bill_monthly_data is not None:
    # combine data into DataFrame
    df = pd.DataFrame({'10Y': treasury_10_yr, '3MO': t_bill_monthly_data})
    
    # calculate spread (10Y - 3MO)
    df['T10Y3MO'] = df['10Y'] - df['3MO']

# SET UP & GET S&P500 DATA
#------------------------#

# unfortunately, the yfinance api doesn't have enough data
# on yfinance, the S&P500 data only goes back to the mid-80s
# therefore we need to use an external data source
# we use the data from datahub
# https://datahub.io/core/s-and-p-500#pandas
# this goes until 2018 and we can get the rest from yfinance
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# import the sp500 data
# adjust file path
raw_sp500_data = pd.read_csv('ENTER PATH')

# create a series of the relevant data
sp500_data = pd.Series(raw_sp500_data['SP500'].values, index=raw_sp500_data['Date'])

# extract the last date so we can use it 
# to get the start date for the yfinance api call
raw_sp500_end_date = sp500_data.index[-1]

# we now need to convert the string to a format we can pass
# curr format: '%m/%d/%Y'
# yfinance format: '%Y-%m-%d'
# we first convert to datetime, then back to string
# pd.to_datetime: https://shorturl.at/dfwzP
# strftime: https://www.programiz.com/python-programming/datetime/strftime
# add one month to the end date so that it gets the data starting in following month
# where the datahub dataset ends
# we can us the relativedelta package to manipulate dates
# dateutil: https://dateutil.readthedocs.io/en/stable/relativedelta.html
formatted_start_date = pd.to_datetime(raw_sp500_end_date, format='%m/%d/%Y')
formatted_start_date = formatted_start_date + relativedelta(months=1)
formatted_start_date = formatted_start_date.strftime('%Y-%m-%d')

# set ticker for sp500
ticker_yf = "^GSPC"

# set start date to match other data
start_date_yf = formatted_start_date

# set end date
# we can use the same end_date we used earilier to make the call
# for the t_bill data
# see lines 33-37

# set interval to monthly
interval = "1mo"

# get stock market return data
market_data = yf.download(ticker_yf, start=start_date_yf, end=end_date_yf, interval=interval)

# create series to extract closing prices
market_data = pd.Series(market_data['Close'])

# now that we have the yfinance data, we convert the original index data to DateTime
# to make the api call, we need to pass in the date as a string
# but it returns the data as DateTime
# the format for the date changes so we pass in mixed as the format of the string
# this means it will match string format with a format it recognizes and converts it
# this is sufficient for converting to the same DateTime strucure used by yfinance
sp500_data.index = pd.to_datetime(sp500_data.index, format='mixed')

# concat the market data with other raw data
all_sp500_data = pd.concat([sp500_data, market_data], axis=0, ignore_index=False)

# GET SENTIMENT DATA
#------------------#

# retrieve data for UMCSENT series 
# University of Michigan's Index of Consumer Sentiment
# this is a reliable sentiment tracker which is used as part
# of the LEI index
# http://www.sca.isr.umich.edu
consumer_sentiment_data = fred.get_series('UMCSENT')
# up until 1978-01-01 the data was quarterly
# after that point it is collected monthly
# to account for this we can use the same cubic spline technique
# to decide which order of interpolation to use we can do a visual inspection
# depending on what the data looks like, we can decide on the order
# first step is to only select data after the date where data is collected monthly
# create the timestamp as a datetime object
test_start_date = pd.to_datetime('1978-01-01')
monthly_data_copy = consumer_sentiment_data[consumer_sentiment_data.index >= test_start_date]
# plot the data
plt.figure(figsize=(10,6))
plt.plot(monthly_data_copy, linewidth=3, color="lightblue")
plt.title("Monthly Consumer Sentiment Data (1978 - Today)")
plt.ylabel("Sentiment Level")
plt.xlim(monthly_data_copy.index[0], monthly_data_copy.index[-1])
plt.grid(True)
plt.show()
# we will run the interpolation and see if the graphs look similar
# we can adjust the order and perform futher visual inspection to find the best fit
monthly_consumer_sentiment_data = consumer_sentiment_data.interpolate(method='spline', order=3)
# select the data prior to 1978-01-01 and plot the data
interpolated_quarterly_sentiment = monthly_consumer_sentiment_data[monthly_consumer_sentiment_data.index < test_start_date]
# plot the data
plt.figure(figsize=(10,6))
plt.plot(interpolated_quarterly_sentiment, label="Interpolated Monthly Data", linewidth=3, color="lightblue")
plt.plot(consumer_sentiment_data[consumer_sentiment_data.index < test_start_date], label="Quarterly Consumer Sentiment",
         linewidth=0, marker='o', markersize=3, color='red')
plt.title("Interpolated Quarterly Consumer Sentiment Data")
plt.ylabel("Sentiment Level")
plt.xlim(interpolated_quarterly_sentiment.index[0], interpolated_quarterly_sentiment.index[-1])
plt.legend()
plt.grid(True)
plt.show()
# plot all the data to check
plt.figure(figsize=(10,6))
plt.plot(monthly_consumer_sentiment_data, linewidth=3, color="lightblue")
plt.title("Interpolated Monthly Consumer Sentiment Data")
plt.ylabel("Sentiment Level")
plt.xlim(monthly_consumer_sentiment_data.index[0], monthly_consumer_sentiment_data.index[-1])
plt.grid(True)
plt.show()
# from the visual inspection it seems like order=3 fits the data best
# right now the data is not in a format which is particularly useful
# we can transform the data to a different scale to measure changes
# in the overall sentiment which is more useful
# to do this, we first need to check the normality of the data

# create probability plot
#-----------------------#
# the way this type of plot works is by sorting the data in ascending order
# for each data point, the corresponding quantile (i.e., the cumulative probability)
# of the theoretical distribution is calculated
# this theoretical data is plotted and then compared with the actual data
# the closer the points hug the line in the middle, the more normal the data is
probplot(monthly_consumer_sentiment_data, dist='norm', plot=plt)
plt.title('Probability Plot of Monthly Sentiment Data')
plt.grid(True)
plt.show()
# the data is not totally normally distributed which is not surprising
# however, the data follows a normal distribution pretty closely
# given this, we'll use scaled values as a measure of deviations from the mean
# unlike z-score standardization, which assumes normality, scaled values 
# provide a straightforward way to transform the data into a more interpretable format
# without relying on specific distribution assumptions
# each scaled value represents its position within the range of the original data
# this metric is valuable for capturing sharp rises and falls in sentiment
# which likely correlate with economic recessions
# while normality is assumed, since the data approaches normality
# the logic of the metric makes sense as a way to transform the data to a more useful format
# we are also not making any assumption about the distrubtion of the datas
# i.e. whether or not it is normal affects the transformation we can do
# but is functionaly irrelevant for the analysis
# we're simply transforming the raw values to a standardized scale for easier interpretation
# which doesn't rely on normality, otherwise we could have used something
# like z-score standardization to measure devaitions from the mean
# Min-Max normalization: https://shorturl.at/esyDZ
# sv = (X - Xmin) / (Xmax - Xmin)

# sklearn expects a 2D array as an input
# i.e. a data structure of rows and columns
# right now the data is a series which is incompatible
# so we can transform this into a 2d array
# we can do this by using the .reshape() method from numpy
# by using reshape(1, -1) we are telling numpy to create
# a 2d array with 1 column that holds all the data
# and the -1 is telling nump[y to infer the number of columns
# based on the data structure which in our case is only 1
# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
reshaped_monthly_consumer_sentiment_data = monthly_consumer_sentiment_data.values.reshape(-1, 1)
# define the scaler object
scaler = MinMaxScaler()
# transfom original data
scaled_monthly_sentiment_data = scaler.fit_transform(reshaped_monthly_consumer_sentiment_data)
# convert the numpy array back to series so we can add back the index
# by passing in reshape(-1) we are telling numpy to flatten the array back into one dimensional
scaled_monthly_sentiment_data = pd.Series(scaled_monthly_sentiment_data.reshape(-1), 
                                          index=monthly_consumer_sentiment_data.index)
# perform some exploratory analysis
# since scaled values are not directly interpretable
# i.e. a value of 0.5 doesn't necessarily mean the average
# we can look at the average unscaled value and compare it to the scaled
# value to be better guage what the sentiment values mean
raw_sentiment_avg = np.average(monthly_consumer_sentiment_data)
print(raw_sentiment_avg)
scaled_sentiment_avg = np.average(scaled_monthly_sentiment_data)
print(scaled_sentiment_avg)
# so we can see the avg raw score is about 86
# and the avg scaled score is about 0.6
# so now we can say a scaled score of 0.6 is equivalent to 86
# so anything that falls above 0.6 is above average sentiment
# and anything below 0.6 represents pessemistic sentiment

# SET UP CONSUMER SPENDING VARIBLE
#--------------------------------#

# changes in consumer spending have long been used 
# as a leading recession indicator
# the FED monitors this metric closely
# we can use the FRED to get the PCE dataset
# Personal Consumption Expenditures
# basically a measure of how much US consumers spend
# on durable and non-durable goods
# the spending level is seasonally adjusted
# https://www.investopedia.com/terms/p/pce.asp
# the format of the series is the seasonally adjusted
# annual rate which we then need to adjust for inflation

# get the retail sales series
consumer_spending_raw = fred.get_series('PCE')
# get inflation data
consumer_goods_inflation = fred.get_series('CPIAUCSL')
# edit the formatting to make it fit
# we want to change the raw spending to the annual change
annual_consumer_spending_change = np.log(consumer_spending_raw / consumer_spending_raw.shift(12))
# the first 12 monthly entries are NaN so we can  safely dropna()
annual_consumer_spending_change = annual_consumer_spending_change.dropna()
# now we do the same thing for the consumer goods inflation index
# the format is a monthly inflation level with 1983-08-01 as ~100
annual_consumer_inflation_change = np.log(consumer_goods_inflation / consumer_goods_inflation.shift(12))
# we can now dropna() here too
annual_consumer_inflation_change = annual_consumer_inflation_change.dropna()
# make a dataframe to line up the date ranges
consumer_spending_inflation_df = pd.DataFrame({'annual_consumer_spending_change': annual_consumer_spending_change,
                                               'annual_consumer_inflation_change': annual_consumer_inflation_change})
# drop the na values so we can find the real change
consumer_spending_inflation_df = consumer_spending_inflation_df.dropna()
# create function to calculate the after inflation growth/decline
def find_real_growth(nom_change, infl_rate):
    return (nom_change - infl_rate) / (1 + infl_rate)
# create column to find teh real change
consumer_spending_inflation_df['real_consumer_spending_change'] = find_real_growth(consumer_spending_inflation_df['annual_consumer_spending_change'],
                                                                                   consumer_spending_inflation_df['annual_consumer_inflation_change'])
# extract the column for the real change
change_consumer_spending_real = consumer_spending_inflation_df['real_consumer_spending_change']

# GET AND SET UP GDP DATA
#-----------------------#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# the data we recieve is quarterly so we need to rebase this to monthly 
# to achieve this as best as possible, we use spline interpolation   
# this smooths the data between periods to a monthly basis
# this method is also employed in literature which uses FRED API data
# https://www.geeksforgeeks.org/cubic-spline-interpolation/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# get gdp data
# GDPC1 get real inflation adjusted quarterly data
# the data is in 2017 dollars as a base
gdp_data = fred.get_series('GDPC1')
# in this step we are converting the series index to monthly
# this isn't resampling, it is just changing the date format
# this is necessary to perform the subsequent data manipulation
gdp_data.index = gdp_data.index.to_period(freq='M')

# resample to monthly frequency and fill missing months with NaN
# we resample the quarterly data to monthly
# resample requires some sort of chaining method, but using .asfreq()
# without passing anything in it fills the missing monthly values with NaN
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
monthly_gdp_data = gdp_data.resample('M').asfreq()

# convert the index to DateTime
# we need to do this, otherwise the splie method won't work
# right now the index is a period index and not DateTime
# certain pandas methods only work when the index is DateTime
# the method for converting a PeriodIndex to DateTime is to_timestamp()
# https://shorturl.at/KQ468
monthly_gdp_data.index = monthly_gdp_data.index.to_timestamp()

# perform spline interpolation to fill NaN values
# spline interpolation uses polynomials to estimate missing values
# the order property is used to determine the smoothness of the curve
# order=3 means a cubic polynomial function is used (i.e. linear would be order=1)
# this is often used to balance between capturing patterns and avoiding overfitting
monthly_gdp_data_interpolated = monthly_gdp_data.interpolate(method='spline', order=3)

# calculate the date range for the last 20 years
# viewing plot of whole data will make it hard to accurately inspect
# by picking subset we can better see if the data is very different
end_date = monthly_gdp_data_interpolated.index[-1]
start_date = end_date - timedelta(days=365 * 20)

# find the nearest available timestamps in the DataFrame's index
# we do this to set the limits for the y-axis for easier interpresation
# if we didn't do this the gdp line would look like just a flat line
# we need to make this adjustment because the start and end date might not choose
# exact dates available in the index
# for setting the timespan limit this is fine but for the y-axis this is relevant
# we use the asof() method to find the closest available valid date based on the start and end
# this lets us pass in a date and find the closeset match
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.asof.html
start_date_nearest = monthly_gdp_data_interpolated.index.asof(start_date)
end_date_nearest = monthly_gdp_data_interpolated.index.asof(end_date)

# create a plot of the GDP data for visual inspection
plt.figure(figsize=(10,6))
plt.plot(monthly_gdp_data_interpolated, label='Interpolated Monthly GDP', linewidth=3, color="lightblue")
plt.plot(gdp_data, label='Quarterly Actual', linewidth=0, marker='o', markersize=3, color='red')
plt.title('Interpolated Real Monthly GDP vs. Quarterly Actual')
plt.xlabel('Time')
# set x limits to date range
plt.xlim(start_date, end_date)
# create y limit for min to better see growth/decline
plt.ylim(monthly_gdp_data_interpolated[start_date_nearest], monthly_gdp_data_interpolated[end_date_nearest] + 1000)
plt.ylabel('GDP ($ Billion)')
plt.legend()
plt.grid(True)
plt.show()

# plot looks good
# looks like the interpolation fit the data well
# we now convert the monthly gdp data to the growth/decline
# we will use continous log change
real_gdp_change = np.log(monthly_gdp_data_interpolated / monthly_gdp_data_interpolated.shift(1)).dropna()

# GET INDUSTRIAL PRODUCTION DATA
#------------------------------#

# make call to api
# indutrial production index
# gives broad measure of overall manufacturing sector
industrial_production_raw = fred.get_series('INDPRO')
# make plot to inspect the data
# get the dates for recessions
recession_periods = []
# start at index 0
i = 0
# we use this while loop to go through the recession date data
# this loop creates tuples of starttand end dates we can use to plot
while i < len(nber_recession_data):
    # check if curr period is a recession
    if nber_recession_data[i] == 1:
        # set the start date of the recession
        recession_start = nber_recession_data.index[i]
        # keep iterating to find the end of the recession
        # start at i + 1 and keep going until no longer recession
        j = i + 1
        while j < len(nber_recession_data) and nber_recession_data[j] == 1:
            j += 1
        # once the end of the recession is found we can create recession pair
        # the default behavior of axvspan is to go up to the end date
        # so we need to specify the end date as the month AFTER the recession ends
        # this will it will include the final month of the recession
        # so this is saying the recession start at recession_start and goes
        # up until recession_end
        recession_end = nber_recession_data.index[j]
        # append a tuple to the recession periods list
        recession_periods.append((recession_start, recession_end))
        # set i to the period after the recession
        i = j + 1
    else:
        i += 1

# define the plot object
plt.figure(figsize=(10,6))
plt.plot(industrial_production_raw, linewidth=3, color="lightblue")
plt.title("Industrial Production Index - 2017 Base")
# we can use the axvspan to plot recessions
# this matplotlib method lets you add a span i.e.
# shaded bar across the vertical axis
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
# set time span to 1960 for clearer view
x_lim_start = pd.to_datetime('1960-01-01')
plt.xlim(x_lim_start,industrial_production_raw.index[-1])
plt.ylabel("Industrial Production Idx Level")
plt.show()
# based on visual inspection it seems
# there are somewhate consistent expansions and contractions that correlate
# with recessions, but to be useful as a leading indicator the data needs
# to be transformed to some other format
# we can try to look at different moving averages of the line to see
# if a change in the moving average can be more useful

# define function to calculate the moving averages
def calc_simple_moving_avg(data, window):
    '''
    this function takes two arguments
    an array like data structure and
    a window that defines the degree of
    moving average to return
    '''
    # create list to store calculated values
    moving_averages = []
    # initialize an index val at 0
    i = 0
    # loop through all the available windows
    # we need to add 1 after subtracting the window size
    # we do this to ensure that the final window is included
    # otherwise it would always ignore the last window
    while i < len(data) - window + 1:
        # create temp arr to store the values
        # for the given window
        temp_window = data[i : i + window]
        # now we can calculate the average of the window
        window_avg = round(sum(temp_window) / window, 3)
        # append moving average to list
        moving_averages.append(window_avg)
        # shift window to right by 1
        i += 1
    # return the resulting averages as series
    # this lets us preserve the dates as the index
    # we start index at window - 1 so the series starts
    # at the end of the very first window
    return pd.Series(moving_averages, index=data.index[window - 1:])

# define a few windows to test
windows = [3, 6, 9, 12, 18]
# loop over windos and get averages
moving_averages = [calc_simple_moving_avg(industrial_production_raw, window) 
                   for window in windows]

# plot all the moving averages to compare
# define the plot object
# define a list of colors to use for each moving average
moving_avg_colors = ['orange', 'darkred', 'midnightblue', 'steelblue', 'saddlebrown']
plt.figure(figsize=(10,6))
plt.plot(industrial_production_raw, label="Industrial production", linewidth=3, color="lightblue")
plt.title("Industrial Production Index - 2017 Base")
# set up recession periods
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
# loop over the moving averages
for idx, moving_avg_data in enumerate(moving_averages):
    plt.plot(moving_avg_data, label=f"{str(windows[idx])} Month Moving Average", linewidth=1, color=moving_avg_colors[idx])
# set time span to 1980 for clearer view
x_lim_start = pd.to_datetime('1980-01-01')
# get value to pass in as y limit
y_start = industrial_production_raw.loc[x_lim_start]
plt.xlim(x_lim_start, industrial_production_raw.index[-1])
plt.ylim(y_start - 10, industrial_production_raw.max() + 5)
plt.ylabel("Industrial Production Idx Level")
plt.legend()
plt.show()

# the plot renders correctly but still hard to see
# better approach is to plot combinations of the moving averages
# we can then see if crossing of certain averages is a better leading indicator
# initialize empty list
all_combinations = []
# iterate through all combinations of moving averages and prodution data
for i in range(len(moving_averages)):
     # this gives us all combinations of the i element
     # as i increases the amount of new combinations decreases
    for j in range(i + 1, len(moving_averages)):
        # create tuple of combination and industrial data
        window_tuple = (moving_averages[i], moving_averages[j], f"{windows[i]} Month", f"{windows[j]} Month", industrial_production_raw)
        # append to list
        all_combinations.append(window_tuple)
# make a plot of each tuple for comparison
for combination in all_combinations:
    plt.figure(figsize=(10,6))
    plt.title(f"{combination[2]} vs. {combination[3]} Moving Average")
    plt.plot(combination[4], label="Industrial Production Index", linewidth=1.5, linestyle='--', color='midnightblue')
    plt.plot(combination[0], label=f"{combination[2]} Moving Average", linewidth=1, color='darkred')
    plt.plot(combination[1], label=f"{combination[3]} Moving Average", linewidth=1, color='steelblue')
    # set up recession periods
    for recession_length in recession_periods:
        plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
    # set time span to 1990 for clearer view
    x_lim_start = pd.to_datetime('1990-01-01')
    # get value to pass in as y limit
    y_start = industrial_production_raw.loc[x_lim_start]
    plt.xlim(x_lim_start, industrial_production_raw.index[-1])
    plt.ylim(y_start - 10, industrial_production_raw.max() + 5)
    plt.ylabel("Industrial Production Idx Level")
    plt.legend()
    plt.show()
    
# seems like looking at how the moving averages cross each other
# but no combination seems to reliably act as a leading indicator individually
# however, it seems the index seems to cross the 6 month moving average
# in a way that could be used as a leading indicator
# make plot using 6 month mopving average
plt.figure(figsize=(12,8))
plt.title("Industrial Production Idx w/ 6 Month Moving Average")
# extract the 6 month moving average from caluclated averages list
plt.plot(moving_averages[1], label="6 Month Moving Average", linewidth=1.5, color="darkred")
plt.plot(industrial_production_raw, label="Industrial production", 
         linestyle='--',linewidth=1.5, color="midnightblue")
# set up recession periods
for recession_length in recession_periods:
    plt.axvspan(recession_length[0], recession_length[1], color="lightgray", alpha=0.7)
# set time span to 1960 for clearer view
x_lim_start = pd.to_datetime('2000-01-01')
# get value to pass in as y limit
y_start = industrial_production_raw.loc[x_lim_start]
plt.xlim(x_lim_start, industrial_production_raw.index[-1])
plt.ylim(y_start - 10, industrial_production_raw.max() + 5)
plt.ylabel("Industrial Production Idx Level")
plt.legend()
plt.show()

# this also doesn't look like the best approach
# using the moving average is not providing enough of a lead to be useful
# by looking at a plot of the index and recession shading, there is
# certainly a relationship between the index and a coming recession
# so a better approach is to use the monthly change in index level
# just by observing the trendlines it is clear that the growth
# in the production index slows, turns flat, and then negative
# while not to the same degree for every recession it seems
# reliable enough to use as input in a model which could capture
# a multi-faceted somewhat complex relationship like this
log_change_industrial_production = np.log(industrial_production_raw / 
                                         industrial_production_raw.shift(1)).dropna()

# SET UP INDEPENDENT VARIABLES
#----------------------------#
# calculate log returns for each day using numpy log method
# shift function returns the close in the previous row
sp500_log_change = np.log(all_sp500_data / all_sp500_data.shift(1)).dropna()

# extract the diff as yield curve
yield_curve = df['T10Y3MO'].dropna()


# create dataframe of all inputs
all_data = pd.DataFrame({'sp_log_change': sp500_log_change, 'yield_curve': yield_curve,
                         'real_gdp_change': real_gdp_change,
                         'scaled_consumer_sentiment_level': scaled_monthly_sentiment_data,
                         'change_consumer_spending': change_consumer_spending_real,
                         'change_industrial_production': log_change_industrial_production,
                         'nber_recession': nber_recession_data})

# since the data is all monthly, but the dates are different we need to make an adjustment
# some data is collected at the beggining of month and some at the end
# we group it therefore by the month and not the specific day
# we need to add an aggregation function to the groupby
# since the data is already monthly anyways, by adding sum we are not changing any values
# it is esentially rebasing the data to monthly and adding 0 to whatever the value is
# the original monthly values from the unstructured dataframe remains unchanged
all_data = all_data.groupby(all_data.index.to_period('M')).sum()

# find the first row which is not zero for all columns besides the last one
# iloc[] uses indexing so we select all rows, then all columns 
# besides the last one by passing in -1
# we then filter for non-zero values
# != 0 performs element wise operations on each "cell"
# this creates a boolean dataframe where any non-zero value is true
# and any value which is 0 or NaN is considered false
# second, we use all() to find the rows where all values are truthy
# all() essentailly checks whether all the elements are truhy or not
# we pass in axis=1 to check column wise
# this is saying: "check each column and see where they are all truthy"
# this returns a series the length of the original dataframe with true or false values
# each row of the returned series represents an index value in our case
# so we are using this series of boolean values to extract the index
# that we want to use as the start of our data, so everything following that index
# https://shorturl.at/bpBL8
# idxmax() is then used to find the row where the index is maxed
# idxmax() returns the maximum value, which in our case is true (1)
# since the values are all either 0 or 1, it returns the first instance
# https://www.geeksforgeeks.org/python-pandas-dataframe-idxmax/
first_non_zero_row = (all_data.iloc[:, :-1] != 0).all(axis=1).idxmax()
# use loc[] to select all the rows and columns after the specific index we calculated
# we can use a slicing operation to select all the rows and columns after this main 
all_data = all_data.loc[first_non_zero_row:]
# now also do the same thing but check the end
# the gdp data is released quarterly we need a full quarter to get full data
# for that reason, as of Mar. 2023 the current querter is not yet over and most recent
# data has not been released so we need to also make sure that we have full gdp data
# and other data that is released quarterly which we need to interpolate
# we do the exact same thing as before, we just now go in the opposite direction
last_non_zero_row = (all_data.iloc[:, :-1] != 0).all(axis=1)[::-1].idxmax()
# now update the data and extract up to that column
all_data = all_data.loc[:last_non_zero_row]

# explore the data
# goal is to make sure the subset of data is sufficiently representative
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# check how many recession months are in the dataset as a whole
# this is the recession series from the production dataset
recession_series = all_data['nber_recession']
# this data is taken from the raw series from the FRED API
total_recession_months_all_data = np.sum(nber_recession_data)
print('Total Recession Months - All Data: ', total_recession_months_all_data)

total_recession_months = np.sum(recession_series)
print('\nTotal Recession Months - Production Data: ', total_recession_months)

total_share_of_all_recession_data = np.sum(nber_recession_data) / len(nber_recession_data)
print('\nShare of Total Months - All Data:', total_share_of_all_recession_data)

share_of_dataset_total = total_recession_months / len(recession_series)
print('\nShare of Total Months - Production Data:', share_of_dataset_total)

# create function to find the total amount of recessions
def total_recessions_for_period(data):
    # initialize count
    recession_count = 0
    # iterate through the data points starting from the second month
    # we do this so we have a previous month to compare to
    for i in range(1, len(data)):
        current = data[i]
        prev = data[i - 1]
        # check if the current month is in a recession (current == 1) 
        # and the previous month was not in a recession (prev == 0)
        if current == 1 and prev == 0:
            recession_count += 1
    return recession_count

# call function for entire recession series
total_recessions_raw = total_recessions_for_period(nber_recession_data)
print('\nTotal Recessions for Entire Raw Dataset: ', total_recessions_raw)
total_recessions_production = total_recessions_for_period(recession_series)
print('\nTotal Recessions for Production Dataset: ', total_recessions_production)

# look at how often recessions occur now that we have a total count
recession_frequency_raw = (len(nber_recession_data) / 12) / total_recessions_raw
print('Recession Frequency Raw Data: ', recession_frequency_raw)
recession_frequency_production = (len(recession_series) / 12)/ total_recessions_production
print('Recession Frequency Production Data: ', recession_frequency_production)

# NOTES
#~~~~~#

# V1 - 2024-02-16
#---------------#
# so it looks like at first glance that the production data is not representative
# there are significantly more recessions in the previous data than in the prodcution set
# the dataset is also very small and likely won't be effective for machine learning algorithms
# the main bottlenecks are how short the SP500 data and 3-month t-bill rates are
# the dataset as a whole needs to be expanded
# it should be noted that the nber recession data goes back much farther than any
# other input that can be collected, the data literally goes back to the 1850s
# right now, the dataset starts in the early 80s, so the next steps are to get
# better datasets that can expand the production dataset as a whole

# V2 - 2024-02-26
#---------------#
# several improvements have been made and the production dataset has doubles
# a new datasource for sp500 data was found which goes from inception to 2018
# the rest can then be pulled from the Yahoo Finance API to plug the missing data
# from 2018 until today
# another data source was found to get the 3-month t-bill data from
# additionally, a sentiment indicator was added as an input
# overall, the dataset has now been expanded by about 20 years
# and now starts in 1960 as opposed to the early 80s
# while not perfect, the data is now much more representative of
# the dataset as a whole and the total recessions in the production
# dataset has almost doubled from V1

# V3 - 2024-03-02
#---------------#
# a variable representing consumer sentiment, consumer spending, as well as industrial production
# have been added to the dataset
# I originally wanted to use the ISM PMI index as a benchmark for industrial
# production but it has been discontinued and is now paywalled
# so I decided to use the Industrial Prodution Index which is a general measure
# of production output in the US economy
# properly formatting this data was really challenging and I tooke several
# approaches to make the metric more likely to be useful
# I first tried to look at crossings of moving averages but based on visual
# inspection alone none of these moving average approaches made sense
# so upon further inspection I believe it makes sense to use the raw
# change in index level since it seems to reliably flatten then turn negative
# prior to a recession
# manipulating the data for the annual change in consumer spending was
# more straightforward, the rate was adjusted for inflation
# additionally, the consumer sentiment variable was also simple to set up
# I tested the data for normality, which is wasn't and then adjusted the
# data using a min-max scaler since the raw format wasn't useful
# the addition of these variables has not shortened the dataset

# export the data to excel
#------------------------#
# create the path and create writer
path = 'ENTER PATH'
all_data.to_excel(path, index=True)