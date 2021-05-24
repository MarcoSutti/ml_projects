import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# Import the datetime module from the datetime library
from datetime import datetime

# Close all already opened plots (if any)
plt.close("all")

# MS: Read the data into a Pandas DataFrame
df = pd.read_csv("sphist.csv")
# print(df.head())

# MS: Convert the "Date" column to a Pandas date type
df["Date"] = pd.to_datetime(df["Date"])

# MS: You can perform comparisons with
# print(df["Date"] > datetime(year=2015, month=4, day=1))

# MS: Sort the dataframe on the Date column in ascending order:
df.sort_values(["Date"], inplace=True, ascending=True)

# MS: Inspect the first 5 rows to check that it is correctly sorted:
# print(df.head())

# print(df.iloc[0:4, 3])
"""
N = len(df)

day_5 = [0]*N
day_30 = [0]*N
day_365 = [0]*N

# Pick 3 indicators to compute 
for idx in range(0,N):
    # MS: Select the five rows prior to the current one,
    # for the column "Close"
    day_5[idx] = np.average(df.iloc[idx-5:idx, 3])
    day_30[idx] = np.average(df.iloc[idx-30:idx, 3])
    day_365[idx] = np.average(df.iloc[idx-365:idx, 3])

# Generate a different column for each indicator
df["day_5"] = day_5
df["day_30"] = day_30
df["day_365"] = day_365

# Check the last five rows
print(df.head(10))
"""

"""
MS: Using Pandas has some time series tools instead.
NB: Note: There is a giant caveat here, which is that the rolling mean will use
the current day's price. You'll need to reindex the resulting series to shift
all the values "forward" one day. For example, the rolling mean calculated for
1950-01-03 will need to be assigned to 1950-01-04, and so on. You can use the
shift method on dataframes to do this.
"""
df["day_5"] = df["Close"].shift(periods=1, freq=None, axis=0).rolling(5).mean()

# MS: Check that the averages coincide with the ones given in the énoncé:
# print(df["day_5"].iloc[250:260])

df["day_30"] = df["Close"].shift(periods=1, freq=None, axis=0).rolling(10).mean()
df["day_365"] = df["Close"].shift(periods=1, freq=None, axis=0).rolling(365).mean()

# MS: Add another indicators:
# The standard deviation of the price over the past 5 days.
df["std_5"] = df["Close"].shift(periods=1, freq=None, axis=0).rolling(5).std()

# Ratio between the average price for the past 5 days, and the average price for the past 365 days.
df["ratio"] = df["day_5"]/df["day_365"]

# MS: The average volume over the past five days.
df["vol_5"] = df["Volume"].shift(periods=1, freq=None, axis=0).rolling(5).mean()

# The average volume over the past year.
df["vol_365"] = df["Volume"].shift(periods=1, freq=None, axis=0).rolling(365).mean()

# The month component of the date.
df["month"] = df["Date"].dt.month

# The year component of the date.
df["year"] = df["Date"].dt.year

# The standard deviation of the average volume over the past five days.
df["std_vol_5"] = df["Volume"].shift(periods=1, freq=None, axis=0).rolling(5).std()

# The standard deviation of the average volume over the past year.
df["std_vol_365"] = df["Volume"].shift(periods=1, freq=None, axis=0).rolling(365).std()

# MS: Inspect first 10 rows of the dataframe:
print(df.head(10))

# MS: Remove any rows from the DataFrame that occur before 1951-01-03.
df = df[df["Date"] > datetime(year=1951, month=1, day=2)]

# Use the dropna method to remove any rows with NaN values.
# Pass in the axis=0 argument to drop rows.
df.dropna(axis=0, inplace=True)

"""
Split the DataFrame into train and test.
train should contain any rows in the data with a date less than 2013-01-01.
test should contain any rows with a date greater than or equal to 2013-01-01.
"""
train = df[df["Date"] < datetime(year=2013, month=1, day=1)]

# Check:
# print(train.tail())

test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]

# Check:
# print(test.head())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# MS: Initialize an instance of the LinearRegression class.
linear_model = LinearRegression()

"""
Train a linear regression model, using the train dataframe. Leave out all of
the original columns (Close, High, Low, Open, Volume, Adj Close, Date) when
training your model. These all contain knowledge of the future that you don't
want to feed the model. Use the Close column as the target.
"""
features = ['day_5', 'day_30', 'day_365', 'vol_365', 'std_vol_365']
target = ['Close']

linear_model.fit(train[features], train[target])

# MS: Make predictions for the Close column of the test data.
predictions = linear_model.predict(test[features])

# MS: Compute the error between the predictions and the Close column of test.
mae = mean_absolute_error(test[target], predictions)
print(mae)


# Make plot:
fig, ax = plt.subplots( figsize=(10,7) )   # figsize=(10,7) increase the default size of the window

ax.scatter(test["Date"], test[target], marker='o', c='0.7', label='historical data')
ax.scatter(test["Date"], predictions, marker='x', c='red', label='model predictions')

# MS: Put a tick every 100 entries:
ax.set_xticks(test["Date"][::100])
# .dt.date removes time from datetime:
ax.set_xticklabels(test["Date"][::100].dt.date, rotation=45)
# ax.set_xlabel('Date')
ax.set_ylabel('Closing price')
ax.set_title('S&P500 Index')

# Turn on legend
ax.legend(loc='lower right')

# Turn on the grid
ax.grid(linestyle='dashed', color='0.7')
# Place the grid BEHIND the other graphs:
ax.set_axisbelow(True)

plt.show(block=False)

plt.savefig('S&P500_predictions.jpg')  
