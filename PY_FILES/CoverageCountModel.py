# necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, datetime

date_range = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
print(date_range)

# I generated dates from 2020-01-01 to today
date_range = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')

# I created random coverage count and payout data
np.random.seed(1)
coverage_count = np.random.poisson(lam=50, size=len(date_range))  #I'm averaging 50 coverages per day

# I created random payouts ranging from $1000 to $10,000
payout_amount = coverage_count * np.random.uniform(1000, 10000, size=len(date_range))  

# the data frame table where everything is merged together
coverage_data = pd.DataFrame({
    'date': date_range,
    'coverage_count': coverage_count,
    'payout_amount': payout_amount
})

# in order to model the fluctuation in the data, I Introduced some random manual shifts
for i in range(0, len(coverage_data), 200):
    coverage_data.loc[i:i+10, 'coverage_count'] += np.random.randint(-10, 10)
    coverage_data.loc[i:i+10, 'payout_amount'] += np.random.uniform(-10000, 10000)

print(coverage_data.head())

#here, I plotted the coverage count trend
plt.figure(figsize=(12, 6))
plt.plot(coverage_data['date'], coverage_data['coverage_count'], label='Coverage Count', color='blue')
plt.xlabel('Date')
plt.ylabel('Number of Coverages')
plt.title('Coverage Count Trend Since 2020')
plt.legend()
plt.grid(True)
plt.show()

# and this would be the payout trend
plt.figure(figsize=(12, 6))
plt.plot(coverage_data['date'], coverage_data['payout_amount'], label='Payout Amount', color='green')
plt.xlabel('Date')
plt.ylabel('Payout Amount ($)')
plt.title('Payout Amount Trend Since 2020')
plt.legend()
plt.grid(True)
plt.show()

# I decomposed the time series for coverage count
coverage_data.set_index('date', inplace=True)
decomposition = seasonal_decompose(coverage_data['coverage_count'], model='additive', period=365)
decomposition.plot()
plt.show()


# since the aim of this project is to detect anomolies in the number of coverages, 
#I used Isolation Forest which is ideal for detecing anomoly trends in the data
iso_forest = IsolationForest(contamination=0.01, random_state=42)
coverage_data['anomaly'] = iso_forest.fit_predict(coverage_data[['coverage_count']])

# i marked anomalies here
anomalies = coverage_data[coverage_data['anomaly'] == -1]

# and ploted coverage count with anomalies highlighted
plt.figure(figsize=(12, 6))
plt.plot(coverage_data.index, coverage_data['coverage_count'], label='Coverage Count', color='blue')
plt.scatter(anomalies.index, anomalies['coverage_count'], color='red', label='Anomaly', s=50)
plt.xlabel('Date')
plt.ylabel('Number of Coverages')
plt.title('Coverage Count Trend with Anomalies Detected')
plt.legend()
plt.grid(True)
plt.show()

# I utilized ARIMA for time series forecasting here, since I'm interested in predicting future values
arima_model = ARIMA(coverage_data['coverage_count'], order=(5, 1, 0))
arima_result = arima_model.fit()

# and set prediction for the next 30 days
forecast = arima_result.forecast(steps=30)
forecast_index = [coverage_data.index[-1] + timedelta(days=i) for i in range(1, 31)]

# I plotted the forecast
coverage_data_2024 = coverage_data[coverage_data.index >= '2024-01-01']
plt.figure(figsize=(12, 6))
plt.plot(coverage_data_2024.index, coverage_data_2024['coverage_count'], label='Historical Coverage Count', color='blue')
plt.plot(forecast_index, forecast, label='Forecast', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Coverage Count')
plt.title('Coverage Count Forecast for Next 30 Days')
plt.legend()
plt.grid(True)
plt.show()

# a table view of the predictions
forecast_table = pd.DataFrame({
    'Date': forecast_index,
    'Predicted Coverage Count': forecast
})

print(forecast_table)




