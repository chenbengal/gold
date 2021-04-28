import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from modules import *
import numpy as np


gold_prices = pd.read_csv('monthly_csv.csv')

gold_prices.describe()
type(gold_prices.Date[0])
gold_prices.index = [datetime.strptime(gold_prices.Date[i], '%Y-%m') for i in range(len(gold_prices))]
gold_prices.drop(['Date'], axis=1, inplace=True)
gold_prices.head()


moving_av = np.array([np.mean(gold_prices.Price[i-12:i]) for i in range(12,len(gold_prices)+1)])
moving_std = np.array([np.std(gold_prices.Price[i-12:i]) for i in range(12,len(gold_prices)+1)])
std_by_three = [i * 3 for i in moving_std]
anomaly_by_variance = list((gold_prices.Price[12:] >= moving_av[:-1] + std_by_three[:-1]) | (gold_prices.Price[12:] <= moving_av[:-1]-std_by_three[:-1]))

# Check for anomaly

sum(anomaly_by_variance)/len(anomaly_by_variance)

gold_prices['anomaly_by_variance'] = list(np.repeat(False,12)) + anomaly_by_variance
colors =  list(gold_prices['anomaly_by_variance'].replace({False:'b',True:'r'}))
plt.scatter(gold_prices.index,gold_prices.Price, c = colors)
plt.show()

moving_25q = np.array([np.quantile(gold_prices.Price[i-12:i],.25) for i in range(12,len(gold_prices)+1)])
moving_75q = np.array([np.quantile(gold_prices.Price[i-12:i],.75) for i in range(12,len(gold_prices)+1)])
moving_iqr = moving_75q-moving_25q
noving_med = np.array([np.quantile(gold_prices.Price[i-12:i],.5) for i in range(12,len(gold_prices)+1)])

moving_iqr_by_three = [2* i for i in moving_iqr]
anomaly_by_iqr = list((gold_prices.Price[12:] >= moving_75q[:-1] + moving_iqr_by_three[:-1]) | (gold_prices.Price[12:] <= moving_25q[:-1]-moving_iqr_by_three[:-1]))


gold_prices['anomaly_by_iqr'] = list(np.repeat(False,12)) + anomaly_by_iqr
colors =  list(gold_prices['anomaly_by_iqr'].replace({False:'b',True:'r'}))
plt.scatter(gold_prices.index,gold_prices.Price, c = colors)
plt.show()

# Prediction

