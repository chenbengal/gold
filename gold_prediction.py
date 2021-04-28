import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from modules import *


gold_prices = pd.read_csv('monthly_csv.csv')

gold_prices.describe()
type(gold_prices.Date[0])
gold_prices.index = [datetime.strptime(gold_prices.Date[i], '%Y-%m') for i in range(len(gold_prices))]
gold_prices.drop(['Date'], axis=1, inplace=True)
gold_prices.head()

gold_prices.plot.line()

# Check for anomaly
#
foo = Welford()

foo(range(100))
foo(gold_prices.Price)
foo

Welford

# Prediction