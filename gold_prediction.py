from datetime import datetime
from matplotlib import pyplot as plt
from modules import *
import numpy as np
import plotly.graph_objects as go

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


# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# contrived dataset
# fit model

model = SARIMAX(gold_prices.Price, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(start = 1, end = len(gold_prices))

len(gold_prices.Price)
len(yhat)
plt.plot(gold_prices.index,gold_prices.Price,label = 'y')
plt.plot(gold_prices.index,yhat, label = 'y_hat')
plt.legend()
plt.show()

loss = abs(yhat-gold_prices.Price)
anomaly_by_sarimax = loss[1:-1] > np.repeat(np.quantile(loss[1:-1],.9),len(loss)-2)
sum(anomaly_by_sarimax)
anomaly_by_sarimax.reset_index(drop=True,inplace=True)
anomaly_by_sarimax = [False] + list(anomaly_by_sarimax)
len(anomaly_by_sarimax)


gold_prices['anomaly_by_sarimax'] = anomaly_by_sarimax
colors =  list(gold_prices['anomaly_by_iqr'].replace({False:'b',True:'r'}))
plt.scatter(gold_prices.index,gold_prices.Price, c = colors)
plt.show()


gold_prices.describe(include='all')

# Predict

train, valid = gold_prices.iloc[:int(round(len(gold_prices)*0.8,0)),],gold_prices.iloc[int(round(len(gold_prices)*0.8,0)):,]

train.head()

n_factors = 10
train_ds = CPUDataset(train, n_factors)
valid_ds = CPUDataset(valid, n_factors)


epochs = 50
cnn_model = CNN(out_size=1)
dataloaders = {
    'train': DataLoader(train_ds, batch_size=128, shuffle=True),
    'valid': DataLoader(valid_ds, batch_size=128)
}
optim = opt.Adam(cnn_model.parameters(), lr=1e-1, weight_decay=1e-3)
sched = opt.lr_scheduler.OneCycleLR(optim, max_lr=1e-3, steps_per_epoch=len(dataloaders['train']), epochs=epochs)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#if __name__ == "__main__":

losses = train_model(cnn_model, dataloaders, optim, sched, criterion, device, epochs)

plt.plot(range(epochs),losses['train'],label = 'train')
plt.plot(range(epochs),losses['valid'],label = 'valid')
plt.legend()
plt.show()


# Switching model into evaluation mode
cnn_model = cnn_model.eval()

# Calculation of the predictions for training data
with torch.no_grad():
    res_train = cnn_model(train_ds[:][0].to(device))
res_train = res_train.cpu()

# Calculation of the predictions for validation data
with torch.no_grad():
    res_valid = cnn_model(valid_ds[:][0].to(device))
res_valid = res_valid.cpu()

len(res_train)

plt.plot(train.index[10:] ,train['Price'][10:] , label = 'y')
plt.plot(train.index[10:],res_train,label = 'y_hat')
plt.legend()
plt.show()
