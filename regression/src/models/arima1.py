import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import  root_mean_squared_error

data = pd.read_csv("../../data/dataset_training.csv", index_col="InvoiceDate")

test = pd.read_csv("../../data/dataset_prediction.csv", index_col="InvoiceDate")

# result = adfuller(data["Quantity"])
# print(f"ADF Statistic: {result[0]}")
# print(f"p-value: {result[1]}") # p-value > 0.05, data is not stationary

p, d, q = 2, 0, 2
model = ARIMA(data["Quantity"], order=(p, d, q))
model_fit = model.fit()
model_summary = model_fit.summary()
print(model_summary)

print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")

test.index = pd.to_datetime(test.index)
test = test.asfreq('MS')

# Predecir las ventas para diciembre de 2010
forecast_steps = len(test.loc['2010-12'])
forecast = model_fit.forecast(steps=forecast_steps)

# Filtrar los datos de test para diciembre de 2010
test_december = test.loc['2010-12']

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Quantity"], label='Train', color='blue')
plt.plot(test_december.index, test_december["Quantity"], label='Test', color='green')
plt.plot(test_december.index, forecast, label='Forecast', color='red')
plt.legend()
plt.title('ARIMA Forecast vs Actual for December 2010')
plt.show()

# Calcular el RMSE para diciembre de 2010
rmse = root_mean_squared_error(test_december["Quantity"], forecast, squared=False)
print(f"RMSE for December 2010: {rmse}")