import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Cargar datasets
df_train = pd.read_csv("../../data/dataset_train.csv")
df_test = pd.read_csv("../../data/dataset_prediction.csv")

# Preparar entrenamiento
df_train['InvoiceDate'] = pd.to_datetime(df_train['InvoiceDate'])
daily_train = df_train.groupby(df_train['InvoiceDate'].dt.date)['TotalSales'].sum()
daily_train.index = pd.to_datetime(daily_train.index)
daily_train = daily_train.asfreq('D').fillna(0)

# Preparar test
df_test['InvoiceDate'] = pd.to_datetime(df_test['InvoiceDate'])
daily_test = df_test.groupby(df_test['InvoiceDate'].dt.date)['TotalSales'].sum()
daily_test.index = pd.to_datetime(daily_test.index)
daily_test = daily_test.asfreq('D').fillna(0)

# Usar auto_arima para encontrar el mejor modelo
stepwise_model = auto_arima(
    daily_train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,              # encuentra el mejor valor de d
    seasonal=False,      # cambia a True si hay estacionalidad clara
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# Mostrar los mejores parámetros encontrados
print("Mejor modelo ARIMA:", stepwise_model.order)

# Ajustar el modelo final con statsmodels
model = ARIMA(daily_train, order=stepwise_model.order)
model_fit = model.fit()

# Predecir
forecast = model_fit.forecast(steps=len(daily_test))

# Calcular métricas
mae = mean_absolute_error(daily_test, forecast)
rmse = mean_squared_error(daily_test, forecast)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Visualizar
plt.figure(figsize=(14, 6))
plt.plot(daily_test, label='Ventas reales')
plt.plot(forecast, label='Predicción ARIMA', linestyle='--')
plt.title('Predicción vs Realidad con ARIMA (auto)')
plt.xlabel('Fecha')
plt.ylabel('Ventas (€)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()