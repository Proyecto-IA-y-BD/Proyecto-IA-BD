import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Cargar el dataset limpio
df = pd.read_csv("../data/train_data_set.csv", encoding='ISO-8859-1')

# Asegurarse de que los datos estén ordenados por fecha si es necesario
df['InvoiceDate'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.sort_values('InvoiceDate')

# Seleccionar la columna que se va a predecir, por ejemplo 'Quantity'
data = df.set_index('InvoiceDate')['Quantity']

# Dividir los datos en entrenamiento y prueba
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Entrenar el modelo ARIMA
model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) son los parámetros del modelo
model_fit = model.fit()

# Hacer predicciones para el conjunto de prueba
predictions = model_fit.forecast(steps=len(test))

# Calcular la precisión del modelo
mse = mean_squared_error(test, predictions)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse}')

# Hacer predicciones para los próximos 15 años
future_steps = 365 * 15  # Número de días en 15 años
future_predictions = model_fit.forecast(steps=future_steps)

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(data.index, data, label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.plot(pd.date_range(start=test.index[-1], periods=future_steps, freq='D'), future_predictions, label='Future Predictions', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Sales Forecast for the Next 15 Years')
plt.show()