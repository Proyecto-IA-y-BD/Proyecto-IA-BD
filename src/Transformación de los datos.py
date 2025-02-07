TRANSFORMACIÓN DE DATOS CON GENERACIÓN DE NUEVAS VARIABLES


# Importación de bibliotecas necesarias
import pandas as pd # Manejo de datos con aplicaciones matemáticas estadísticas
from sklearn.model_selection import train_test_split # Dividir los datos
from sklearn.ensemble import RandomForestRegressor # Modelo Random Forest
from sklearn.metrics import classification_report, confusion_matrix # Evaluación del modelo

path ='/content/drive/MyDrive/Colab Notebooks/Datasets varios/mi_dataframe_limpio.csv'

# Cargamos el dataset (el archivo CSV)
df = pd.read_csv(path)

# Revisamos los primeros registros
df.head(10)

InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country
0	536365	85123A	WHITE HANGING HEART T-LIGHT HOLDER	6	2010-12-01 08:26:00	2.55	17850.0	United Kingdom
1	536365	71053	WHITE METAL LANTERN	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom
2	536365	84406B	CREAM CUPID HEARTS COAT HANGER	8	2010-12-01 08:26:00	2.75	17850.0	United Kingdom
3	536365	84029G	KNITTED UNION FLAG HOT WATER BOTTLE	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom
4	536365	84029E	RED WOOLLY HOTTIE WHITE HEART.	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom
5	536365	22752	SET 7 BABUSHKA NESTING BOXES	2	2010-12-01 08:26:00	7.65	17850.0	United Kingdom
6	536365	21730	GLASS STAR FROSTED T-LIGHT HOLDER	6	2010-12-01 08:26:00	4.25	17850.0	United Kingdom
7	536366	22633	HAND WARMER UNION JACK	6	2010-12-01 08:28:00	1.85	17850.0	United Kingdom
8	536366	22632	HAND WARMER RED POLKA DOT	6	2010-12-01 08:28:00	1.85	17850.0	United Kingdom
9	536367	84879	ASSORTED COLOUR BIRD ORNAMENT	32	2010-12-01 08:34:00	1.69	13047.0	United Kingdom



# Calculamos las ventas totales por cada transacción
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Crear una nueva columna 'TotalSales' como el valor de la venta en cada transacción
# Suponemos que la columna 'Price' corresponde al precio de cada transacción y 'Quantity' la cantidad vendida
df['TotalSales'] = df['UnitPrice'] * df['Quantity']

# Revisamos los primeros registros
df.head(10)


InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country	TotalSales
0	536365	85123A	WHITE HANGING HEART T-LIGHT HOLDER	6	2010-12-01 08:26:00	2.55	17850.0	United Kingdom	15.30
1	536365	71053	WHITE METAL LANTERN	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom	20.34
2	536365	84406B	CREAM CUPID HEARTS COAT HANGER	8	2010-12-01 08:26:00	2.75	17850.0	United Kingdom	22.00
3	536365	84029G	KNITTED UNION FLAG HOT WATER BOTTLE	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom	20.34
4	536365	84029E	RED WOOLLY HOTTIE WHITE HEART.	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom	20.34
5	536365	22752	SET 7 BABUSHKA NESTING BOXES	2	2010-12-01 08:26:00	7.65	17850.0	United Kingdom	15.30
6	536365	21730	GLASS STAR FROSTED T-LIGHT HOLDER	6	2010-12-01 08:26:00	4.25	17850.0	United Kingdom	25.50
7	536366	22633	HAND WARMER UNION JACK	6	2010-12-01 08:28:00	1.85	17850.0	United Kingdom	11.10
8	536366	22632	HAND WARMER RED POLKA DOT	6	2010-12-01 08:28:00	1.85	17850.0	United Kingdom	11.10
9	536367	84879	ASSORTED COLOUR BIRD ORNAMENT	32	2010-12-01 08:34:00	1.69	13047.0	United Kingdom	54.08

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
fecha_inicio_entrenamiento = '2010-12-01'
fecha_fin_entrenamiento = '2011-11-08'
fecha_inicio_prueba = '2011-11-09'
fecha_fin_prueba = '2011-12-09'

# Filtrar los datos de entrenamiento y prueba
train_data = df_daily_sales[(df_daily_sales['InvoiceDate'] >= fecha_inicio_entrenamiento) & (df_daily_sales['InvoiceDate'] <= fecha_fin_entrenamiento)]
test_data = df_daily_sales[(df_daily_sales['InvoiceDate'] >= fecha_inicio_prueba) & (df_daily_sales['InvoiceDate'] <= fecha_fin_prueba)]

         
PREPARACIÓN DE LAS CARACTERÍSTICAS Y LA VARIABLE OBJETO

# Creamos columnas para el año, mes, día, día de la semana y semana del año
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df.head(10)

InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country	TotalSales	Year	Month	Day	DayOfWeek
0	536365	85123A	WHITE HANGING HEART T-LIGHT HOLDER	6	2010-12-01 08:26:00	2.55	17850.0	United Kingdom	15.30	2010	12	1	2
1	536365	71053	WHITE METAL LANTERN	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom	20.34	2010	12	1	2
2	536365	84406B	CREAM CUPID HEARTS COAT HANGER	8	2010-12-01 08:26:00	2.75	17850.0	United Kingdom	22.00	2010	12	1	2
3	536365	84029G	KNITTED UNION FLAG HOT WATER BOTTLE	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom	20.34	2010	12	1	2
4	536365	84029E	RED WOOLLY HOTTIE WHITE HEART.	6	2010-12-01 08:26:00	3.39	17850.0	United Kingdom	20.34	2010	12	1	2
5	536365	22752	SET 7 BABUSHKA NESTING BOXES	2	2010-12-01 08:26:00	7.65	17850.0	United Kingdom	15.30	2010	12	1	2
6	536365	21730	GLASS STAR FROSTED T-LIGHT HOLDER	6	2010-12-01 08:26:00	4.25	17850.0	United Kingdom	25.50	2010	12	1	2
7	536366	22633	HAND WARMER UNION JACK	6	2010-12-01 08:28:00	1.85	17850.0	United Kingdom	11.10	2010	12	1	2
8	536366	22632	HAND WARMER RED POLKA DOT	6	2010-12-01 08:28:00	1.85	17850.0	United Kingdom	11.10	2010	12	1	2
9	536367	84879	ASSORTED COLOUR BIRD ORNAMENT	32	2010-12-01 08:34:00	1.69	13047.0	United Kingdom	54.08	2010	12	1	2

         
# Selección de características (X) y variable objetivo (y)
X_train = df[['DayOfWeek', 'Day', 'Month', 'Year']]
y_train = df['TotalSales']
X_test = df[['DayOfWeek', 'Day', 'Month', 'Year']]
y_test = df['TotalSales']


ENTRENAMIENTO DEL MODELO

# Graficar las predicciones vs los valores reales
plt.figure(figsize=(10, 6))
plt.plot(df['InvoiceDate'], y_test, label='Valores reales', color='blue')
plt.plot(df['InvoiceDate'], y_pred, label='Predicciones', color='red', linestyle='--')
plt.title('Predicción de ventas diarias')
plt.xlabel('Fecha')
plt.ylabel('Ventas Totales')
plt.legend()
plt.show()






         

