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


# Convertimos 'InvoiceDate' a formato datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Creamos columnas para el año, mes, día, día de la semana y semana del año
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek


# Revisamos los primeros registros
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



# Preparamos las variables con los datos que vamos a utilizar
X = df[['Year', 'Month', 'Day', 'DayOfWeek', 'Quantity', 'UnitPrice']] # Variables de entrada
y = df['TotalSales'] # Variable de salida


 # Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Datos divididos en entrenamiento y prueba")

Datos divididos en entrenamiento y prueba

CREACIÓN Y ENTRENAMIENTO DEL MODELO

 model = RandomForestRegressor(n_estimators=100, random_state=42) # Inicializar y entrenar el modelo de RandomForestRegressor
model.fit(X_train, y_train) # Entrenamos el modelo

y_pred = model.predict(X_test) # Realizar predicciones sobre el conjunto de prueba

print("Modelo entrenado")
Modelo entrenado


      
 














