
2 - Procesamiento y análisis de dataset limpio - RANDOM FOREST.ipynb


[ ]
# PROCESAMIENTO y ANÁLISIS DE DATOS LIMPIOS

# Importación de biblioteca necesaria
import pandas as pd

path ='/content/drive/MyDrive/Colab Notebooks/Datasets varios/mi_dataframe_limpio.csv'

# Cargamos el dataset (el archivo CSV)
df = pd.read_csv(path)

# Revisamos los primeros registros
df.head(10)



[ ]
# Índice de columnas
df.columns
Index(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
       'UnitPrice', 'CustomerID', 'Country'],
      dtype='object')

[ ]
# Mostrar información sobre las columnas
print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 401604 entries, 0 to 401603
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   InvoiceNo    401604 non-null  object 
 1   StockCode    401604 non-null  object 
 2   Description  401604 non-null  object 
 3   Quantity     401604 non-null  int64  
 4   InvoiceDate  401604 non-null  object 
 5   UnitPrice    401604 non-null  float64
 6   CustomerID   401604 non-null  float64
 7   Country      401604 non-null  object 
dtypes: float64(2), int64(1), object(5)
memory usage: 24.5+ MB
None

[ ]
# Verificamos si tras la limpieza hay valores nulos
print(df.isnull().sum())
InvoiceNo      0
StockCode      0
Description    0
Quantity       0
InvoiceDate    0
UnitPrice      0
CustomerID     0
Country        0
dtype: int64

[ ]
# Verificamos si tras la limpieza hay filas duplicadas
print(df.duplicated().sum())
0

[ ]
# Descripción estadística de los datos numéricos
df.describe()


[ ]

Empieza a programar o a crear código con IA.
