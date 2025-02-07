
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

Quantity	UnitPrice	CustomerID
count	401604.000000	401604.000000	401604.000000
mean	13.542995	3.474064	15281.160818
std	250.213145	69.764035	1714.006089
min	1.000000	0.000000	12346.000000
25%	2.000000	1.250000	13939.000000
50%	6.000000	1.950000	15145.000000
75%	12.000000	3.750000	16784.000000
max	80995.000000	38970.000000	18287.0000

