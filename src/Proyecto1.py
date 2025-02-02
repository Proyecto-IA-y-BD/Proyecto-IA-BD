
#Proyecto de RANDOM FOREST
# Importación de bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



[2]
path= "/content/drive/MyDrive/Colab Notebooks/Datasets varios/ecommerce_data.csv"

# Cargar el dataset (el archivo CSV)
df = pd.read_csv(path)

#Revisar los primeros registros
df.head(10)


[3]
# Mostrar información sobre las columnas
print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 541909 entries, 0 to 541908
Data columns (total 8 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   InvoiceNo    541909 non-null  object 
 1   StockCode    541909 non-null  object 
 2   Description  540455 non-null  object 
 3   Quantity     541909 non-null  int64  
 4   InvoiceDate  541909 non-null  object 
 5   UnitPrice    541909 non-null  float64
 6   CustomerID   406829 non-null  float64
 7   Country      541909 non-null  object 
dtypes: float64(2), int64(1), object(5)
memory usage: 33.1+ MB
None

[4]
# Examinar valores únicos en columnas relevantes
for col in df.columns:
    print(f"\nValores únicos en {col}:")
    print(df[col].unique()[:5])  # Mostrar los primeros 5 valores únicos

Valores únicos en InvoiceNo:
['536365' '536366' '536367' '536368' '536369']

Valores únicos en StockCode:
['85123A' '71053' '84406B' '84029G' '84029E']

Valores únicos en Description:
['WHITE HANGING HEART T-LIGHT HOLDER' 'WHITE METAL LANTERN'
 'CREAM CUPID HEARTS COAT HANGER' 'KNITTED UNION FLAG HOT WATER BOTTLE'
 'RED WOOLLY HOTTIE WHITE HEART.']

Valores únicos en Quantity:
[ 6  8  2 32  3]

Valores únicos en InvoiceDate:
['2010-12-01 08:26:00' '2010-12-01 08:28:00' '2010-12-01 08:34:00'
 '2010-12-01 08:35:00' '2010-12-01 08:45:00']

Valores únicos en UnitPrice:
[2.55 3.39 2.75 7.65 4.25]

Valores únicos en CustomerID:
[17850. 13047. 12583. 13748. 15100.]

Valores únicos en Country:
['United Kingdom' 'France' 'Australia' 'Netherlands' 'Germany']
Haz doble clic (o pulsa Intro) para editar


[5]
# Ver los valores únicos de cada columna
print(df.nunique())

InvoiceNo      25900
StockCode       4070
Description     4223
Quantity         722
InvoiceDate    23260
UnitPrice       1630
CustomerID      4372
Country           38
dtype: int64

[6]
# Descripción estadística de los datos numéricos
print(df.describe())
            Quantity      UnitPrice     CustomerID
count  541909.000000  541909.000000  406829.000000
mean        9.552250       4.611114   15287.690570
std       218.081158      96.759853    1713.600303
min    -80995.000000  -11062.060000   12346.000000
25%         1.000000       1.250000   13953.000000
50%         3.000000       2.080000   15152.000000
75%        10.000000       4.130000   16791.000000
max     80995.000000   38970.000000   18287.000000

[7]
# Verificar valores nulos
print(df.isnull().sum())

InvoiceNo           0
StockCode           0
Description      1454
Quantity            0
InvoiceDate         0
UnitPrice           0
CustomerID     135080
Country             0
dtype: int64

[8]
# Identificar registros con valores nulos en columnas específicas
print(df[df['CustomerID'].isnull()])

       InvoiceNo StockCode                      Description  Quantity  \
622       536414     22139                              NaN        56   
1443      536544     21773  DECORATIVE ROSE BATHROOM BOTTLE         1   
1444      536544     21774  DECORATIVE CATS BATHROOM BOTTLE         2   
1445      536544     21786               POLKADOT RAIN HAT          4   
1446      536544     21787            RAIN PONCHO RETROSPOT         2   
...          ...       ...                              ...       ...   
541536    581498    85099B          JUMBO BAG RED RETROSPOT         5   
541537    581498    85099C   JUMBO  BAG BAROQUE BLACK WHITE         4   
541538    581498     85150    LADIES & GENTLEMEN METAL SIGN         1   
541539    581498     85174                S/4 CACTI CANDLES         1   
541540    581498       DOT                   DOTCOM POSTAGE         1   

                InvoiceDate  UnitPrice  CustomerID         Country  
622     2010-12-01 11:52:00       0.00         NaN  United Kingdom  
1443    2010-12-01 14:32:00       2.51         NaN  United Kingdom  
1444    2010-12-01 14:32:00       2.51         NaN  United Kingdom  
1445    2010-12-01 14:32:00       0.85         NaN  United Kingdom  
1446    2010-12-01 14:32:00       1.66         NaN  United Kingdom  
...                     ...        ...         ...             ...  
541536  2011-12-09 10:26:00       4.13         NaN  United Kingdom  
541537  2011-12-09 10:26:00       4.13         NaN  United Kingdom  
541538  2011-12-09 10:26:00       4.96         NaN  United Kingdom  
541539  2011-12-09 10:26:00      10.79         NaN  United Kingdom  
541540  2011-12-09 10:26:00    1714.17         NaN  United Kingdom  

[135080 rows x 8 columns]

[15]
df = df.dropna(subset=['CustomerID', 'Description']) # Eliminar valores nulos de CustomerID y Description

[9]
df = df.drop_duplicates()
print("\nDimensiones del dataset tras eliminar los duplicados:", df.shape)


Dimensiones del dataset tras eliminar los duplicados: (536641, 8)

[16]
df = df.dropna(subset=['CustomerID', 'Description']) # Eliminar valores nulos de CustomerID y Description
# Verificar si existen valores nulos después de eliminarlos
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
# Manejo de outliers: verificar valores atípicos en Quantity y UnitPrice: valores negativos o cero
df[df['Quantity'] <= 0]  # Filtrar transacciones con cantidades no válidas
df[df['UnitPrice'] <= 0]  # Filtrar transacciones con precios no válidos

df['Quantity'] = df['Quantity'].abs()
df['UnitPrice'] = df['UnitPrice'].abs()

print("Valores negativos en 'Quantity':", (df['Quantity'] < 0).sum())
print("Valores negativos en 'UnitPrice':", (df['UnitPrice'] < 0).sum())

Valores negativos en 'Quantity': 0
Valores negativos en 'UnitPrice': 0

[18]
print("\nDimensiones del dataset tras eliminar valores atípicos (outliers):", df.shape)

Dimensiones del dataset tras eliminar valores atípicos (outliers): (401604, 8)

[19]
# Información sobre columnas y tipos de datos tras eliminar duplicados y valores atípicos (outliers)
df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 401604 entries, 0 to 541908
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
memory usage: 27.6+ MB

[23]
# Descripción estadística de datos numéricos
df.describe()

                Quantity	        UnitPrice	           CustomerID
count	       401604.000000	   401604.000000	       401604.000000
mean	           12.183273	        3.474064	        15281.160818
std	            250.283037	       69.764035	         1714.006089
min	         -80995.000000	        0.000000	        12346.000000
25%	              2.000000	        1.250000	        13939.000000
50%	              5.000000	        1.950000	        15145.000000
75%	             12.000000	        3.750000	        16784.000000
