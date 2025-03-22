# Proyecto de GAUSS MIXTURE MODEL
# Importación de bibliotecas necesarias

from sklearn.mixture import GaussianMixture

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.metrics import silhouette_score


path= "/content/drive/MyDrive/Colab Notebooks/Datasets varios/dataset_for_clustering.csv"

# Cargar el dataset (el archivo CSV)
df = pd.read_csv(path)

# Mostrar información sobre las columnas
print(df.info())


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 336697 entries, 0 to 336696
Data columns (total 15 columns):
 #   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   InvoiceNo    336697 non-null  object 
 1   StockCode    333502 non-null  object 
 2   Description  333502 non-null  object 
 3   Quantity     333502 non-null  float64
 4   InvoiceDate  333502 non-null  object 
 5   UnitPrice    333502 non-null  float64
 6   CustomerID   333502 non-null  float64
 7   Country      333502 non-null  object 
 8   TotalSales   333502 non-null  float64
 9   Year         333502 non-null  float64
 10  Month        333502 non-null  float64
 11  Day          333502 non-null  float64
 12  DayOfWeek    333502 non-null  float64
 13  Quarter      333502 non-null  float64
 14  Season       333502 non-null  object 
dtypes: float64(9), object(6)
memory usage: 38.5+ MB
None


mask = df['Country'] == "United Kingdom"
df = df[mask]
# Revisamos los primeros registros
df.head(10)

InvoiceNo 	StockCode 	Description 	Quantity 	InvoiceDate 	UnitPrice 	CustomerID 	Country 	TotalSales 	Year 	Month 	Day 	DayOfWeek 	Quarter 	Season
0 	536365 	85123A 	WHITE HANGING HEART T-LIGHT HOLDER 	6.0 	2010-12-01 08:26:00 	2.55 	17850.0 	United Kingdom 	15.30 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
1 	536365 	71053 	WHITE METAL LANTERN 	6.0 	2010-12-01 08:26:00 	3.39 	17850.0 	United Kingdom 	20.34 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
2 	536365 	84406B 	CREAM CUPID HEARTS COAT HANGER 	8.0 	2010-12-01 08:26:00 	2.75 	17850.0 	United Kingdom 	22.00 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
3 	536365 	84029G 	KNITTED UNION FLAG HOT WATER BOTTLE 	6.0 	2010-12-01 08:26:00 	3.39 	17850.0 	United Kingdom 	20.34 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
4 	536365 	84029E 	RED WOOLLY HOTTIE WHITE HEART. 	6.0 	2010-12-01 08:26:00 	3.39 	17850.0 	United Kingdom 	20.34 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
5 	536365 	22752 	SET 7 BABUSHKA NESTING BOXES 	2.0 	2010-12-01 08:26:00 	7.65 	17850.0 	United Kingdom 	15.30 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
6 	536365 	21730 	GLASS STAR FROSTED T-LIGHT HOLDER 	6.0 	2010-12-01 08:26:00 	4.25 	17850.0 	United Kingdom 	25.50 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
7 	536366 	22633 	HAND WARMER UNION JACK 	6.0 	2010-12-01 08:28:00 	1.85 	17850.0 	United Kingdom 	11.10 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
8 	536366 	22632 	HAND WARMER RED POLKA DOT 	6.0 	2010-12-01 08:28:00 	1.85 	17850.0 	United Kingdom 	11.10 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter
9 	536367 	22745 	POPPY'S PLAYHOUSE BEDROOM 	6.0 	2010-12-01 08:34:00 	2.10 	13047.0 	United Kingdom 	12.60 	2010.0 	12.0 	1.0 	2.0 	4.0 	Winter



# Configuración de datos

df_grouped = df.groupby(['InvoiceNo', 'CustomerID'])['TotalSales'].sum().reset_index()
df_grouped

df_to_use = df_grouped.groupby(['CustomerID']).agg({
  "TotalSales": "sum",
  "InvoiceNo": "count"
}).reset_index()

df_to_use



















