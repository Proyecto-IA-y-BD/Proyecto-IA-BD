import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

df = pd.read_csv("../data/data_set_ecommerce.csv", encoding='ISO-8859-1')

# Generar reporte de para ahcer la exploración inicial de los datos
# profile = ProfileReport(df, title = "E-Commerce DataSet Profiling Report", explorative = True)
# profile.to_file("../reports/data_set_ecommerce.html", False)

# Eliminar registros con valores negativos en el campo Quantity cuyo campo invoceNo no empiece con 'C'
df = df[~((df['Quantity'] < 0) & (~df['InvoiceNo'].str.startswith('C')))]

# Rellenar campos nulos o vacíos en la columna Description
df['Description'] = df['Description'].fillna('No Description')

# Cambiar valores negativos de UnitPrice a positivos y asegurar que todos sean números reales con decimales
df['UnitPrice'] = df['UnitPrice'].abs().astype(float)

# Rellenar CustomerID vacíos con un número aleatorio de 5 dígitos
df['CustomerID'] = df['CustomerID'].apply(lambda x: np.random.randint(10000, 99999) if pd.isnull(x) else x)

# Guardar el dataset limpio en un nuevo archivo
df.to_csv("../data/data_set_clean.csv", index = False)

# Convertimos `InvoiceDate` a formato datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Creamos columnas para el año, mes, día
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day

df.to_csv("../data/train_data_set.csv", index = False)
