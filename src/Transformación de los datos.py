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


