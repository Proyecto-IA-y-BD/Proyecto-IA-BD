import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("../data/data_set_ecommerce.csv", encoding='ISO-8859-1')

# Generar reporte de para ahcer la exploración inicial de los datos
# profile = ProfileReport(df, title = "E-Commerce DataSet Profiling Report", explorative = True)
# profile.to_file("../reports/data_set_ecommerce.html", False)