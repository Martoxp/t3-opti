import pandas as pd
from sklearn.linear_model import LinearRegression as RL
from sklearn.metrics import mean_squared_error

# Cargar la base de datos
# Asegúrate de tener el archivo CSV disponible, por ejemplo, 'student-mat.csv'
data_a = pd.read_csv('student-mat.csv', sep=';')
data = data_a[['G1', 'G2']]
G3 = data_a['G3']
# modelo
model = RL()
model.fit(data, G3)

#predecir G3
pred_G3 = model.predict(G3)
ECM = mean_squared_error(G3, pred_G3)

print(ECM)

# Usar solo las columnas G1, G2 y G3
#n_data = data[['G1', 'G2']]  # Variables independientes
#y = data['G3']          # Variable dependiente
#n_data.to_csv('independientes.csv')

# Crear y ajustar el modelo
#model = RL()
#model.fit(x, y)

# Obtener coeficientes e intercepto
#coeficientes = model.coef_
#intercepto = model.intercept_

# Calcular el ECM
#y_pred = model.predict(x)
#ecm = mean_squared_error(y, y_pred)

# Mostrar resultados en consola
#print("Coeficientes de los regresores:", coeficientes)
#print("Intercepto (β0):", intercepto)
#print("Error Cuadrático Medio (ECM):", ecm)
