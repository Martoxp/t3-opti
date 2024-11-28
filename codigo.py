import numpy as np
import pandas as pd

# Cargar los datos
data = pd.read_csv("student-mat.csv", sep = ';')

# Seleccionar las columnas G1, G2 como regresores y G3 como la variable dependiente
x = data[['G1', 'G2']].to_numpy()  # Matriz de regresores
y = data['G3'].to_numpy()  # Vector de la variable dependiente

# Agregar una columna de unos a X para el intercepto (beta_0)
x = np.hstack((np.ones((x.shape[0], 1)), x))  # Agregar columna de unos al inicio

# Calcular beta*
B = np.linalg.inv(x.T @ x) @ (x.T @ y)

# Mostrar los coeficientes
print("Coeficientes del modelo (β):")
print(B)

# Predicciones del modelo
y_pred = x @ B

# Calcular el Error Cuadrático Medio
n = len(y)
ecm = (1 / n) * np.sum((y - y_pred) ** 2)

print("ECM = ", ecm)
