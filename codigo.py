import numpy as np
import pandas as pd

data = pd.read_csv("student-mat.csv", sep =';')


X = data[['G1', 'G2']].to_numpy()
Y = data['G3'].to_numpy().reshape(-1, 1)

# Agregar una columna de unos a X para el intercepto (beta_0)
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Agregar columna de unos al inicio

# Calcular beta*
beta = np.linalg.inv(X.T * X) @ (X.T * Y)

# Mostrar los coeficientes
print("Coeficientes del modelo (β):")
print(beta)

# Predicciones del modelo
Y_pred = X @ beta

# Calcular el Error Cuadrático Medio
n = len(Y)
ecm = (1 / n) * np.sum((Y - Y_pred) ** 2)

print("Error Cuadrático Medio (ECM):", ecm)
