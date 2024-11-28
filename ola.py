import numpy as np
import pandas as pd

# datos
data = pd.read_csv("student-mat.csv", sep = ';')
G1 = data['G1'].to_numpy()
G2 = data['G2'].to_numpy()
G3 = data['G3'].to_numpy()

# definir x e y
n = len(G3) 
X_0 = np.ones(n)
print(X_0)
X = np.column_stack((X_0, G1, G2))
Y = G3

# resolver X^T X e X^T Y
XT_X = np.zeros(2)
for i in range(2):
    for j in range(2):
        XT_X[i][j] = sum(X[:, i] * X[:, j])

XT_Y = np.zeros(3)  # Inicializar vector 3x1
for i in range(3):  # Calcular cada entrada
    XT_Y[i] = sum(X[:, i] * Y)

# X^T X * beta = X^T Y
beta = np.linalg.solve(XT_X, XT_Y)

# Mostrar los coeficientes
print("Coeficientes del modelo (β):")
print(beta)

# 6. Calcular predicciones y ECM
Y_pred = np.zeros(n)  # Inicializar las predicciones
for i in range(n):
    Y_pred[i] = beta[0] + beta[1] * G1[i] + beta[2] * G2[i]  # β0 + β1*G1 + β2*G2

# Calcular ECM
ecm = sum((Y - Y_pred) ** 2) / n
print("Error Cuadrático Medio (ECM):", ecm)
