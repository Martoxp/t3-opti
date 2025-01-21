import numpy as np
import pandas as pd

data = pd.read_csv("student-mat.csv", sep=';')
X = data[['G1', 'G2']].to_numpy()
Y = data['G3'].to_numpy()

# creación de la matriz de los datos que usaremos
n = X.shape[0]
X = np.hstack((np.ones((n, 1)), X)) 

# cálculo de las matrices y β*
XtX = np.dot(X.T, X)
XtY = np.dot(X.T, Y)
beta = np.dot(np.linalg.inv(XtX), XtY)

# predicciones
Y_pred = np.dot(X, beta)
# ECM 3.1
ecm = np.sum((Y - Y_pred)**2) / n

# mostrar en consola
print(f"Intercepto β0: {beta[0]}") #B0
print(f"Coeficiente β1: {beta[1]}") #B1
print(f"Coeficiente β2: {beta[2]}") #B2
print("ECM:", ecm) #ecm

