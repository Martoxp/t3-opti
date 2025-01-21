import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Leer el archivo CSV y extraer datos manualmente
with open('student-mat.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    G1 = []
    G2 = []
    G3 = []
    
    for row in reader:
        G1.append(float(['G1']))
        G2.append(float(row['G2']))
        G3.append(float(row['G3']))

# Preparar las variables independientes (X) y dependiente (y)
X = [[g1, g2] for g1, g2 in zip(G1, G2)]
y = G3

# Crear y ajustar el modelo
model = LinearRegression()
model.fit(X, y)

# Obtener coeficientes e intercepto
coeficientes = model.coef_
intercepto = model.intercept_

# Calcular el ECM
y_pred = model.predict(X)
ecm = mean_squared_error(y, y_pred)

# Mostrar resultados en consola
print("Coeficientes de los regresores:", coeficientes)
print("Intercepto (β0):", intercepto)
print("Error Cuadrático Medio (ECM):", ecm)
