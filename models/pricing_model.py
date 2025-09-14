# insurance-pricing-model/models/pricing_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Cargar Datos Sintéticos
print("Cargando datos...")
data = pd.read_csv('data/synthetic_insurance_data.csv')

# 2. Exploración Rápida de Datos
print("\nPrimeras 5 filas de los datos:")
print(data.head())
print(f"\nDimensiones de los datos: {data.shape}")

# 3. Preprocesamiento
# Definir variables (features) y variable objetivo (target)
# X: Características del asegurado y el vehículo
# y: Costo total de siniestros (lo que queremos predecir para calcular la prima)
X = data[['age', 'gender', 'driving_experience', 'vehicle_age', 'annual_mileage']]
y = data['total_claims_cost']

# 4. Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nDatos de entrenamiento: {X_train.shape}, Datos de prueba: {X_test.shape}")

# 5. Entrenar el Modelo (Regresión Lineal)
print("\nEntrenando el modelo de regresión lineal...")
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Hacer Predicciones y Evaluar el Modelo
print("\nEvaluando el modelo...")
y_pred = model.predict(X_test)

# Calcular métricas de error
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error Absoluto Medio (MAE): ${mae:.2f}')
print(f'Coeficiente de Determinación (R²): {r2:.4f}')

# 7. Visualización de Resultados (Opcional pero muy bueno para mostrar)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Línea de perfecta predicción
plt.xlabel('Costo Real de Siniestros')
plt.ylabel('Costo Predicho de Siniestros')
plt.title('Modelo de Pricing: Real vs. Predicho')
plt.grid(True)
plt.savefig('results/predictions_vs_actuals.png') # Guarda la gráfica
plt.show()

# 8. Ejemplo de Cálculo de Prima para un Nuevo Cliente
print("\n--- Ejemplo de Cálculo de Prima ---")
# Supongamos que un nuevo cliente tiene estas características:
new_customer = np.array([[35, 1, 10, 5, 12000]]) # [age, gender (1=M), experience, vehicle_age, mileage]

# Predecir el costo esperado de siniestros para este cliente
predicted_cost = model.predict(new_customer)[0]
# Una prima simple podría ser el costo predicho + un 20% para gastos y ganancia
insurance_premium = predicted_cost * 1.20

print(f"Para el nuevo cliente, el costo de siniestros predicho es: ${predicted_cost:.2f}")
print(f"La prima de seguro sugerida es: ${insurance_premium:.2f}")
Minu