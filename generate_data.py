# generate_data.py
import pandas as pd
import numpy as np

# Configurar semilla para resultados reproducibles
np.random.seed(42)
n_samples = 1000

# Generar datos sintéticos
data = {
    'age': np.random.randint(18, 70, n_samples),
    'gender': np.random.randint(0, 2, n_samples), # 0: F, 1: M
    'driving_experience': np.random.randint(1, 50, n_samples),
    'vehicle_age': np.random.randint(0, 20, n_samples),
    'annual_mileage': np.random.randint(5000, 25000, n_samples),
}

# Crear el costo de siniestros como una función de las variables anteriores + algo de ruido
# Esta es la "verdad" que el modelo intentará aprender
data['total_claims_cost'] = (
    100 * data['age'] +
    -50 * data['gender'] +
    20 * data['driving_experience'] +
    30 * data['vehicle_age'] +
    0.5 * data['annual_mileage'] +
    np.random.normal(0, 500, n_samples) # Ruido aleatorio
)

# Crear DataFrame y guardar como CSV
df = pd.DataFrame(data)
df.to_csv('data/synthetic_insurance_data.csv', index=False)
print("Datos sintéticos generados y guardados en 'data/synthetic_insurance_data.csv'")