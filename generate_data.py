import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# Configuración de la base de datos
engine = create_engine('sqlite:///inventario.db')

# Generar datos aleatorios
np.random.seed(42)
n = 300  # Número de registros

# Datos de productos
productos = [f'Producto_{i}' for i in range(1, 11)]
categorias = ['Electrónica', 'Hogar', 'Ropa', 'Juguetes', 'Alimentos']

# Datos de ventas
ventas_data = {
    'id_venta': range(1, n+1),
    'producto': np.random.choice(productos, n),
    'categoria': np.random.choice(categorias, n),
    'cantidad': np.random.randint(1, 50, n),
    'fecha_venta': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n)]
}

# Datos de inventario
inventario_data = {
    'id_inventario': range(1, n+1),
    'producto': np.random.choice(productos, n),
    'stock_actual': np.random.randint(0, 100, n),
    'stock_minimo': np.random.randint(10, 30, n),
    'fecha_actualizacion': [datetime.now() - timedelta(days=np.random.randint(1, 30)) for _ in range(n)]
}

# Convertir a DataFrames
df_ventas = pd.DataFrame(ventas_data)
df_inventario = pd.DataFrame(inventario_data)

# Guardar en la base de datos
df_ventas.to_sql('ventas', con=engine, if_exists='replace', index=False)
df_inventario.to_sql('inventario', con=engine, if_exists='replace', index=False)

print("Datos generados y guardados en la base de datos.")
