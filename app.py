import streamlit as st
import pandas as pd
import sqlalchemy as db
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Conexión a la base de datos
engine = db.create_engine('sqlite:///inventario.db')
connection = engine.connect()
metadata = db.MetaData()

# Cargar tablas
ventas = db.Table('ventas', metadata, autoload_with=engine)
inventario = db.Table('inventario', metadata, autoload_with=engine)

# Consultas
query_ventas = db.select([ventas])
query_inventario = db.select([inventario])

# Convertir a DataFrames
df_ventas = pd.read_sql(query_ventas, connection)
df_inventario = pd.read_sql(query_inventario, connection)

# Título de la aplicación
st.title('Optimización de Inventarios')

# Visualización de datos
st.header('Datos de Ventas')
st.write(df_ventas)

st.header('Datos de Inventario')
st.write(df_inventario)

# Análisis de patrones de compra
st.header('Análisis de Patrones de Compra')

# Ventas por categoría
st.subheader('Ventas por Categoría')
ventas_por_categoria = df_ventas.groupby('categoria')['cantidad'].sum()
st.bar_chart(ventas_por_categoria)

# Ventas por producto
st.subheader('Ventas por Producto')
ventas_por_producto = df_ventas.groupby('producto')['cantidad'].sum()
st.bar_chart(ventas_por_producto)

# Análisis de series temporales
st.subheader('Análisis de Series Temporales')
df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
ventas_por_fecha = df_ventas.groupby('fecha_venta')['cantidad'].sum()

# Descomposición de la serie temporal
decomposition = seasonal_decompose(ventas_por_fecha, model='additive', period=30)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
decomposition.trend.plot(ax=ax1, title='Tendencia')
decomposition.seasonal.plot(ax=ax2, title='Estacionalidad')
decomposition.resid.plot(ax=ax3, title='Residuos')
ventas_por_fecha.plot(ax=ax4, title='Ventas Totales')
plt.tight_layout()
st.pyplot(fig)

# Análisis ABC de inventario
st.subheader('Análisis ABC de Inventario')
ventas_por_producto = df_ventas.groupby('producto')['cantidad'].sum().sort_values(ascending=False)
ventas_por_producto['contribucion'] = ventas_por_producto.cumsum() / ventas_por_producto.sum() * 100
ventas_por_producto['categoria'] = pd.cut(
    ventas_por_producto['contribucion'],
    bins=[0, 80, 95, 100],
    labels=['A', 'B', 'C']
)

# Gráfico de Pareto
fig, ax1 = plt.subplots()
ax1.bar(ventas_por_producto.index, ventas_por_producto['cantidad'], color='b')
ax1.set_ylabel('Ventas Totales', color='b')
ax2 = ax1.twinx()
ax2.plot(ventas_por_producto.index, ventas_por_producto['contribucion'], color='r', marker='o')
ax2.set_ylabel('Contribución Acumulada (%)', color='r')
plt.xticks(rotation=90)
st.pyplot(fig)

# Análisis de rotación de inventario
st.subheader('Rotación de Inventario')
inventario_promedio = df_inventario.groupby('producto')['stock_actual'].mean()
rotacion_inventario = ventas_por_producto['cantidad'] / inventario_promedio
rotacion_inventario = rotacion_inventario.sort_values(ascending=False)
st.bar_chart(rotacion_inventario)

# Puntos de reorden y stock de seguridad
st.subheader('Puntos de Reorden y Stock de Seguridad')
tiempo_entrega = st.slider('Tiempo de Entrega (días)', 1, 14, 7)
nivel_servicio = st.slider('Nivel de Servicio', 0.90, 0.99, 0.95)
z = norm.ppf(nivel_servicio)

demanda_promedio = df_ventas.groupby('producto')['cantidad'].mean()
desviacion_demanda = df_ventas.groupby('producto')['cantidad'].std()

stock_seguridad = z * desviacion_demanda * np.sqrt(tiempo_entrega)
punto_reorden = demanda_promedio * tiempo_entrega + stock_seguridad

df_reorden = pd.DataFrame({
    'Stock Actual': df_inventario.groupby('producto')['stock_actual'].mean(),
    'Punto de Reorden': punto_reorden,
    'Stock de Seguridad': stock_seguridad
})
st.bar_chart(df_reorden)

# Predicción de demanda
st.subheader('Predicción de Demanda')
modelo = ARIMA(ventas_por_fecha, order=(1, 1, 1))
resultados = modelo.fit()
predicciones = resultados.predict(start=len(ventas_por_fecha), end=len(ventas_por_fecha) + 30)

fig, ax = plt.subplots()
ventas_por_fecha.plot(ax=ax, label='Histórico')
predicciones.plot(ax=ax, label='Predicciones')
plt.legend()
st.pyplot(fig)

# Cerrar conexión
connection.close()
