import streamlit as st
import pandas as pd
import sqlalchemy as db
import matplotlib.pyplot as plt
import seaborn as sns

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
st.subheader('Ventas por Categoría')
ventas_por_categoria = df_ventas.groupby('categoria')['cantidad'].sum()
st.bar_chart(ventas_por_categoria)

st.subheader('Ventas por Producto')
ventas_por_producto = df_ventas.groupby('producto')['cantidad'].sum()
st.bar_chart(ventas_por_producto)

# Análisis de inventario
st.header('Análisis de Inventario')
st.subheader('Stock Actual vs Stock Mínimo')
fig, ax = plt.subplots()
sns.scatterplot(data=df_inventario, x='stock_actual', y='stock_minimo', ax=ax)
st.pyplot(fig)

# Cerrar conexión
connection.close()
