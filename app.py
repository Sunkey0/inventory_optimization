import streamlit as st
import pandas as pd
import sqlalchemy as db
from sqlalchemy import select, Table, MetaData
import plotly.express as px
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Conexión a la base de datos
engine = db.create_engine('sqlite:///inventario.db')
connection = engine.connect()
metadata = MetaData()

# Verificar si las tablas existen
try:
    # Cargar tablas
    ventas = Table('ventas', metadata, autoload_with=engine)
    inventario = Table('inventario', metadata, autoload_with=engine)

    # Consultas
    query_ventas = select(ventas)
    query_inventario = select(inventario)

    # Convertir a DataFrames
    df_ventas = pd.read_sql(query_ventas, connection)
    df_inventario = pd.read_sql(query_inventario, connection)

except db.exc.NoSuchTableError as e:
    st.error(f"Error: {e}. Asegúrate de que las tablas 'ventas' e 'inventario' existen en la base de datos.")
    st.stop()

# Título de la aplicación
st.title('Optimización de Inventarios')

# Menú desplegable para seleccionar DataFrame
st.sidebar.header("Menú de DataFrames")
dataframe_seleccionado = st.sidebar.selectbox(
    "Selecciona un DataFrame:",
    ["Ventas", "Inventario"]
)

# Mostrar el DataFrame seleccionado
if dataframe_seleccionado == "Ventas":
    st.header('Datos de Ventas')
    st.write(df_ventas)
else:
    st.header('Datos de Inventario')
    st.write(df_inventario)

# Pestañas para análisis y explicaciones
tab1, tab2 = st.tabs(["Análisis", "Explicaciones"])

with tab1:
    # Análisis de patrones de compra
    st.header('Análisis de Patrones de Compra')

    # Ventas por categoría
    st.subheader('Ventas por Categoría')
    ventas_por_categoria = df_ventas.groupby('categoria')['cantidad'].sum().reset_index()
    fig = px.bar(ventas_por_categoria, x='categoria', y='cantidad', title='Ventas por Categoría')
    st.plotly_chart(fig)

    # Ventas por producto
    st.subheader('Ventas por Producto')
    ventas_por_producto = df_ventas.groupby('producto')['cantidad'].sum().reset_index()
    fig = px.bar(ventas_por_producto, x='producto', y='cantidad', title='Ventas por Producto')
    st.plotly_chart(fig)

    # Análisis de series temporales
    st.subheader('Análisis de Series Temporales')
    df_ventas['fecha_venta'] = pd.to_datetime(df_ventas['fecha_venta'])
    ventas_por_fecha = df_ventas.groupby('fecha_venta')['cantidad'].sum().reset_index()

    # Gráfico de series temporales
    fig = px.line(ventas_por_fecha, x='fecha_venta', y='cantidad', title='Ventas a lo Largo del Tiempo')
    st.plotly_chart(fig)

    # Descomposición de la serie temporal
    st.subheader('Descomposición de la Serie Temporal')
    decomposition = seasonal_decompose(ventas_por_fecha.set_index('fecha_venta')['cantidad'], model='additive', period=30)
    fig = px.line(decomposition.trend, title='Tendencia de Ventas')
    st.plotly_chart(fig)

    # Análisis ABC de inventario
    st.subheader('Análisis ABC de Inventario')
    ventas_por_producto = df_ventas.groupby('producto')['cantidad'].sum().reset_index()
    ventas_por_producto['contribucion'] = ventas_por_producto['cantidad'].cumsum() / ventas_por_producto['cantidad'].sum() * 100
    ventas_por_producto['categoria'] = pd.cut(
        ventas_por_producto['contribucion'],
        bins=[0, 80, 95, 100],
        labels=['A', 'B', 'C']
    )

    # Gráfico de Pareto
    fig = px.bar(ventas_por_producto, x='producto', y='cantidad', color='categoria', title='Análisis ABC de Inventario')
    st.plotly_chart(fig)

    # Análisis de rotación de inventario
    st.subheader('Rotación de Inventario')
    inventario_promedio = df_inventario.groupby('producto')['stock_actual'].mean().reset_index()
    rotacion_inventario = ventas_por_producto.merge(inventario_promedio, on='producto')
    rotacion_inventario['rotacion'] = rotacion_inventario['cantidad'] / rotacion_inventario['stock_actual']
    rotacion_inventario = rotacion_inventario.sort_values(by='rotacion', ascending=False)

    fig = px.bar(rotacion_inventario, x='producto', y='rotacion', title='Rotación de Inventario por Producto')
    st.plotly_chart(fig)

    # Puntos de reorden y stock de seguridad
    st.subheader('Puntos de Reorden y Stock de Seguridad')
    tiempo_entrega = st.slider('Tiempo de Entrega (días)', 1, 14, 7)
    nivel_servicio = st.slider('Nivel de Servicio', 0.90, 0.99, 0.95)
    z = norm.ppf(nivel_servicio)

    demanda_promedio = df_ventas.groupby('producto')['cantidad'].mean().reset_index()
    desviacion_demanda = df_ventas.groupby('producto')['cantidad'].std().reset_index()

    stock_seguridad = z * desviacion_demanda['cantidad'] * np.sqrt(tiempo_entrega)
    punto_reorden = demanda_promedio['cantidad'] * tiempo_entrega + stock_seguridad

    df_reorden = pd.DataFrame({
        'producto': demanda_promedio['producto'],
        'Stock Actual': df_inventario.groupby('producto')['stock_actual'].mean().values,
        'Punto de Reorden': punto_reorden,
        'Stock de Seguridad': stock_seguridad
    })

    fig = px.bar(df_reorden, x='producto', y=['Stock Actual', 'Punto de Reorden', 'Stock de Seguridad'],
                 title='Puntos de Reorden vs Stock Actual', barmode='group')
    st.plotly_chart(fig)

    # Predicción de demanda
    st.subheader('Predicción de Demanda')
    modelo = ARIMA(ventas_por_fecha.set_index('fecha_venta')['cantidad'], order=(1, 1, 1))
    resultados = modelo.fit()
    predicciones = resultados.predict(start=len(ventas_por_fecha), end=len(ventas_por_fecha) + 30)

    fig = px.line(predicciones, title='Predicción de Demanda')
    st.plotly_chart(fig)

with tab2:
    # Explicaciones de modelos, ecuaciones y métodos
    st.header('Explicaciones y Métodos')

    st.subheader('1. Análisis ABC')
    st.write("""
    El análisis ABC clasifica los productos en tres categorías según su contribución a las ventas:
    - **Categoría A**: Productos que generan el 80% de las ventas (20% de los productos).
    - **Categoría B**: Productos que generan el 15% de las ventas (30% de los productos).
    - **Categoría C**: Productos que generan el 5% de las ventas (50% de los productos).
    """)
    st.latex(r"""
    \text{Contribución Acumulada} = \frac{\text{Ventas Acumuladas}}{\text{Ventas Totales}} \times 100
    """)

    st.subheader('2. Punto de Reorden y Stock de Seguridad')
    st.write("""
    El **punto de reorden** es el nivel de inventario en el cual se debe realizar un nuevo pedido para evitar desabastecimientos.
    - **Fórmula del Punto de Reorden**:
    """)
    st.latex(r"""
    \text{ROP} = \text{Demanda Promedio} \times \text{Tiempo de Entrega} + \text{Stock de Seguridad}
    """)
    st.write("""
    - **Fórmula del Stock de Seguridad**:
    """)
    st.latex(r"""
    \text{SS} = Z \times \sigma \times \sqrt{\text{Tiempo de Entrega}}
    """)
    st.write("""
    Donde:
    - \( Z \): Valor crítico de la distribución normal para un nivel de servicio dado.
    - \( \sigma \): Desviación estándar de la demanda.
    """)

    st.subheader('3. Predicción de Demanda (SARIMA)')
    st.write("""
    El modelo **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) se utiliza para predecir la demanda futura basándose en patrones históricos.
    - **Ecuación General**:
    """)
    st.latex(r"""
    \text{SARIMA}(p, d, q)(P, D, Q)_s
    """)
    st.write("""
    Donde:
    - \( p \): Orden del componente autoregresivo.
    - \( d \): Orden de diferenciación.
    - \( q \): Orden del componente de media móvil.
    - \( P, D, Q \): Componentes estacionales.
    - \( s \): Periodicidad estacional.
    """)

    st.subheader('4. Interpretación del Dashboard')
    st.write("""
    - **Ventas por Categoría/Producto**: Muestra la distribución de ventas.
    - **Series Temporales**: Identifica tendencias y estacionalidad.
    - **Análisis ABC**: Clasifica productos según su importancia.
    - **Rotación de Inventario**: Evalúa la eficiencia del inventario.
    - **Punto de Reorden**: Ayuda a planificar pedidos.
    - **Predicción de Demanda**: Proyecta ventas futuras.
    """)

# Cerrar conexión
connection.close()
