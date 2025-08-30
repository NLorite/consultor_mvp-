import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Consultor AI de Producción", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- CARGAR DATOS ---
@st.cache_data
def load_data():
    return pd.read_excel("template_sample.xlsx", engine="openpyxl")

df = load_data()

st.title("📊 Consultor AI para Producción")
st.markdown("Dashboard interactivo + Chatbot consultor para análisis de datos industriales.")

# --- DASHBOARD ---
st.subheader("Indicadores Clave de Producción")

col1, col2, col3 = st.columns(3)
col1.metric("Promedio OEE", round(df["OEE"].mean(), 2))
col2.metric("Inventario medio (días)", round(df["Inventario_días"].mean(), 2))
col3.metric("Scrap medio (%)", round(df["Scrap_%"].mean(), 2))

# --- VISUALIZACIONES ---
st.subheader("📈 Análisis Visual")

tab1, tab2, tab3, tab4 = st.tabs(["OEE vs Lote", "Evolución OEE", "Inventario", "Scrap"])

with tab1:
    fig1 = px.scatter(df, x="Tamaño_lote", y="OEE", color="Referencia",
                      trendline="ols", title="Tamaño de lote vs OEE")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.line(df, x="Fecha", y="OEE", color="Referencia",
                   title="Evolución del OEE")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.line(df, x="Fecha", y="Inventario_días", color="Referencia",
                   title="Inventario en días")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = px.bar(df, x="Fecha", y="Scrap_%", color="Referencia",
                  title="Scrap (%) por referencia")
    st.plotly_chart(fig4, use_container_width=True)

# --- CHATBOT ---
st.subheader("🤖 Consultor Virtual")

user_input = st.text_area("Escribe tu pregunta sobre los datos:")

if user_input:
    # Resumen de datos para no sobrecargar tokens
    data_preview = df.head(50).to_csv(index=False)

    prompt = f"""
    Eres un consultor experto en operaciones industriales.
    Estos son ejemplos de datos de producción (primeras 50 filas):

    {data_preview}

    Pregunta del usuario:
    {user_input}

    Responde con un análisis claro y recomendaciones prácticas (ejemplo: mejorar OEE, reducir inventario, scrap, lead time).
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un consultor industrial experto en Lean Manufacturing, productividad y eficiencia."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        st.markdown(f"**Respuesta del consultor:**\n\n{answer}")
    except Exception as e:
        st.error(f"Error llamando a OpenAI: {e}")
