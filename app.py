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

# --- VISUALIZACIONES ---
st.subheader("📈 Análisis Visual")

# 1. Evolución del OEE
st.markdown("### Evolución del OEE")
fig1 = px.line(df, x="Fecha", y="OEE", color="Referencia",
               title="OEE vs Fecha")
st.plotly_chart(fig1, use_container_width=True)

# 2. Evolución del Scrap
st.markdown("### Scrap (%)")
fig2 = px.line(df, x="Fecha", y="Scrap_%", color="Referencia",
               title="Scrap vs Fecha")
st.plotly_chart(fig2, use_container_width=True)

# 3. Media del Tamaño de lote por fecha
st.markdown("### Media del Tamaño de Lote")
df_lote = df.groupby("Fecha", as_index=False)["Tamaño_lote"].mean()
fig3 = px.line(df_lote, x="Fecha", y="Tamaño_lote",
               title="Media Tamaño de Lote vs Fecha")
st.plotly_chart(fig3, use_container_width=True)

# 4. Media del Inventario por fecha
st.markdown("### Media del Inventario")
df_inv = df.groupby("Fecha", as_index=False)["Inventario_días"].mean()
fig4 = px.line(df_inv, x="Fecha", y="Inventario_días",
               title="Media Inventario (días) vs Fecha")
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
