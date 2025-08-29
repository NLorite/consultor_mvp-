# app.py
import os
import io
import tempfile
import sqlite3
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import openai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Consultor Digital Industrial - MVP", layout="wide")

# Lee API key desde variable de entorno o Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    # si ejecutas en Streamlit Cloud pon OPENAI_API_KEY en secrets o env
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.warning("OPENAI_API_KEY no configurada. El chatbot no funcionará hasta que la pongas (ver README).")

# Base de datos local simple (sqlite) para acciones
DB_PATH = "actions.db"

def init_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        reference TEXT,
        title TEXT,
        detail TEXT,
        owner TEXT,
        due_date TEXT,
        status TEXT
    )
    """)
    con.commit()
    return con

db = init_db()

# --------------------
# UTIL: KPIs, análisis, gráficos, PDF
# --------------------

def compute_kpis(df):
    """Devuelve un dict con KPIs agregados y un resumen por referencia."""
    out = {}
    numeric_cols = ["Tamaño_lote", "OEE", "Inventario_días", "Scrap_%", "LeadTime_días"]
    # Asegurar columnas
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    # Globales
    out["OEE_prom"] = float(df["OEE"].mean())
    out["Scrap_prom"] = float(df["Scrap_%"].mean())
    out["Inventario_prom"] = float(df["Inventario_días"].mean())
    out["N_registros"] = int(len(df))
    # Correlaciones básicas
    if df["Tamaño_lote"].notna().sum() > 1 and df["OEE"].notna().sum() > 1:
        out["corr_lote_oee"] = float(np.corrcoef(df["Tamaño_lote"].fillna(0), df["OEE"].fillna(0))[0,1])
    else:
        out["corr_lote_oee"] = None
    # resumen por referencia
    out["by_reference"] = df.groupby("Referencia").agg({
        "Tamaño_lote": "mean",
        "OEE": "mean",
        "Inventario_días": "mean",
        "Scrap_%": "mean",
        "LeadTime_días": "mean"
    }).reset_index().round(2)
    return out

def plot_trends(df):
    """Genera plotly charts para el dashboard."""
    plots = {}
    # Convertir fecha
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        df = df.sort_values("Fecha")
    # OEE trend
    if "Fecha" in df.columns and "OEE" in df.columns:
        fig_oee = px.line(df, x="Fecha", y="OEE", color="Referencia", markers=True, title="Evolución OEE por referencia")
        plots["oee_trend"] = fig_oee
    # Lote vs OEE scatter
    if "Tamaño_lote" in df.columns and "OEE" in df.columns:
        fig_scatter = px.scatter(df, x="Tamaño_lote", y="OEE", color="Referencia",
                                 trendline="ols", title="Tamaño de lote vs OEE (scatter + trendline)")
        plots["scatter_lote_oee"] = fig_scatter
    # Inventario days stacked
    if "Fecha" in df.columns and "Inventario_días" in df.columns:
        fig_inv = px.bar(df, x="Fecha", y="Inventario_días", color="Referencia", title="Inventario días por referencia")
        plots["inv_trend"] = fig_inv
    return plots

def generate_pdf(kpi_summary, df, recommendations_text, filename="informe.pdf"):
    """Genera un PDF con un resumen, una gráfica (scatter) y recomendaciones."""
    tmp_img = None
    # Crear gráfica scatter como imagen si hay datos
    try:
        if "Tamaño_lote" in df.columns and "OEE" in df.columns and df[["Tamaño_lote","OEE"]].dropna().shape[0] > 1:
            plt.figure(figsize=(6,4))
            plt.scatter(df["Tamaño_lote"], df["OEE"])
            plt.xlabel("Tamaño de lote")
            plt.ylabel("OEE (%)")
            corr = kpi_summary.get("corr_lote_oee", None)
            title = f"Correlación Lote - OEE = {corr:.2f}" if corr is not None else "Lote vs OEE"
            plt.title(title)
            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            plt.tight_layout()
            plt.savefig(tmp_img, dpi=150)
            plt.close()
    except Exception as e:
        print("No se pudo generar imagen:", e)
        tmp_img = None

    # Generar PDF con reportlab
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Informe de Consultoría Industrial - MVP", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # KPIs resumen
    story.append(Paragraph("Resumen de KPIs:", styles["Heading2"]))
    text_kpis = f"OEE promedio: {kpi_summary.get('OEE_prom', 'N/A'):.2f} %<br/>"
    text_kpis += f"Scrap promedio: {kpi_summary.get('Scrap_prom', 'N/A'):.2f} %<br/>"
    text_kpis += f"Inventario (días) promedio: {kpi_summary.get('Inventario_prom', 'N/A'):.2f} <br/>"
    if kpi_summary.get('corr_lote_oee') is not None:
        text_kpis += f"Correlación Tamaño lote - OEE: {kpi_summary.get('corr_lote_oee'):.2f} <br/>"
    story.append(Paragraph(text_kpis, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Insert image si existe
    if tmp_img:
        story.append(Paragraph("Gráfica: Tamaño de lote vs OEE", styles["Heading3"]))
        story.append(RLImage(tmp_img, width=400, height=250))
        story.append(Spacer(1, 12))

    # Recomendaciones
    story.append(Paragraph("Recomendaciones del consultor IA:", styles["Heading2"]))
    # dividir por líneas y añadir
    for p in recommendations_text.split("\n"):
        if p.strip():
            story.append(Paragraph(p.strip(), styles["Normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    return filename

# --------------------
# CHAT: prompt y envío a OpenAI
# --------------------
def build_prompt(kpi_summary, df, user_question=None):
    """Construye prompt para el LLM: le damos resumen de KPIs + tabla resumida por referencia + pregunta del usuario."""
    system = (
        "Eres un consultor de operaciones industriales para PYMEs. "
        "Aporta recomendaciones prácticas, priorizadas, con pasos concretos, responsables y estimación de impacto."
    )
    # resumen
    resumen = f"OEE_prom: {kpi_summary.get('OEE_prom'):.2f}, Scrap_prom: {kpi_summary.get('Scrap_prom'):.2f}, Inventario_prom: {kpi_summary.get('Inventario_prom'):.2f}."
    if kpi_summary.get('corr_lote_oee') is not None:
        resumen += f" Correlación lote-OEE: {kpi_summary.get('corr_lote_oee'):.2f}."
    # tabla por referencia (hasta 10 rows)
    sample_table = kpi_summary["by_reference"].to_string(index=False) if "by_reference" in kpi_summary else ""
    prompt = f"{system}\n\nResumen: {resumen}\n\nKPIs por referencia:\n{sample_table}\n\n"
    if user_question:
        prompt += f"Pregunta del usuario: {user_question}\n\n"
    prompt += (
        "1) Da un diagnóstico corto (2-3 frases). "
        "2) Propón 3 recomendaciones accionables (cada una con objetivo cuantificable si es posible). "
        "3) Sugiere el siguiente paso operativo (quién y en cuántos días)."
    )
    return prompt

def ask_openai(prompt, max_tokens=400, temperature=0.2):
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY no configurada. Configura la variable OPENAI_API_KEY y reinicia."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un consultor experto en operaciones industriales."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = resp["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        return f"Error llamando a OpenAI: {e}"

# --------------------
# ACCIONES: CRUD simple con sqlite
# --------------------
def insert_action(reference, title, detail, owner, due_date):
    cur = db.cursor()
    cur.execute("""
    INSERT INTO actions (created_at, reference, title, detail, owner, due_date, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), reference, title, detail, owner, due_date, "pending"))
    db.commit()

def list_actions():
    cur = db.cursor()
    cur.execute("SELECT id, created_at, reference, title, detail, owner, due_date, status FROM actions ORDER BY created_at DESC")
    rows = cur.fetchall()
    cols = ["id","created_at","reference","title","detail","owner","due_date","status"]
    return pd.DataFrame(rows, columns=cols)

def update_action_status(action_id, new_status):
    cur = db.cursor()
    cur.execute("UPDATE actions SET status=? WHERE id=?", (new_status, action_id))
    db.commit()

# --------------------
# STREAMLIT UI
# --------------------
st.title("🤖 Consultor Digital Industrial — MVP")

# Sidebar: upload / sample / actions
st.sidebar.header("1) Datos")
uploaded_file = st.sidebar.file_uploader("Sube template Excel/CSV (hoja KPIS con columnas)", type=["xlsx","csv"])
if st.sidebar.button("Cargar ejemplo de plantilla"):
    with open("template_sample.xlsx", "rb") as f:
        st.download_button("Descargar plantilla ejemplo", f, "template_sample.xlsx")

st.sidebar.markdown("---")
st.sidebar.header("2) Chat / acciones")
st.sidebar.markdown("Crea una acción desde las recomendaciones o manualmente.")
new_action_ref = st.sidebar.text_input("Referencia (ej: PROD-A)")
new_action_title = st.sidebar.text_input("Título acción")
new_action_detail = st.sidebar.text_area("Detalle acción")
new_action_owner = st.sidebar.text_input("Owner (nombre)")
new_action_due = st.sidebar.date_input("Fecha límite", value=datetime.now().date() + timedelta(days=7))
if st.sidebar.button("Crear acción"):
    if new_action_title.strip() == "":
        st.sidebar.error("Pon un título para la acción.")
    else:
        insert_action(new_action_ref, new_action_title, new_action_detail, new_action_owner, new_action_due.isoformat())
        st.sidebar.success("Acción creada ✅")

st.sidebar.markdown("---")
st.sidebar.header("3) Acciones actuales")
actions_df = list_actions()
st.sidebar.dataframe(actions_df[["id","reference","title","owner","due_date","status"]].head(10))

# Main area: cargar datos y mostrar dashboard
if uploaded_file is not None:
    # leer excel o csv
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, sheet_name="KPIS")
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        st.stop()

    st.subheader("Datos — preview")
    st.dataframe(df.head(50))

    # Compute KPIs
    kpi_summary = compute_kpis(df)
    st.subheader("KPIs resumen")
    cols = st.columns(4)
    cols[0].metric("OEE promedio", f"{kpi_summary['OEE_prom']:.2f}%")
    cols[1].metric("Scrap promedio", f"{kpi_summary['Scrap_prom']:.2f}%")
    cols[2].metric("Inventario (días)", f"{kpi_summary['Inventario_prom']:.2f}")
    cols[3].metric("Registros", f"{kpi_summary['N_registros']}")

    # Plots
    st.subheader("Dashboards")
    plots = plot_trends(df)
    if "oee_trend" in plots:
        st.plotly_chart(plots["oee_trend"], use_container_width=True)
    if "scatter_lote_oee" in plots:
        st.plotly_chart(plots["scatter_lote_oee"], use_container_width=True)
    if "inv_trend" in plots:
        st.plotly_chart(plots["inv_trend"], use_container_width=True)

    # Chatbot area
    st.subheader("Chatbot consultor")
    st.markdown("Haz una pregunta al consultor (ej: '¿Qué acciones puntuales propones para reducir scrap?').")
    user_q = st.text_input("Tu pregunta para el consultor:", value="")
    if st.button("Consultar IA"):
        with st.spinner("Consultando al modelo..."):
            prompt = build_prompt(kpi_summary, df, user_question=user_q if user_q.strip() else None)
            answer = ask_openai(prompt)
            st.markdown("**Respuesta IA:**")
            st.info(answer)
            # botón para generar PDF y crear acciones desde la recomendación
            if st.button("Generar informe PDF con estas recomendaciones"):
                pdf_name = f"informe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                generate_pdf(kpi_summary, df, answer, filename=pdf_name)
                with open(pdf_name, "rb") as f:
                    st.download_button("Descargar informe (PDF)", f, file_name=pdf_name)
            # sugerir convertir líneas en acciones
            st.markdown("Puedes crear acciones manualmente en la barra lateral.")            

    # Acciones list + actualizar estado
    st.subheader("Gestión de acciones")
    actions_df = list_actions()
    st.dataframe(actions_df)
    sel = st.number_input("ID acción para marcar completada (introduce id)", min_value=0, step=1)
    if st.button("Marcar acción como completada"):
        if sel > 0:
            update_action_status(sel, "completed")
            st.success("Acción marcada como completada.")
            st.experimental_rerun()

else:
    st.info("Sube el Excel/CSV con la hoja KPIS para empezar (usa la plantilla de ejemplo si quieres).")
    st.write("Estructura esperada (columnas): Fecha, Referencia, Tamaño_lote, OEE, Inventario_días, Scrap_%, LeadTime_días")
