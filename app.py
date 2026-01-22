import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io
import unicodedata
import re
from fpdf import FPDF
from datetime import datetime
from database_manager import init_db, guardar_en_memoria, obtener_historia, borrar_historia
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Analítica Pro", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Inicializar DB
try: init_db()
except Exception as e: st.error(f"Error base de datos: {e}")

# --- 2. DISEÑO CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #1e293b; }
    .stApp { background-color: #f8fafc; background-image: radial-gradient(#e2e8f0 1px, transparent 1px); background-size: 20px 20px; }
    
    .header-container { background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3); }
    .header-title { font-size: 28px; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .header-subtitle { font-size: 14px; opacity: 0.9; font-weight: 300; margin-top: 5px; line-height: 1.2; }

    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid #ffffff; padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); text-align: center; height: 100%; }
    div[data-testid="stMetricValue"] { font-size: 26px !important; white-space: normal !important; line-height: 1.2 !important; }
    
    .chat-panel { background-color: white; border-left: 1px solid #e2e8f0; height: 100%; padding: 1rem; border-radius: 10px; box-shadow: -5px 0 15px rgba(0,0,0,0.02); }
    .chat-msg-user { background-color: #eff6ff; color: #1e3a8a; padding: 10px; border-radius: 10px 10px 0 10px; margin-bottom: 10px; text-align: right; font-size: 0.9rem; }
    .chat-msg-ai { background-color: #f1f5f9; color: #334155; padding: 10px; border-radius: 10px 10px 10px 0; margin-bottom: 10px; text-align: left; font-size: 0.9rem; border: 1px solid #e2e8f0; }

    [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child { visibility: hidden; height: 0px !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2) { visibility: hidden; height: 0px !important; }
    [data-testid="stFileUploaderDropzoneInstructions"]::before { content: "Arrastra archivos aquí"; visibility: visible; display: block; text-align: center; font-size: 16px; font-weight: 600; color: #4b5563; margin-bottom: 5px; }
    [data-testid="stFileUploaderDropzoneInstructions"]::after { content: "Límite 200MB"; visibility: visible; display: block; text-align: center; font-size: 12px; color: #9ca3af; }
    section[data-testid="stFileUploader"] button { color: transparent !important; }
    section[data-testid="stFileUploader"] button::after { content: "Buscar archivos"; color: #31333F; position: absolute; left: 0; right: 0; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES DE LÓGICA ---

def cargar_google_sheet(url):
    try:
        match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
        if match:
            csv_url = f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"
            return pd.read_csv(csv_url), None
        return None, "Enlace no válido."
    except Exception as e: return None, f"Error: {e}"

def limpiar_excel_inteligente(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        df = None
        if filename.endswith(('.xls', '.xlsx')):
            try: df = pd.read_excel(uploaded_file)
            except Exception as e: return None, f"Excel error: {e}"
        elif filename.endswith('.csv'):
            configs = [(';', 'utf-8'), (';', 'latin-1'), (';', 'utf-8-sig'), (',', 'utf-8'), (',', 'latin-1'), ('\t', 'utf-16'), (None, 'python')]
            for sep, enc in configs:
                try:
                    uploaded_file.seek(0)
                    df_temp = pd.read_csv(uploaded_file, sep=sep, encoding=enc, engine='python' if sep is None else None)
                    if len(df_temp.columns) > 1: df = df_temp; break
                except: continue
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(how='all')
            return df, None
        return None, "Error de formato."
    except Exception as e: return None, str(e)

def normalizar_texto(texto):
    if not isinstance(texto, str): return str(texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-zA-Z0-9]', '', texto) 
    return texto.lower()

def buscar_columna_por_puntos(df, keywords_pos, keywords_neg=[]):
    best_col = None; max_score = 0; cols = df.columns.tolist()
    for col in cols:
        col_norm = normalizar_texto(col); score = 0
        if any(normalizar_texto(neg) in col_norm for neg in keywords_neg): score -= 1000
        for kw in keywords_pos:
            kw_norm = normalizar_texto(kw)
            if kw_norm in col_norm: 
                score += 100 
                if len(col_norm) <= len(kw_norm) + 5: score += 50
        if score > max_score and score > 0: max_score = score; best_col = col
    return best_col

def detectar_mapa_completo(df):
    """
    Motor de detección mejorado con 'Canal' y Diccionario Retail.
    """
    mapa = {}
    
    # --- DICCIONARIOS DE PALABRAS CLAVE ---
    kw_factura = ['factura', 'invoice', 'consecutivo', 'folio', 'ticket', 'documento', 'doc_ref', 'comprobante']
    kw_cli_id = ['id_cliente', 'nit', 'cedula', 'rut', 'dni', 'identificacion', 'cif', 'cliente_id', 'cod_cliente']
    kw_cli_nom = ['nombre', 'cliente', 'razon', 'social', 'tercero', 'comprador', 'usuario', 'adquiriente', 'nom_cliente']
    kw_prod_id = ['id_producto', 'sku', 'codigo', 'referencia', 'ref', 'ean', 'item_id', 'cod_prod']
    kw_prod_nom = ['producto', 'articulo', 'descripcion', 'item', 'detalle', 'material', 'concepto', 'mercancia', 'desc_prod']
    kw_venta = ['total', 'venta', 'importe', 'monto', 'valor', 'precio', 'subtotal', 'neto', 'total_venta']
    kw_fecha = ['fecha', 'date', 'dia', 'registro', 'emision', 'creacion', 'fecha_factura']
    kw_canal = ['canal', 'medio', 'origen', 'plataforma', 'tipo_venta', 'tienda', 'source', 'vendedor', 'sucursal']
    
    # --- DETECCION ---
    mapa['factura'] = buscar_columna_por_puntos(df, kw_factura, ['fecha', 'venc', 'total'])
    mapa['cliente_id'] = buscar_columna_por_puntos(df, kw_cli_id, ['nom', 'razon', 'factura', 'prod'])
    mapa['cliente_nom'] = buscar_columna_por_puntos(df, kw_cli_nom, ['id', 'cod', 'nit', 'producto', 'articulo', 'sku']) 
    mapa['producto_id'] = buscar_columna_por_puntos(df, kw_prod_id, ['nom', 'desc', 'cli', 'razon'])
    mapa['producto_nom'] = buscar_columna_por_puntos(df, kw_prod_nom, ['id', 'cod', 'sku', 'cliente', 'razon', 'nit'])
    
    if not mapa['producto_nom']: mapa['producto_nom'] = mapa['producto_id']
    
    mapa['venta'] = buscar_columna_por_puntos(df, kw_venta, ['unitario', 'impuesto', 'cantidad', 'id', 'cod'])
    mapa['fecha'] = buscar_columna_por_puntos(df, kw_fecha, ['venc', 'nacimiento'])
    mapa['canal'] = buscar_columna_por_puntos(df, kw_canal, ['total', 'fecha', 'id'])
    
    if mapa['producto_nom'] == mapa['venta'] and mapa['venta'] is not None:
        mapa['producto_nom'] = None 

    return mapa

def auditar_calidad_datos(df, mapa):
    conflictos = []
    col_cli_nom = mapa.get('cliente_nom')
    col_cli_id = mapa.get('cliente_id')
    col_fact = mapa.get('factura')
    col_ref = col_cli_nom if col_cli_nom else col_cli_id 

    if col_cli_nom and col_cli_id:
        try:
            df_tmp = df[[col_cli_nom, col_cli_id]].astype(str)
            dup = df_tmp.groupby(col_cli_nom)[col_cli_id].nunique()
            for nombre, cuenta in dup[dup > 1].items():
                conflictos.append(f"🔴 <b>Identidad:</b> '{nombre}' tiene {cuenta} IDs diferentes.")
        except: pass
    
    if col_fact and col_ref:
        try:
            df_tmp = df[[col_fact, col_ref]].astype(str)
            dup = df_tmp.groupby(col_fact)[col_ref].nunique()
            for fact, cuenta in dup[dup > 1].items():
                conflictos.append(f"🧾 <b>Error Factura:</b> '{fact}' asignada a {cuenta} clientes distintos.")
        except: pass
    elif not col_fact:
        conflictos.append("⚠️ No se pudo auditar facturas: No se detectó columna de Factura.")

    return conflictos

def calcular_kpis(df, mapa):
    kpis = {}
    col_venta = mapa.get('venta'); col_cli = mapa.get('cliente_id') or mapa.get('cliente_nom'); col_fecha = mapa.get('fecha')
    
    # Ventas Totales
    if col_venta:
        if df[col_venta].dtype == 'object': 
            try: df[col_venta] = pd.to_numeric(df[col_venta].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            except: pass
        kpis['total_ventas'] = df[col_venta].sum()
    else: kpis['total_ventas'] = 0
    
    # Clientes Totales
    kpis['total_clientes'] = df[col_cli].nunique() if col_cli else 0
    
    # Fechas y Nuevos Clientes
    if col_fecha and col_venta:
        try:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
            kpis['fecha_inicio'] = df[col_fecha].min(); kpis['fecha_cierre'] = df[col_fecha].max()
            
            # Tendencia
            ventas_diarias = df.groupby(df[col_fecha].dt.date)[col_venta].sum()
            kpis['promedio_dia'] = ventas_diarias.mean()
            kpis['tendencia_data'] = ventas_diarias.reset_index().rename(columns={col_fecha: 'fecha', col_venta: 'venta_total'})
            
            # Ventas por día de semana
            df['dia_semana'] = df[col_fecha].dt.day_name()
            dias_esp = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}
            df['dia_semana_esp'] = pd.Categorical(df['dia_semana'].map(dias_esp), categories=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], ordered=True)
            kpis['ventas_por_dia'] = df.groupby('dia_semana_esp')[col_venta].sum().reset_index().rename(columns={col_venta: 'venta_total'})
            kpis['alerta'] = ("good", "🚀 Cierre fuerte") if ventas_diarias.iloc[-1] > kpis['promedio_dia']*1.2 else ("neutral", "👍 Estable")
            
            # Clientes Nuevos (Primer compra en el último mes de los datos)
            if col_cli:
                ultima_fecha = df[col_fecha].max()
                inicio_mes_final = ultima_fecha - pd.Timedelta(days=30)
                primeras_compras = df.groupby(col_cli)[col_fecha].min()
                nuevos = primeras_compras[primeras_compras >= inicio_mes_final].count()
                kpis['clientes_nuevos'] = nuevos
            else:
                kpis['clientes_nuevos'] = 0

        except: 
            kpis.update({'promedio_dia':0, 'alerta':("neutral", "Error Fechas"), 'fecha_inicio': datetime.now(), 'fecha_cierre': datetime.now(), 'clientes_nuevos':0})
    else: 
        kpis.update({'promedio_dia':0, 'alerta':("neutral", "Faltan Columnas"), 'fecha_inicio': datetime.now(), 'fecha_cierre': datetime.now(), 'clientes_nuevos':0})
    
    # Top 10 Productos
    col_prod = mapa.get('producto_nom')
    if col_prod and col_venta: 
        kpis['top_productos'] = df.groupby(col_prod)[col_venta].sum().sort_values(ascending=False).head(10)
    else: kpis['top_productos'] = None
    
    # Top 10 Clientes (Dinero)
    col_cli_n = mapa.get('cliente_nom')
    if col_cli_n and col_venta: 
        kpis['top_clientes'] = df.groupby(col_cli_n)[col_venta].sum().sort_values(ascending=False).head(10)
    else: kpis['top_clientes'] = None
    
    # Top 10 Clientes Fieles (Frecuencia)
    col_fact = mapa.get('factura')
    if col_cli_n:
        if col_fact:
            # Contar facturas únicas
            kpis['top_fieles'] = df.groupby(col_cli_n)[col_fact].nunique().sort_values(ascending=False).head(10)
        else:
            # Contar filas (si no hay factura)
            kpis['top_fieles'] = df[col_cli_n].value_counts().head(10)
    else: kpis['top_fieles'] = None

    # Ventas por Canal
    col_canal = mapa.get('canal')
    if col_canal and col_venta:
        kpis['ventas_canal'] = df.groupby(col_canal)[col_venta].sum().sort_values(ascending=False)
    else: kpis['ventas_canal'] = None

    # Margen
    col_costo = buscar_columna_por_puntos(df, ['costo', 'compra'], [])
    if col_costo and col_venta:
        if df[col_costo].dtype == 'object':
             try: df[col_costo] = pd.to_numeric(df[col_costo].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
             except: pass
        kpis['ganancia_total'] = kpis['total_ventas'] - df[col_costo].sum()
        kpis['margen'] = (kpis['ganancia_total'] / kpis['total_ventas']) * 100 if kpis['total_ventas'] > 0 else 0
    else: kpis['ganancia_total'] = None
    
    return kpis

# --- AGENTE INTELIGENTE ---
def agente_inteligente_langchain(df, query, api_key):
    if not api_key: return "🔒 Por favor configura tu API Key en la barra lateral."
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True, 
            handle_parsing_errors=True
        )
        response = agent.invoke(query)
        return response['output']
    except Exception as e: return f"Error procesando tu pregunta: {str(e)}"

# --- PDF ---
class PDFReport(FPDF):
    def header(self): 
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Reporte de Desempeño Comercial', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f'Generado el: {datetime.now().strftime("%d/%m/%Y")}', 0, 1, 'C')
        self.ln(10)

def generar_pdf_reporte(kpis, nombre_archivo, df, api_key):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Resumen de KPIs', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Ventas Totales: ${kpis['total_ventas']:,.0f}", 0, 1)
    pdf.cell(0, 8, f"Total Clientes: {kpis['total_clientes']}", 0, 1)
    if kpis.get('clientes_nuevos'):
        pdf.cell(0, 8, f"Clientes Nuevos (Ult. 30 dias): {kpis['clientes_nuevos']}", 0, 1)
    
    pdf.ln(10)

    if api_key:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '2. Análisis de Inteligencia Artificial', 0, 1)
        pdf.set_font('Arial', 'I', 11)
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key)
            prompt = f"Genera un resumen ejecutivo breve (máx 3 lineas) sobre estos datos: Ventas {kpis['total_ventas']}, Clientes {kpis['total_clientes']}. Da una recomendación."
            resumen_ia = llm.invoke(prompt).content
            pdf.multi_cell(0, 8, resumen_ia)
        except:
            pdf.multi_cell(0, 8, "No se pudo generar el análisis IA (Verifique API Key).")
        pdf.ln(10)

    if kpis.get('top_productos') is not None:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '3. Top 5 Productos', 0, 1)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(220, 220, 220)
        
        pdf.cell(100, 8, 'Producto', 1, 0, 'C', 1)
        pdf.cell(60, 8, 'Ventas ($)', 1, 1, 'C', 1)
        
        pdf.set_font('Arial', '', 10)
        top_prod = kpis['top_productos'].head(5)
        for prod, venta in top_prod.items():
            nombre_p = str(prod)[:45] + "..." if len(str(prod)) > 45 else str(prod)
            pdf.cell(100, 8, nombre_p, 1)
            pdf.cell(60, 8, f"${venta:,.0f}", 1, 1, 'R')

    return pdf.output(dest='S').encode('latin-1', 'replace')

def generar_excel_descarga(df):
    output = io.BytesIO(); 
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='Datos')
    return output.getvalue()

def formatear_fecha_es(dt):
    if not isinstance(dt, datetime): return "N/A"
    try:
        meses = ["", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        return f"{meses[dt.month]} {dt.day:02d} de {dt.year}"
    except: return str(dt)

def explicar_visualizacion(titulo, datos, key):
    api_key_val = st.session_state.get("api_key_input", "") or st.secrets.get("GOOGLE_API_KEY", "")
    if api_key_val:
        if st.button(f"✨ Analizar", key=key):
            with st.spinner("..."):
                try:
                    os.environ["GOOGLE_API_KEY"] = api_key_val
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
                    st.success(llm.invoke(f"Analiza brevemente: {titulo}. Datos: {datos}").content)
                except Exception as e: st.error(str(e))
    else: st.caption("🔒 Configura API Key.")

def solucionar_conflictos_ia(lista_errores):
    api_key_val = st.session_state.get("api_key_input", "") or st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key_val: st.error("🔒 Falta API Key"); return
    with st.spinner("🤖 Analizando..."):
        try:
            os.environ["GOOGLE_API_KEY"] = api_key_val
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
            st.info(llm.invoke(f"Da pasos para corregir en Excel: {str(lista_errores[:10])}").content)
        except Exception as e: st.error(str(e))

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="header-container">
        <p class="header-title">Analítica Pro 🚀</p>
        <p class="header-subtitle">Inteligencia de Negocios</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Carga de Datos")
    tipo_fuente = st.radio("Fuente:", ["Subir Archivo", "Google Sheets"], index=0)
    
    uploaded_files = None
    df_google = None
    
    if tipo_fuente == "Subir Archivo":
        uploaded_files = st.file_uploader("Arrastra archivos aquí", accept_multiple_files=True)
    else:
        st.info("El sheet debe ser público.")
        sheet_url = st.text_input("Enlace Google Sheets:")

    try: 
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        st.markdown("---")
        api_key = st.text_input("🔑 API Key Gemini", type="password", key="api_key_input")

    st.markdown("---")
    if st.button("🔄 Reiniciar App"): st.session_state.clear(); st.rerun()

# --- 5. LOGICA PRINCIPAL ---
if uploaded_files or df_google is not None:
    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if st.session_state.get("last_files") != current_names:
            all_dfs = []
            for f in uploaded_files:
                df, err = limpiar_excel_inteligente(f)
                if df is not None: all_dfs.append(df)
                else: st.error(f"Error {f.name}: {err}")
            if all_dfs:
                df_final = pd.concat(all_dfs, ignore_index=True)
                st.session_state.update({"df_raw": df_final, "last_files": current_names, "mapa": detecting_mapa_completo(df_final) if "detecting_mapa_completo" in globals() else detectar_mapa_completo(df_final)})
                st.rerun()
    elif df_google is not None:
        if "df_raw" not in st.session_state or st.session_state.get("last_url") != sheet_url:
            st.session_state.update({"df_raw": df_google, "last_url": sheet_url, "mapa": detectar_mapa_completo(df_google)})
            st.rerun()

# --- 6. LAYOUT DASHBOARD ---
if "df_raw" in st.session_state:
    df = st.session_state["df_raw"]
    kpis = calcular_kpis(df, st.session_state["mapa"])
    conflictos = auditar_calidad_datos(df, st.session_state["mapa"])
    
    c_dashboard, c_chat = st.columns([3, 1], gap="medium")
    
    # ---------------- DASHBOARD (SINGLE PAGE) ----------------
    with c_dashboard:
        # SECCIÓN 1: AUDITORÍA Y KPIs
        n_conflicts = len(conflictos)
        label_exp = f"⚠️ Auditoría: {n_conflicts} conflictos" if n_conflicts > 0 else "✅ Auditoría: Datos Limpios"
        with st.expander(label_exp, expanded=False):
            if n_conflicts > 0:
                if st.button("✨ Ayuda IA", key="btn_audit"): solucionar_conflictos_ia(conflictos)
                for c in conflictos: st.markdown(f'<div class="audit-item">{c}</div>', unsafe_allow_html=True)
            else: st.success("No se encontraron duplicados ni errores lógicos graves.")
            
            with st.expander("🕵️ Ver columnas detectadas"):
                st.write("Columnas que el sistema está usando:")
                st.json(st.session_state["mapa"])

        tipo, msg = kpis.get('alerta', ("neutral", ""))
        st.markdown(f'<div class="custom-alert alert-{tipo}">{msg}</div>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ventas Totales", f"${kpis.get('total_ventas', 0):,.0f}")
        m2.metric("Total Clientes", kpis.get('total_clientes', 0))
        m3.metric("Clientes Nuevos (Mes)", kpis.get('clientes_nuevos', 0))
        m4.metric("Promedio Diario", f"${kpis.get('promedio_dia', 0):,.0f}")
        
        st.markdown("---")

        # SECCIÓN 2: GRÁFICOS (VISUALIZACIÓN)
        with st.container(border=True):
            st.subheader("📊 Análisis Visual")
            
            # FILA 1: Top Productos y Clientes
            tc1, tc2 = st.columns(2)
            with tc1:
                if kpis.get('top_productos') is not None:
                    df_p = kpis['top_productos'].reset_index()
                    df_p.columns = ['Producto', 'Total_Venta'] 
                    fig = px.bar(df_p, x='Total_Venta', y='Producto', orientation='h', title="🏆 Top 10 Productos", color='Total_Venta', color_continuous_scale=['#90caf9', '#0d47a1'])
                    fig.update_layout(yaxis=dict(autorange="reversed")) # Invertir para que el #1 salga arriba
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Sin datos de productos.")
            
            with tc2:
                if kpis.get('top_clientes') is not None:
                    df_c = kpis['top_clientes'].reset_index()
                    df_c.columns = ['Cliente', 'Total_Venta']
                    fig = px.bar(df_c, x='Total_Venta', y='Cliente', orientation='h', title="💎 Top 10 Clientes ($)", color='Total_Venta', color_continuous_scale=['#a5d6a7', '#1b5e20'])
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
            
            # FILA 2: Fidelidad y Canales
            tc3, tc4 = st.columns(2)
            with tc3:
                if kpis.get('top_fieles') is not None:
                    df_f = kpis['top_fieles'].reset_index()
                    df_f.columns = ['Cliente', 'Frecuencia']
                    fig_f = px.bar(df_f, x='Frecuencia', y='Cliente', orientation='h', title="❤️ Clientes Más Fieles (Frecuencia)", color='Frecuencia', color_continuous_scale='Oranges')
                    fig_f.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_f, use_container_width=True)
                else: st.info("Falta información para medir fidelidad.")
            
            with tc4:
                if kpis.get('ventas_canal') is not None:
                    df_can = kpis['ventas_canal'].reset_index()
                    df_can.columns = ['Canal', 'Ventas']
                    fig_can = px.pie(df_can, values='Ventas', names='Canal', title="🌐 Ventas por Canal (Físico vs Digital)", hole=0.4)
                    st.plotly_chart(fig_can, use_container_width=True)
                else: 
                    st.info("⚠️ No se detectó columna 'Canal'.")

            # MANTENEMOS SOLO VENTAS POR DÍA DE SEMANA (NO CRONOLÓGICO)
            st.markdown("#### 🗓️ Análisis Semanal")
            if kpis.get('ventas_por_dia') is not None:
                st.caption("¿Qué día de la semana es más fuerte?")
                fig_d = px.bar(kpis['ventas_por_dia'], x='dia_semana_esp', y='venta_total', color='venta_total', color_continuous_scale=['#d1c4e9', '#311b92'])
                fig_d.update_traces(marker_line_color='rgba(0,0,0,0.5)', marker_line_width=1)
                st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("---")

        # SECCIÓN 3: EXPORTACIÓN Y DATOS
        with st.container(border=True):
            st.subheader("📥 Exportación y Datos")
            ex1, ex2 = st.columns(2)
            api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
            pdf_bytes = generar_pdf_reporte(kpis, st.session_state.get("last_files", ["Reporte"]), df, api_key_val)
            ex1.download_button("📄 PDF Reporte (Con IA)", pdf_bytes, "reporte_inteligente.pdf")
            xls_bytes = generar_excel_descarga(df)
            ex2.download_button("📊 Excel Limpio", xls_bytes, "data.xlsx")
            with st.expander("Ver tabla completa de datos"):
                st.dataframe(df.head(100), use_container_width=True)

        # SECCIÓN 4: HISTORIAL
        with st.container(border=True):
            st.subheader("🕰️ Historial de Cierres")
            if st.button("Guardar Snapshot de hoy"):
                f_str = kpis.get('fecha_cierre').strftime("%Y-%m-%d") if isinstance(kpis.get('fecha_cierre'), datetime) else datetime.now().strftime("%Y-%m-%d")
                guardar_en_memoria("Manual", kpis, f_str)
                st.success("Guardado")
            
            hist = obtener_historia()
            if not hist.empty:
                def limpiar_celda(valor):
                    if valor is None: return ""
                    if isinstance(valor, (int, float)): return valor
                    if isinstance(valor, bytes):
                        try: return valor.decode('utf-8')
                        except: return str(valor)
                    return str(valor)
                hist_disp = hist.copy()
                for col in hist_disp.columns:
                    if hist_disp[col].dtype == 'object': hist_disp[col] = hist_disp[col].apply(limpiar_celda)
                st.dataframe(hist_disp, use_container_width=True)
                if st.button("Borrar Historial"): borrar_historia(); st.rerun()

    # ---------------- CHAT PERMANENTE ----------------
    with c_chat:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.markdown("### 🤖 Asistente")
        st.caption("Pregunta a tus datos:")
        
        if "messages" not in st.session_state: st.session_state.messages = []
        
        chat_container = st.container(height=500)
        with chat_container:
            if not st.session_state.messages:
                st.info("👋 ¡Hola! Soy tu analista de datos.")
            for m in st.session_state.messages:
                cls = "chat-msg-user" if m["role"] == "user" else "chat-msg-ai"
                st.markdown(f'<div class="{cls}">{m["content"]}</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("Escribe aquí..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with chat_container:
                with st.spinner("Analizando..."):
                    api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
                    response = agente_inteligente_langchain(df, st.session_state.messages[-1]["content"], api_key_val)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
