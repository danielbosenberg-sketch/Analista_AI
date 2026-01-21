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
try:
    init_db()
except Exception as e:
    st.error(f"Error base de datos: {e}")

# Inicializar estado del chat flotante
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# --- 2. DISEÑO CSS ROBUSTO (CORREGIDO ESPAÑOL) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #1e293b; }
    .stApp { background-color: #f8fafc; background-image: radial-gradient(#e2e8f0 1px, transparent 1px); background-size: 20px 20px; }
    
    /* HEADER EN SIDEBAR */
    .header-container { 
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); 
        padding: 20px; 
        border-radius: 15px; 
        color: white; 
        text-align: center; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3); 
    }
    .header-title { font-size: 28px; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .header-subtitle { font-size: 14px; opacity: 0.9; font-weight: 300; margin-top: 5px; line-height: 1.2; }

    /* ESTILO DE MÉTRICAS */
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid #ffffff; padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); text-align: center; height: 100%; }
    div[data-testid="stMetricValue"] { font-size: 26px !important; white-space: normal !important; line-height: 1.2 !important; }
    
    /* --- CHAT FLOTANTE FIJO (ESTABLE) --- */
    div.floating-chat-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 400px;
        z-index: 9999;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid #e2e8f0;
    }
    
    /* --- HACKS PARA TRADUCIR EL FILE UPLOADER (CORREGIDO) --- */
    
    /* 1. Ocultar textos originales usando visibilidad y tamaño */
    [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child {
        visibility: hidden;
        height: 0px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2) {
        visibility: hidden;
        height: 0px !important;
    }
    
    /* 2. Insertar texto en español */
    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "Arrastra y suelta archivos aquí";
        visibility: visible;
        display: block;
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 5px;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "Límite de 200MB por archivo";
        visibility: visible;
        display: block;
        text-align: center;
        font-size: 12px;
        color: #9ca3af;
    }

    /* 3. Traducir botón */
    section[data-testid="stFileUploader"] button {
        color: transparent !important;
    }
    section[data-testid="stFileUploader"] button::after {
        content: "Buscar archivos";
        color: #31333F;
        position: absolute;
        left: 0;
        right: 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES DE CARGA Y NORMALIZACIÓN ---

def cargar_google_sheet(url):
    """Convierte un enlace de Google Sheets en un DataFrame"""
    try:
        # Extraer el ID de la hoja
        pattern = r"/d/([a-zA-Z0-9-_]+)"
        match = re.search(pattern, url)
        if match:
            sheet_id = match.group(1)
            # URL de exportación a CSV
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            df = pd.read_csv(csv_url)
            return df, None
        else:
            return None, "El enlace no parece ser de Google Sheets."
    except Exception as e:
        return None, f"Error al acceder al Sheet (Asegúrate de que sea público): {e}"

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
        return None, "No se pudo leer el archivo. Intenta revisar el formato."
    except Exception as e: return None, str(e)

def normalizar_texto(texto):
    if not isinstance(texto, str): return str(texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-zA-Z0-9]', '', texto) 
    return texto.lower()

# --- 4. MOTOR DE DETECCIÓN (SCORING) ---

def buscar_columna_por_puntos(df, keywords_pos, keywords_neg=[]):
    best_col = None
    max_score = 0
    cols = df.columns.tolist()
    for col in cols:
        col_norm = normalizar_texto(col)
        score = 0
        if any(normalizar_texto(neg) in col_norm for neg in keywords_neg): score -= 1000
        for kw in keywords_pos:
            kw_norm = normalizar_texto(kw)
            if kw_norm in col_norm:
                score += 100
                if len(col_norm) <= len(kw_norm) + 5: score += 50
        if score > max_score and score > 0:
            max_score = score
            best_col = col
    return best_col

def detectar_mapa_completo(df):
    mapa = {}
    mapa['factura'] = buscar_columna_por_puntos(df, ['factura', 'invoice', 'consecutivo', 'folio', 'documento', 'ticket', 'id_factura'], ['fecha', 'venc'])
    mapa['cliente_id'] = buscar_columna_por_puntos(df, ['id_cliente', 'nit', 'cedula', 'rut', 'dni', 'identificacion', 'cif'], ['nom', 'razon', 'prod', 'fac'])
    mapa['cliente_nom'] = buscar_columna_por_puntos(df, ['nombre', 'cliente', 'razon', 'social', 'tercero', 'comprador'], ['id', 'cod', 'nit'])
    mapa['producto_id'] = buscar_columna_por_puntos(df, ['id_producto', 'sku', 'codigo', 'referencia', 'ref', 'ean', 'item_id'], ['nom', 'desc', 'cli'])
    mapa['producto_nom'] = buscar_columna_por_puntos(df, ['producto', 'articulo', 'descripcion', 'item', 'detalle', 'material', 'concepto'], ['id', 'cod', 'sku', 'ref'])
    if not mapa['producto_nom']: mapa['producto_nom'] = mapa['producto_id']
    mapa['venta'] = buscar_columna_por_puntos(df, ['total', 'venta', 'importe', 'monto', 'valor', 'precio'], ['unitario', 'impuesto', 'cantidad'])
    mapa['fecha'] = buscar_columna_por_puntos(df, ['fecha', 'date', 'dia', 'registro'], ['venc'])
    if not mapa['venta']: mapa['venta'] = buscar_columna_por_puntos(df, ['precio', 'valor'], [])
    return mapa

# --- 5. AUDITORÍA Y KPIs ---

def auditar_calidad_datos(df, mapa):
    conflictos = []
    col_fact = mapa.get('factura')
    col_cli_nom = mapa.get('cliente_nom')
    col_cli_id = mapa.get('cliente_id')
    col_prod_id = mapa.get('producto_id')
    col_prod_nom = mapa.get('producto_nom')

    if col_cli_nom and col_cli_id:
        df_tmp = df[[col_cli_nom, col_cli_id]].astype(str)
        dup_nom = df_tmp.groupby(col_cli_nom)[col_cli_id].nunique()
        for nom, cant in dup_nom[dup_nom > 1].items(): conflictos.append(f"🔴 <b>Identidad:</b> '{nom}' tiene {cant} IDs distintos.")
    
    col_ref = col_cli_nom if col_cli_nom else col_cli_id
    if col_fact and col_ref:
        df_tmp = df[[col_fact, col_ref]].astype(str)
        dup_fact = df_tmp.groupby(col_fact)[col_ref].nunique()
        for fac, cant in dup_fact[dup_fact > 1].items(): conflictos.append(f"🧾 <b>Error Factura:</b> '{fac}' asignada a {cant} clientes.")
    elif not col_fact: conflictos.append("⚠️ No se puede auditar facturas: Falta columna Factura.")
            
    if col_prod_nom and col_prod_id and col_prod_nom != col_prod_id:
        df_tmp = df[[col_prod_nom, col_prod_id]].astype(str)
        dup_prod = df_tmp.groupby(col_prod_nom)[col_prod_id].nunique()
        for p, cant in dup_prod[dup_prod > 1].items(): conflictos.append(f"📦 <b>Producto Confuso:</b> '{p}' tiene {cant} códigos.")
    return conflictos

def calcular_kpis(df, mapa):
    kpis = {}
    col_venta = mapa.get('venta')
    col_cli = mapa.get('cliente_id') or mapa.get('cliente_nom')
    col_fecha = mapa.get('fecha')
    
    if col_venta:
        if df[col_venta].dtype == 'object':
            try: df[col_venta] = pd.to_numeric(df[col_venta].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            except: pass
        kpis['total_ventas'] = df[col_venta].sum()
    else: kpis['total_ventas'] = 0
    kpis['total_clientes'] = df[col_cli].nunique() if col_cli else 0

    if col_fecha and col_venta:
        try:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
            kpis['fecha_inicio'] = df[col_fecha].min()
            kpis['fecha_cierre'] = df[col_fecha].max()
            ventas_diarias = df.groupby(df[col_fecha].dt.date)[col_venta].sum()
            kpis['promedio_dia'] = ventas_diarias.mean()
            kpis['tendencia_data'] = ventas_diarias.reset_index().rename(columns={col_fecha: 'fecha', col_venta: 'venta_total'})
            df['dia_semana'] = df[col_fecha].dt.day_name()
            dias_esp = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}
            df['dia_semana_esp'] = df['dia_semana'].map(dias_esp)
            dias_orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            df['dia_semana_esp'] = pd.Categorical(df['dia_semana_esp'], categories=dias_orden, ordered=True)
            kpis['ventas_por_dia'] = df.groupby('dia_semana_esp')[col_venta].sum().reset_index().rename(columns={col_venta: 'venta_total'})
            ult_val = ventas_diarias.iloc[-1] if not ventas_diarias.empty else 0
            prom = kpis['promedio_dia']
            if ult_val > prom * 1.2: kpis['alerta'] = ("good", "🚀 Cierre fuerte")
            elif ult_val < prom * 0.8: kpis['alerta'] = ("bad", "📉 Cierre bajo")
            else: kpis['alerta'] = ("neutral", "👍 Estable")
        except:
            kpis['alerta'] = ("neutral", "Error en fechas")
            kpis['promedio_dia'] = 0; kpis['tendencia_data'] = None; kpis['fecha_cierre'] = datetime.now(); kpis['ventas_por_dia'] = None
    else:
        kpis['promedio_dia'] = 0; kpis['tendencia_data'] = None; kpis['fecha_cierre'] = datetime.now(); kpis['ventas_por_dia'] = None; kpis['alerta'] = ("neutral", "Faltan columnas clave")

    col_prod = mapa.get('producto_nom')
    if col_prod and col_venta: kpis['top_productos'] = df.groupby(col_prod)[col_venta].sum().sort_values(ascending=False).head(5)
    else: kpis['top_productos'] = None
    col_cli_n = mapa.get('cliente_nom')
    if col_cli_n and col_venta: kpis['top_clientes'] = df.groupby(col_cli_n)[col_venta].sum().sort_values(ascending=False).head(5)
    else: kpis['top_clientes'] = None
    col_costo = buscar_columna_por_puntos(df, ['costo', 'compra'], [])
    if col_costo and col_venta:
        if df[col_costo].dtype == 'object':
             try: df[col_costo] = pd.to_numeric(df[col_costo].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
             except: pass
        kpis['ganancia_total'] = kpis['total_ventas'] - df[col_costo].sum()
        kpis['margen'] = (kpis['ganancia_total'] / kpis['total_ventas']) * 100 if kpis['total_ventas'] > 0 else 0
    else: kpis['ganancia_total'] = None
    return kpis

# --- 6. AGENTE LANGCHAIN ---

def agente_inteligente_langchain(df, query, api_key):
    if not api_key: return "🔒 Falta API Key."
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)
        res = agent.invoke(query)
        return res['output']
    except Exception as e: return f"Error en análisis profundo: {str(e)}"

# --- 7. AUXILIARES ---

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Reporte Ejecutivo', 0, 1, 'C'); self.ln(5)

def generar_pdf_reporte(kpis, nombre_archivo):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font('Arial', '', 12); pdf.set_fill_color(245, 247, 250)
    pdf.cell(0, 10, f"Archivo: {str(nombre_archivo).encode('latin-1','replace').decode('latin-1')}", 0, 1, 'L'); pdf.ln(10)
    pdf.cell(0, 10, f"Ventas Totales: ${kpis['total_ventas']:,.0f}", 0, 1)
    pdf.cell(0, 10, f"Clientes Totales: {kpis['total_clientes']}", 0, 1)
    if kpis.get('ganancia_total'): pdf.cell(0, 10, f"Ganancia: ${kpis['ganancia_total']:,.0f} (Margen: {kpis['margen']:.1f}%)", 0, 1)
    return pdf.output(dest='S').encode('latin-1', 'replace')

def generar_excel_descarga(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='Datos')
    return output.getvalue()

def limpiar_texto_ia(contenido_crudo): return str(contenido_crudo)

def solucionar_conflictos_ia(lista_errores):
    api_key_val = st.session_state.get("api_key_input", "")
    if not api_key_val:
         try: api_key_val = st.secrets["GOOGLE_API_KEY"]
         except: api_key_val = ""
    if not api_key_val: st.error("🔒 Configura API Key"); return
    with st.spinner("🤖 Analizando..."):
        try:
            os.environ["GOOGLE_API_KEY"] = api_key_val
            prompt = f"Como experto en datos, da pasos breves para corregir en Excel:\n{str(lista_errores[:10])}"
            llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.5)
            res = llm.invoke(prompt)
            st.info(f"**💡 Solución IA:**\n\n{res.content}")
        except Exception as e: st.error(str(e))

def explicar_visualizacion(titulo, datos, key):
    api_key_val = st.session_state.get("api_key_input", "")
    if not api_key_val:
         try: api_key_val = st.secrets["GOOGLE_API_KEY"]
         except: api_key_val = ""
    if api_key_val:
        if st.button(f"✨ Analizar con IA", key=key):
            with st.spinner(f"Consultando..."):
                try:
                    os.environ["GOOGLE_API_KEY"] = api_key_val
                    prompt = f"Analiza gráfico '{titulo}': {datos}. Breve."
                    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)
                    res = llm.invoke(prompt)
                    st.success(f"**💡 Insight:** {res.content}")
                except Exception as e: st.error(str(e))
    else: st.caption("🔒 Configura API Key.")

def formatear_fecha_es(dt):
    if not isinstance(dt, datetime): return "N/A"
    try:
        meses = ["", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
        return f"{meses[dt.month]} {dt.day:02d} de {dt.year} {dt.strftime('%H:%M')}"
    except: return str(dt)

# --- 8. UI PRINCIPAL ---

# SIDEBAR: HEADER, UPLOADER, SETTINGS
with st.sidebar:
    st.markdown("""
    <div class="header-container">
        <p class="header-title">Analítica Pro 🚀</p>
        <p class="header-subtitle">Inteligencia de Negocios</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- OPCIONES DE CARGA ---
    st.markdown("### 📂 Carga de Datos")
    tipo_fuente = st.radio("Selecciona fuente:", ["Subir Archivo", "Google Sheets"], index=0)
    
    uploaded_files = None
    df_google = None
    
    if tipo_fuente == "Subir Archivo":
        uploaded_files = st.file_uploader("Arrastra tu archivo aquí", accept_multiple_files=True)
    else:
        st.info("Asegúrate de que el Google Sheet sea público (Cualquiera con el enlace).")
        sheet_url = st.text_input("Pega el enlace de Google Sheets:")
        if sheet_url:
            with st.spinner("Descargando de Google..."):
                df_google, err_g = cargar_google_sheet(sheet_url)
                if err_g: st.error(err_g)
                else: st.success("✅ ¡Datos cargados exitosamente!")

    st.markdown("---")
    
    with st.expander("⚙️ Ajustes"):
        try: clave_guardada = st.secrets["GOOGLE_API_KEY"]
        except: clave_guardada = ""
        api_key = st.text_input("API Key", value=clave_guardada, type="password", key="api_key_input")
        if st.button("Borrar Conversación"): st.session_state.messages = []; st.rerun()
    st.markdown("---")
    if st.button("🔄 Reiniciar Todo"): st.session_state.clear(); st.rerun()

# --- CHAT FLOTANTE ESTABLE ---
st.markdown('<div class="floating-chat-container">', unsafe_allow_html=True)
with st.expander("🤖 Asistente IA (Clic para abrir)", expanded=False):
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Hola, ¿qué analizamos?"}]
    
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])
    
    if user_query := st.chat_input("Pregunta..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                if not api_key:
                    resp = "🔒 Configura tu API Key en el menú lateral."
                elif "df_raw" in st.session_state:
                    resp = agente_inteligente_langchain(st.session_state["df_raw"], user_query, api_key)
                else:
                    try:
                        os.environ["GOOGLE_API_KEY"] = api_key
                        llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                        resp = llm_chat.invoke(user_query).content
                    except Exception as e: resp = f"Error: {e}"
                
                st.write(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
st.markdown('</div>', unsafe_allow_html=True)

# LÓGICA DE PROCESAMIENTO (ARCHIVOS LOCALES O GOOGLE SHEETS)
if uploaded_files or df_google is not None:
    # Caso 1: Archivos locales
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
                st.session_state["df_raw"] = df_final
                st.session_state["last_files"] = current_names
                st.session_state["mapa"] = detecting_mapa_completo(df_final) if "detecting_mapa_completo" in globals() else detectar_mapa_completo(df_final)
                st.rerun()
    
    # Caso 2: Google Sheets
    elif df_google is not None:
        if "df_raw" not in st.session_state or st.session_state.get("last_url") != sheet_url:
            st.session_state["df_raw"] = df_google
            st.session_state["last_url"] = sheet_url
            st.session_state["mapa"] = detectar_mapa_completo(df_google)
            st.rerun()

if "df_raw" in st.session_state:
    df = st.session_state["df_raw"]
    kpis = calcular_kpis(df, st.session_state["mapa"])
    conflictos = auditar_calidad_datos(df, st.session_state["mapa"])
    st.session_state["kpis_chat"] = kpis
    n_conflicts = len(conflictos)
    label_exp = f"⚠️ Auditoría: {n_conflicts} conflictos encontrados" if n_conflicts > 0 else "✅ Auditoría: Datos Limpios"
    
    with st.expander(label_exp, expanded=(n_conflicts > 0)):
        if n_conflicts > 0:
            if st.button("✨ Ayúdame a arreglar esto con IA"): solucionar_conflictos_ia(conflictos)
            for c in conflictos: st.markdown(f'<div class="audit-item">{c}</div>', unsafe_allow_html=True)
        else: st.success("No se encontraron duplicados ni errores lógicos graves.")

    tipo, msg = kpis.get('alerta', ("neutral", ""))
    st.markdown(f'<div class="custom-alert alert-{tipo}">{msg}</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas Totales", f"${kpis.get('total_ventas', 0):,.0f}")
    c2.metric("Promedio Diario", f"${kpis.get('promedio_dia', 0):,.0f}")
    c3.metric("Total Clientes", kpis.get('total_clientes', 0))
    st.markdown("---")
    c4, c5 = st.columns(2)
    f_ini = kpis.get('fecha_inicio'); f_fin = kpis.get('fecha_cierre')
    c4.metric("📅 Fecha Inicio", formatear_fecha_es(f_ini)); c5.metric("📅 Fecha Fin", formatear_fecha_es(f_fin))

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráficos", "📅 Día a Día", "📥 Exportar", "🕰️ Historial"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if kpis.get('top_productos') is not None:
                df_p = kpis['top_productos'].reset_index()
                col_n = df_p.columns[0]; val_col = df_p.columns[1] 
                fig = px.bar(df_p, x=val_col, y=col_n, orientation='h', title="Top Productos", color=val_col, color_continuous_scale=['#90caf9', '#0d47a1'])
                fig.update_traces(marker_line_color='rgba(0,0,0,0.5)', marker_line_width=1)
                st.plotly_chart(fig, use_container_width=True)
                explicar_visualizacion("Top Productos", df_p.to_string(), "k1")
            else: st.info("No se detectó columna de productos.")
        with c2:
            if kpis.get('top_clientes') is not None:
                df_c = kpis['top_clientes'].reset_index()
                col_n = df_c.columns[0]; val_col = df_c.columns[1]
                fig = px.bar(df_c, x=val_col, y=col_n, orientation='h', title="Top Clientes", color=val_col, color_continuous_scale=['#a5d6a7', '#1b5e20'])
                fig.update_traces(marker_line_color='rgba(0,0,0,0.5)', marker_line_width=1)
                st.plotly_chart(fig, use_container_width=True)
                explicar_visualizacion("Top Clientes", df_c.to_string(), "k2")
                
    with tab2:
        st.markdown("### 📅 Análisis de Calendario")
        if kpis.get('tendencia_data') is not None:
            fig = px.area(kpis['tendencia_data'], x='fecha', y='venta_total', title="📈 Evolución de Ventas")
            st.plotly_chart(fig, use_container_width=True)
        if kpis.get('ventas_por_dia') is not None:
            st.markdown("##### 🗓️ ¿Qué día es más fuerte?")
            fig_d = px.bar(kpis['ventas_por_dia'], x='dia_semana_esp', y='venta_total', color='venta_total', color_continuous_scale=['#d1c4e9', '#311b92'])
            fig_d.update_traces(marker_line_color='rgba(0,0,0,0.5)', marker_line_width=1)
            st.plotly_chart(fig_d, use_container_width=True)
            
    with tab3:
        c1, c2 = st.columns(2)
        pdf_bytes = generar_pdf_reporte(kpis, st.session_state.get("last_files", ["Reporte"]))
        c1.download_button("📄 PDF Reporte", pdf_bytes, "reporte.pdf")
        xls_bytes = generar_excel_descarga(df)
        c2.download_button("📊 Excel Limpio", xls_bytes, "data.xlsx")
        st.dataframe(df.head(100), use_container_width=True)

    with tab4:
        if st.button("Guardar Snapshot"):
            f_cierre = kpis.get('fecha_cierre')
            f_str = f_cierre.strftime("%Y-%m-%d") if isinstance(f_cierre, datetime) else datetime.now().strftime("%Y-%m-%d")
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
            hist_display = hist.copy()
            for col in hist_display.columns:
                if hist_display[col].dtype == 'object': hist_display[col] = hist_display[col].apply(limpiar_celda)
            st.dataframe(hist_display)
            if st.button("Borrar Historial"): borrar_historia(); st.rerun()
