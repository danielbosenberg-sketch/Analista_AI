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
except: pass

# --- 2. DISEÑO CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #1e293b; background-color: #f8fafc; }
    
    .header-container { background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3); }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid #ffffff; padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); text-align: center; height: 100%; }
    
    .chat-panel { background-color: white; border-left: 1px solid #e2e8f0; height: 100%; padding: 1rem; border-radius: 10px; box-shadow: -5px 0 15px rgba(0,0,0,0.02); }
    .chat-msg-user { background-color: #eff6ff; color: #1e3a8a; padding: 10px; border-radius: 10px 10px 0 10px; margin-bottom: 10px; text-align: right; font-size: 0.9rem; }
    .chat-msg-ai { background-color: #f1f5f9; color: #334155; padding: 10px; border-radius: 10px 10px 10px 0; margin-bottom: 10px; text-align: left; font-size: 0.9rem; border: 1px solid #e2e8f0; }
    
    .filter-box { background-color: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;}

    [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child { visibility: hidden; height: 0px !important; }
    [data-testid="stFileUploaderDropzoneInstructions"]::before { content: "Arrastra archivos aquí"; visibility: visible; display: block; text-align: center; font-size: 16px; font-weight: 600; color: #4b5563; margin-bottom: 5px; }

    /* --- CURSOR DE MANITO --- */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly text, .js-plotly-plot .plotly path, .js-plotly-plot .plotly g {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES DE LÓGICA ---

def limpiar_excel_inteligente(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        df = None
        if filename.endswith(('.xls', '.xlsx')):
            try: df = pd.read_excel(uploaded_file)
            except: return None
        elif filename.endswith('.csv'):
            configs = [(';', 'utf-8'), (';', 'latin-1'), (',', 'utf-8'), ('\t', 'utf-16'), (None, 'python')]
            for sep, enc in configs:
                try:
                    uploaded_file.seek(0)
                    df_temp = pd.read_csv(uploaded_file, sep=sep, encoding=enc, engine='python' if sep is None else None)
                    if len(df_temp.columns) > 1: df = df_temp; break
                except: continue
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(how='all')
            return df
    except: return None

def normalizar_texto(texto):
    if not isinstance(texto, str): return str(texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-zA-Z0-9]', '', texto) 
    return texto.lower()

def buscar_columna_por_puntos(df, keywords_pos, keywords_neg=[]):
    best_col = None; max_score = -1000; cols = df.columns.tolist()
    for col in cols:
        col_norm = normalizar_texto(col); score = 0
        if any(normalizar_texto(neg) in col_norm for neg in keywords_neg): score -= 500
        for kw in keywords_pos:
            kw_norm = normalizar_texto(kw)
            if kw_norm in col_norm: 
                score += 100 
                if len(col_norm) <= len(kw_norm) + 5: score += 50
        if score > max_score and score > 0: max_score = score; best_col = col
    return best_col

def detecting_mapa_completo(df):
    mapa = {}
    kw_factura = ['factura', 'invoice', 'consecutivo', 'folio', 'ticket', 'documento']
    kw_cli_id = ['id_cliente', 'nit', 'cedula', 'rut', 'dni', 'identificacion', 'cliente_id', 'cod_cliente']
    kw_cli_nom = ['nombre', 'cliente', 'razon', 'social', 'tercero', 'comprador', 'usuario', 'adquiriente', 'nom_cliente']
    kw_prod_id = ['id_producto', 'sku', 'codigo', 'referencia', 'ref', 'ean', 'item_id', 'cod_prod']
    kw_prod_nom = ['producto', 'articulo', 'descripcion', 'item', 'detalle', 'material', 'desc_prod']
    kw_venta = ['total', 'venta', 'importe', 'monto', 'valor', 'precio', 'subtotal']
    kw_fecha = ['fecha', 'date', 'dia', 'registro', 'emision', 'creacion']
    kw_canal = ['canal', 'medio', 'origen', 'plataforma', 'tipo_venta', 'tienda', 'source', 'sucursal']
    
    mapa['factura'] = buscar_columna_por_puntos(df, kw_factura, ['fecha', 'venc', 'total'])
    mapa['cliente_id'] = buscar_columna_por_puntos(df, kw_cli_id, ['nombre', 'razon', 'social', 'nom_'])
    mapa['cliente_nom'] = buscar_columna_por_puntos(df, kw_cli_nom, ['id', 'cod', 'nit', 'cedula', 'rut', 'dni', 'codigo']) 
    mapa['producto_id'] = buscar_columna_por_puntos(df, kw_prod_id, ['nombre', 'desc'])
    mapa['producto_nom'] = buscar_columna_por_puntos(df, kw_prod_nom, ['id', 'cod', 'sku'])
    mapa['venta'] = buscar_columna_por_puntos(df, kw_venta, ['unitario', 'impuesto', 'cantidad', 'id', 'cod'])
    mapa['fecha'] = buscar_columna_por_puntos(df, kw_fecha, ['nacimiento', 'vencimiento'])
    mapa['canal'] = buscar_columna_por_puntos(df, kw_canal, ['id', 'total'])
    
    if not mapa['producto_nom']: mapa['producto_nom'] = mapa['producto_id']
    if not mapa['cliente_nom']: mapa['cliente_nom'] = mapa['cliente_id']
    if mapa['producto_nom'] == mapa['venta'] and mapa['venta'] is not None: mapa['producto_nom'] = None 
    return mapa

def auditar_calidad_datos(df, mapa):
    conflictos = []
    col_cli_nom = mapa.get('cliente_nom'); col_cli_id = mapa.get('cliente_id')
    if col_cli_nom and col_cli_id:
        try:
            df_tmp = df[[col_cli_nom, col_cli_id]].astype(str)
            dup = df_tmp.groupby(col_cli_nom)[col_cli_id].nunique()
            for nombre, cuenta in dup[dup > 1].items(): conflictos.append(f"🔴 <b>Identidad:</b> '{nombre}' tiene {cuenta} IDs diferentes.")
        except: pass
    return conflictos

def preprocesar_datos(df, mapa):
    col_fecha = mapa.get('fecha')
    if col_fecha and col_fecha in df.columns:
        try:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
            dias_esp = {0:"Lunes", 1:"Martes", 2:"Miércoles", 3:"Jueves", 4:"Viernes", 5:"Sábado", 6:"Domingo"}
            df['dia_semana_esp'] = df[col_fecha].dt.dayofweek.map(dias_esp)
            orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            df['dia_semana_esp'] = pd.Categorical(df['dia_semana_esp'], categories=orden, ordered=True)
            mapa['dia_semana_esp'] = 'dia_semana_esp' 
        except: pass
        
    col_venta = mapa.get('venta')
    if col_venta and col_venta in df.columns:
        if df[col_venta].dtype == 'object':
             try: df[col_venta] = pd.to_numeric(df[col_venta].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
             except: pass
    return df, mapa

def calcular_kpis(df, mapa):
    kpis = {}
    col_venta = mapa.get('venta'); col_cli = mapa.get('cliente_id') or mapa.get('cliente_nom'); col_fecha = mapa.get('fecha')
    
    kpis['total_ventas'] = df[col_venta].sum() if col_venta else 0
    kpis['total_clientes'] = df[col_cli].nunique() if col_cli else 0
    kpis['reg_count'] = len(df)
    
    if col_fecha and col_venta:
        try:
            kpis['fecha_inicio'] = df[col_fecha].min(); kpis['fecha_cierre'] = df[col_fecha].max()
            ventas_diarias = df.groupby(df[col_fecha].dt.date)[col_venta].sum()
            kpis['promedio_dia'] = ventas_diarias.mean()
            if col_cli:
                ultima = df[col_fecha].max(); inicio_mes = ultima - pd.Timedelta(days=30)
                primeras = df.groupby(col_cli)[col_fecha].min()
                kpis['clientes_nuevos'] = primeras[primeras >= inicio_mes].count()
            else: kpis['clientes_nuevos'] = 0
        except: kpis.update({'promedio_dia':0, 'clientes_nuevos':0})
    else: kpis.update({'promedio_dia':0, 'clientes_nuevos':0})
    return kpis

# --- 4. GESTIÓN DE FILTROS ---
if "filters" not in st.session_state: st.session_state.filters = {}

def update_filter(key, value):
    if st.session_state.filters.get(key) == value: del st.session_state.filters[key]
    else: st.session_state.filters[key] = value
    st.rerun()

def clear_filters():
    st.session_state.filters = {}
    st.rerun()

# --- 5. AGENTE IA & PDF ---
def agente_inteligente_langchain(df, query, api_key):
    if not api_key: return "🔒 Configura tu API Key."
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True, 
            handle_parsing_errors=True
        )
        return agent.invoke(query)['output']
    except Exception as e:
        error_msg = str(e)
        if "Could not parse LLM output:" in error_msg:
            match = re.search(r"`(.*)`", error_msg)
            if match: return match.group(1)
        return f"Error procesando: {str(e)}"

def sanitizar_texto_pdf(texto):
    if not isinstance(texto, str): return str(texto)
    return texto.encode('latin-1', 'replace').decode('latin-1')

def generar_pdf_reporte(kpis, df, api_key):
    try:
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, sanitizar_texto_pdf(f"Fecha: {datetime.now().strftime('%Y-%m-%d')}"), 0, 1)
        pdf.cell(0, 8, sanitizar_texto_pdf(f"Ventas Totales: ${kpis.get('total_ventas',0):,.0f}"), 0, 1)
        pdf.cell(0, 8, sanitizar_texto_pdf(f"Clientes Totales: {kpis.get('total_clientes',0)}"), 0, 1)
        
        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key)
                res = llm.invoke(f"Genera un resumen ejecutivo de 3 lineas sobre: Ventas {kpis.get('total_ventas')}, Clientes {kpis.get('total_clientes')}").content
                pdf.ln(10)
                pdf.set_font('Arial', 'I', 11)
                pdf.multi_cell(0, 8, sanitizar_texto_pdf(res))
            except: pass
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except Exception as e:
        return f"Error PDF: {str(e)}".encode('utf-8')

def generar_excel_descarga(df):
    output = io.BytesIO(); 
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='Datos')
    return output.getvalue()

def solucionar_conflictos_ia(lista_errores):
    api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key_val: st.error("🔒 Falta API Key"); return
    with st.spinner("🤖 Analizando..."):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key_val)
            st.info(llm.invoke(f"Tips breves para corregir data duplicada: {str(lista_errores[:3])}").content)
        except: pass

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="header-container"><p class="header-title">Analítica Pro 🚀</p></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Cargar Archivos", accept_multiple_files=True)
    try: api_key = st.secrets["GOOGLE_API_KEY"]
    except: st.markdown("---"); api_key = st.text_input("🔑 API Key Gemini", type="password", key="api_key_input")
    if st.button("🔄 Reiniciar App"): st.session_state.clear(); st.rerun()

# --- 7. CARGA DE DATOS ---
if uploaded_files:
    curr_files = [f.name for f in uploaded_files]
    if st.session_state.get("last_files") != curr_files:
        all_dfs = []
        for f in uploaded_files:
            df = limpiar_excel_inteligente(f)
            if df is not None: all_dfs.append(df)
        if all_dfs:
            df_final = pd.concat(all_dfs, ignore_index=True)
            mapa = detecting_mapa_completo(df_final) if "detecting_mapa_completo" in globals() else detectar_mapa_completo(df_final)
            df_final, mapa = preprocesar_datos(df_final, mapa)
            st.session_state.update({"df_raw": df_final, "last_files": curr_files, "mapa": mapa})
            st.rerun()

# --- 8. DASHBOARD LOGIC ---
if "df_raw" in st.session_state and "mapa" in st.session_state:
    df_raw = st.session_state["df_raw"]; mapa = st.session_state["mapa"]
    
    # MOTOR DE FILTRADO EXCLUSIVO
    def get_data_for_chart(exclusion_key=None):
        df_res = df_raw.copy()
        for key, val in st.session_state.filters.items():
            if key == exclusion_key: continue
            col = mapa.get(key)
            if col and col in df_res.columns:
                df_res = df_res[df_res[col] == val]
        return df_res

    df_fully_filtered = get_data_for_chart(exclusion_key=None)
    kpis = calcular_kpis(df_fully_filtered, mapa)
    conflictos = auditar_calidad_datos(df_raw, mapa)

    # --- LAYOUT PRINCIPAL ---
    c_dash, c_chat = st.columns([3, 1], gap="medium")
    
    with c_dash:
        # HEADER FILTROS
        if st.session_state.filters:
            c1, c2 = st.columns([1, 5])
            c1.button("🗑️ Borrar", on_click=clear_filters, type="primary", use_container_width=True)
            fils = [f"{k}: {v}" for k,v in st.session_state.filters.items()]
            c2.markdown(f'<div class="filter-box">Filtros Activos: {" | ".join(fils)}</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Haz clic en las barras de los gráficos para filtrar dinámicamente.")

        # METRICAS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ventas Totales", f"${kpis.get('total_ventas',0):,.0f}")
        m2.metric("Clientes Filtrados", kpis.get('total_clientes',0))
        m3.metric("Nuevos (Contexto)", kpis.get('clientes_nuevos',0))
        m4.metric("Registros", kpis.get('reg_count',0))
        st.markdown("---")

        def get_colors(df_input, col_key_map):
            sel = st.session_state.filters.get(col_key_map)
            col_real = mapa.get(col_key_map)
            if sel: return ['#EF4444' if str(x)==str(sel) else '#CBD5E1' for x in df_input[col_real]]
            return ['#3B82F6'] * len(df_input)

        # --- GRÁFICOS (FILA 1) ---
        tc1, tc2 = st.columns(2)
        
        # 1. TOP PRODUCTOS
        with tc1:
            col_prod = mapa.get('producto_nom'); col_venta = mapa.get('venta')
            if col_prod and col_venta:
                df_p = get_data_for_chart('producto_nom')
                top_p = df_p.groupby(col_prod)[col_venta].sum().reset_index().sort_values(col_venta, ascending=False).head(10)
                
                # --- SOLUCION FINAL DE COLOR ---
                # No usar color_discrete_sequence en px.bar, usar update_traces(marker_color)
                colors = get_colors(top_p, 'producto_nom')
                fig_p = px.bar(top_p, x=col_venta, y=col_prod, orientation='h', title="🏆 Top 10 Productos ($)")
                fig_p.update_traces(marker_color=colors)
                fig_p.update_layout(yaxis=dict(autorange="reversed"), clickmode='event+select', showlegend=False, dragmode=False)
                
                evt_p = st.plotly_chart(fig_p, use_container_width=True, on_select="rerun", selection_mode="points", key="g_prod")
                if evt_p.selection.points:
                    idx = evt_p.selection.points[0]["point_index"]
                    update_filter('producto_nom', top_p.iloc[idx][col_prod])
            else: st.warning("Sin datos de Productos")

        # 2. TOP CLIENTES (DINERO)
        with tc2:
            col_cli = mapa.get('cliente_nom')
            if col_cli and col_venta:
                df_c = get_data_for_chart('cliente_nom')
                top_c = df_c.groupby(col_cli)[col_venta].sum().reset_index().sort_values(col_venta, ascending=False).head(10)
                
                colors = get_colors(top_c, 'cliente_nom')
                fig_c = px.bar(top_c, x=col_venta, y=col_cli, orientation='h', title="💎 Top 10 Clientes ($ Dinero)")
                fig_c.update_traces(marker_color=colors)
                fig_c.update_layout(yaxis=dict(autorange="reversed"), clickmode='event+select', showlegend=False, dragmode=False)
                
                evt_c = st.plotly_chart(fig_c, use_container_width=True, on_select="rerun", selection_mode="points", key="g_cli")
                if evt_c.selection.points:
                    idx = evt_c.selection.points[0]["point_index"]
                    update_filter('cliente_nom', top_c.iloc[idx][col_cli])
            else: st.warning("Sin datos de Clientes")

        # --- GRÁFICOS (FILA 2) ---
        tc3, tc4, tc5 = st.columns(3)

        # 3. FIDELIDAD (FRECUENCIA)
        with tc3:
            col_fact = mapa.get('factura')
            if col_cli:
                df_f = get_data_for_chart('cliente_nom') 
                if col_fact:
                    top_f = df_f.groupby(col_cli)[col_fact].nunique().reset_index()
                    top_f.columns = [col_cli, 'Compras'] 
                else:
                    top_f = df_f[col_cli].value_counts().reset_index()
                    top_f.columns = [col_cli, 'Compras']
                
                top_f = top_f.sort_values('Compras', ascending=False).head(10)
                
                colors = get_colors(top_f, 'cliente_nom')
                fig_f = px.bar(top_f, x='Compras', y=col_cli, orientation='h', title="❤️ Fidelidad (Total Compras #)")
                fig_f.update_traces(marker_color=colors)
                fig_f.update_layout(yaxis=dict(autorange="reversed"), clickmode='event+select', showlegend=False, dragmode=False)
                
                evt_f = st.plotly_chart(fig_f, use_container_width=True, on_select="rerun", selection_mode="points", key="g_fiel")
                if evt_f.selection.points:
                    idx = evt_f.selection.points[0]["point_index"]
                    update_filter('cliente_nom', top_f.iloc[idx][col_cli])
            else: st.info("Falta info Clientes/Facturas")

        # 4. CANALES
        with tc4:
            col_can = mapa.get('canal')
            if col_can and col_venta:
                df_can = get_data_for_chart('canal')
                top_can = df_can.groupby(col_can)[col_venta].sum().reset_index()
                sel = st.session_state.filters.get('canal')
                pull = [0.1 if str(x)==str(sel) else 0 for x in top_can[col_can]]
                fig_can = px.pie(top_can, values=col_venta, names=col_can, title="🌐 Canales", hole=0.4)
                fig_can.update_traces(pull=pull)
                evt_can = st.plotly_chart(fig_can, use_container_width=True, on_select="rerun", selection_mode="points", key="g_can")
                if evt_can.selection.points:
                    idx = evt_can.selection.points[0]["point_index"]
                    update_filter('canal', top_can.iloc[idx][col_can])
            else: st.info("Falta info Canal")

        # 5. SEMANAL
        with tc5:
            col_dia = mapa.get('dia_semana_esp')
            if col_dia and col_venta:
                df_d = get_data_for_chart('dia_semana_esp')
                orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                top_d = df_d.groupby(col_dia, observed=False)[col_venta].sum().reindex(orden).reset_index()
                
                colors = get_colors(top_d, 'dia_semana_esp')
                fig_d = px.bar(top_d, x=col_dia, y=col_venta, title="🗓️ Semanal")
                fig_d.update_traces(marker_color=colors)
                fig_d.update_layout(clickmode='event+select', showlegend=False, dragmode=False)
                
                evt_d = st.plotly_chart(fig_d, use_container_width=True, on_select="rerun", selection_mode="points", key="g_dia")
                if evt_d.selection.points:
                    idx = evt_d.selection.points[0]["point_index"]
                    update_filter('dia_semana_esp', top_d.iloc[idx][col_dia])
            else: st.info("Falta info Fechas")

        # --- EXPORT Y TABLAS ---
        st.markdown("---")
        ex1, ex2 = st.columns(2)
        api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
        pdf_bytes = generar_pdf_reporte(kpis, df_fully_filtered, api_key_val)
        ex1.download_button("📄 PDF Reporte", pdf_bytes, "reporte.pdf", use_container_width=True)
        xls_bytes = generar_excel_descarga(df_fully_filtered)
        ex2.download_button("📊 Excel Filtrado", xls_bytes, "data.xlsx", use_container_width=True)
        
        with st.expander("Ver Datos Detallados (Filtrados)"):
            st.dataframe(df_fully_filtered.head(200), use_container_width=True)
            
        with st.container(border=True):
            st.subheader("Historial y Auditoría")
            c_hist, c_aud = st.columns(2)
            with c_hist:
                if st.button("Guardar Snapshot"): guardar_en_memoria("Manual", kpis, datetime.now().strftime("%Y-%m-%d")); st.success("Guardado")
                h = obtener_historia()
                if not h.empty: st.dataframe(h)
                if st.button("Borrar Historial"): borrar_historia(); st.rerun()
            with c_aud:
                if conflictos:
                    st.error(f"{len(conflictos)} Problemas detectados")
                    if st.button("Ayuda IA"): solucionar_conflictos_ia(conflictos)
                    for c in conflictos: st.caption(c)
                else: st.success("Datos Limpios")

    # --- CHAT ---
    with c_chat:
        st.markdown('<div class="chat-panel"><h3>🤖 Asistente</h3>', unsafe_allow_html=True)
        if "messages" not in st.session_state: st.session_state.messages = []
        
        chat_cont = st.container(height=500)
        with chat_cont:
            if not st.session_state.messages: st.info("Pregunta sobre los datos que ves.")
            for m in st.session_state.messages:
                cls = "chat-msg-user" if m["role"] == "user" else "chat-msg-ai"
                st.markdown(f'<div class="{cls}">{m["content"]}</div>', unsafe_allow_html=True)
        
        if prompt := st.chat_input("..."):
            st.session_state.messages.append({"role": "user", "content": prompt}); st.rerun()
            
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with chat_cont:
                with st.spinner("..."):
                    res = agente_inteligente_langchain(df_fully_filtered, st.session_state.messages[-1]["content"], api_key)
                    st.session_state.messages.append({"role": "assistant", "content": res}); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
