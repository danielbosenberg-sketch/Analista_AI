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
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid #ffffff; padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); text-align: center; height: 100%; }
    .chat-panel { background-color: white; border-left: 1px solid #e2e8f0; height: 100%; padding: 1rem; border-radius: 10px; box-shadow: -5px 0 15px rgba(0,0,0,0.02); }
    .chat-msg-user { background-color: #eff6ff; color: #1e3a8a; padding: 10px; border-radius: 10px 10px 0 10px; margin-bottom: 10px; text-align: right; font-size: 0.9rem; }
    .chat-msg-ai { background-color: #f1f5f9; color: #334155; padding: 10px; border-radius: 10px 10px 10px 0; margin-bottom: 10px; text-align: left; font-size: 0.9rem; border: 1px solid #e2e8f0; }
    .active-filters { background-color: #dbeafe; border: 1px solid #93c5fd; padding: 10px; border-radius: 8px; margin-bottom: 15px; color: #1e40af; font-weight: 600; font-size: 0.9rem; text-align: center;}
    [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child { visibility: hidden; height: 0px !important; }
    [data-testid="stFileUploaderDropzoneInstructions"]::before { content: "Arrastra archivos aquí"; visibility: visible; display: block; text-align: center; font-size: 16px; font-weight: 600; color: #4b5563; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCIONES DE LÓGICA ---

def limpiar_excel_inteligente(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        df = None
        if filename.endswith(('.xls', '.xlsx')):
            try: df = pd.read_excel(uploaded_file)
            except Exception as e: return None, f"Excel error: {e}"
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
    mapa = {}
    kw_factura = ['factura', 'invoice', 'consecutivo', 'folio', 'ticket', 'documento']
    kw_cli_id = ['id_cliente', 'nit', 'cedula', 'rut', 'dni', 'identificacion', 'cliente_id']
    kw_cli_nom = ['nombre', 'cliente', 'razon', 'social', 'tercero', 'comprador', 'usuario']
    kw_prod_id = ['id_producto', 'sku', 'codigo', 'referencia', 'ref', 'ean', 'item_id']
    kw_prod_nom = ['producto', 'articulo', 'descripcion', 'item', 'detalle', 'material']
    kw_venta = ['total', 'venta', 'importe', 'monto', 'valor', 'precio', 'subtotal']
    kw_fecha = ['fecha', 'date', 'dia', 'registro', 'emision', 'creacion']
    kw_canal = ['canal', 'medio', 'origen', 'plataforma', 'tipo_venta', 'tienda', 'source', 'sucursal']
    
    mapa['factura'] = buscar_columna_por_puntos(df, kw_factura, ['fecha', 'venc', 'total'])
    mapa['cliente_id'] = buscar_columna_por_puntos(df, kw_cli_id, ['nom', 'razon', 'factura', 'prod'])
    mapa['cliente_nom'] = buscar_columna_por_puntos(df, kw_cli_nom, ['id', 'cod', 'nit', 'producto', 'articulo', 'sku']) 
    mapa['producto_id'] = buscar_columna_por_puntos(df, kw_prod_id, ['nom', 'desc', 'cli', 'razon'])
    mapa['producto_nom'] = buscar_columna_por_puntos(df, kw_prod_nom, ['id', 'cod', 'sku', 'cliente', 'razon', 'nit'])
    if not mapa['producto_nom']: mapa['producto_nom'] = mapa['producto_id']
    mapa['venta'] = buscar_columna_por_puntos(df, kw_venta, ['unitario', 'impuesto', 'cantidad', 'id', 'cod'])
    mapa['fecha'] = buscar_columna_por_puntos(df, kw_fecha, ['venc', 'nacimiento'])
    mapa['canal'] = buscar_columna_por_puntos(df, kw_canal, ['total', 'fecha', 'id'])
    
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

# --- PREPROCESAMIENTO DE FECHAS (CLAVE PARA EL FILTRADO) ---
def preprocesar_fechas(df, col_fecha):
    if col_fecha and col_fecha in df.columns:
        try:
            df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
            dias_esp = {"Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"}
            # Creamos la columna real en el dataframe
            df['dia_semana_esp'] = df[col_fecha].dt.day_name().map(dias_esp)
        except: pass
    return df

def calcular_kpis(df, mapa):
    kpis = {}
    col_venta = mapa.get('venta'); col_cli = mapa.get('cliente_id') or mapa.get('cliente_nom'); col_fecha = mapa.get('fecha')
    
    # Ventas y Clientes
    if col_venta:
        if df[col_venta].dtype == 'object': 
            try: df[col_venta] = pd.to_numeric(df[col_venta].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            except: pass
        kpis['total_ventas'] = df[col_venta].sum()
    else: kpis['total_ventas'] = 0
    kpis['total_clientes'] = df[col_cli].nunique() if col_cli else 0
    
    # Fechas
    if col_fecha and col_venta:
        try:
            kpis['fecha_inicio'] = df[col_fecha].min(); kpis['fecha_cierre'] = df[col_fecha].max()
            ventas_diarias = df.groupby(df[col_fecha].dt.date)[col_venta].sum()
            kpis['promedio_dia'] = ventas_diarias.mean()
            kpis['alerta'] = ("good", "🚀 Cierre fuerte") if ventas_diarias.iloc[-1] > kpis['promedio_dia']*1.2 else ("neutral", "👍 Estable")
            
            # Semanal (Ahora usa la columna pre-calculada)
            if 'dia_semana_esp' in df.columns:
                orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                df_sem = df.groupby('dia_semana_esp')[col_venta].sum().reindex(orden_dias).reset_index()
                kpis['ventas_por_dia'] = df_sem.rename(columns={col_venta: 'venta_total'})
            
            if col_cli:
                ultima_fecha = df[col_fecha].max(); inicio_mes_final = ultima_fecha - pd.Timedelta(days=30)
                primeras_compras = df.groupby(col_cli)[col_fecha].min()
                kpis['clientes_nuevos'] = primeras_compras[primeras_compras >= inicio_mes_final].count()
            else: kpis['clientes_nuevos'] = 0
        except: kpis.update({'promedio_dia':0, 'alerta':("neutral", "Error Fechas"), 'fecha_inicio': datetime.now(), 'fecha_cierre': datetime.now(), 'clientes_nuevos':0})
    else: kpis.update({'promedio_dia':0, 'alerta':("neutral", "Faltan Columnas"), 'fecha_inicio': datetime.now(), 'fecha_cierre': datetime.now(), 'clientes_nuevos':0})
    
    # Rankings (Top 10)
    col_prod = mapa.get('producto_nom')
    if col_prod and col_venta: kpis['top_productos'] = df.groupby(col_prod)[col_venta].sum().sort_values(ascending=False).head(10)
    else: kpis['top_productos'] = None
    
    col_cli_n = mapa.get('cliente_nom')
    if col_cli_n and col_venta: kpis['top_clientes'] = df.groupby(col_cli_n)[col_venta].sum().sort_values(ascending=False).head(10)
    else: kpis['top_clientes'] = None
    
    col_fact = mapa.get('factura')
    if col_cli_n:
        if col_fact: kpis['top_fieles'] = df.groupby(col_cli_n)[col_fact].nunique().sort_values(ascending=False).head(10)
        else: kpis['top_fieles'] = df[col_cli_n].value_counts().head(10)
    else: kpis['top_fieles'] = None

    col_canal = mapa.get('canal')
    if col_canal and col_venta: kpis['ventas_canal'] = df.groupby(col_canal)[col_venta].sum().sort_values(ascending=False)
    else: kpis['ventas_canal'] = None
    
    return kpis

# --- AGENTE INTELIGENTE ---
def agente_inteligente_langchain(df, query, api_key):
    if not api_key: return "🔒 Por favor configura tu API Key en la barra lateral."
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)
        return agent.invoke(query)['output']
    except Exception as e: return f"Error procesando tu pregunta: {str(e)}"

# --- PDF ---
class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 16); self.cell(0, 10, 'Reporte Comercial', 0, 1, 'C'); self.ln(10)

def generar_pdf_reporte(kpis, nombre_archivo, df, api_key):
    pdf = PDFReport(); pdf.add_page(); pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Ventas Totales: ${kpis['total_ventas']:,.0f}", 0, 1)
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=api_key)
            prompt = f"Resumen breve (3 lineas): Ventas {kpis['total_ventas']}, Clientes {kpis['total_clientes']}."
            resumen_ia = llm.invoke(prompt).content
            pdf.multi_cell(0, 8, resumen_ia)
        except: pass
    return pdf.output(dest='S').encode('latin-1', 'replace')

def generar_excel_descarga(df):
    output = io.BytesIO(); 
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='Datos')
    return output.getvalue()

def formatear_fecha_es(dt):
    if not isinstance(dt, datetime): return "N/A"
    try: return f"{dt.day:02d}/{dt.month:02d}/{dt.year}"
    except: return str(dt)

def solucionar_conflictos_ia(lista_errores):
    api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key_val: st.error("🔒 Falta API Key"); return
    with st.spinner("🤖 Analizando..."):
        try:
            os.environ["GOOGLE_API_KEY"] = api_key_val
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
            st.info(llm.invoke(f"Como arreglar en Excel: {str(lista_errores[:5])}").content)
        except Exception as e: st.error(str(e))

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="header-container"><p class="header-title">Analítica Pro 🚀</p></div>', unsafe_allow_html=True)
    tipo_fuente = st.radio("Fuente:", ["Subir Archivo", "Google Sheets"], index=0)
    uploaded_files = None; df_google = None
    if tipo_fuente == "Subir Archivo": uploaded_files = st.file_uploader("Arrastra archivos", accept_multiple_files=True)
    else: sheet_url = st.text_input("Enlace Google Sheets:")
    try: api_key = st.secrets["GOOGLE_API_KEY"]
    except: api_key = st.text_input("🔑 API Key Gemini", type="password", key="api_key_input")
    if st.button("🔄 Reiniciar"): st.session_state.clear(); st.rerun()

# --- 5. LOGICA PRINCIPAL Y CARGA ---
if uploaded_files or df_google is not None:
    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if st.session_state.get("last_files") != current_names:
            all_dfs = []
            for f in uploaded_files:
                df, err = limpiar_excel_inteligente(f)
                if df is not None: all_dfs.append(df)
            if all_dfs:
                df_final = pd.concat(all_dfs, ignore_index=True)
                st.session_state.update({"df_raw": df_final, "last_files": current_names, "mapa": detecting_mapa_completo(df_final) if "detecting_mapa_completo" in globals() else detectar_mapa_completo(df_final)})
                st.rerun()
    elif df_google is not None:
        if "df_raw" not in st.session_state or st.session_state.get("last_url") != sheet_url:
            st.session_state.update({"df_raw": df_google, "last_url": sheet_url, "mapa": detectar_mapa_completo(df_google)})
            st.rerun()

# --- 6. GESTIÓN DE FILTROS ---
if "filters" not in st.session_state: st.session_state.filters = {}
def update_filter(key, value):
    if st.session_state.filters.get(key) == value: del st.session_state.filters[key]
    else: st.session_state.filters[key] = value
    st.rerun()
def clear_filters(): st.session_state.filters = {}; st.rerun()

# --- FUNCION PARA COLORES ---
def get_colors(df, col_name, filter_key):
    selected = st.session_state.filters.get(filter_key)
    if selected: return df[col_name].apply(lambda x: '#EF4444' if str(x) == str(selected) else '#E5E7EB')
    return df[col_name].apply(lambda x: '#3B82F6')

# --- 7. LAYOUT DASHBOARD ---
if "df_raw" in st.session_state:
    df_raw = st.session_state["df_raw"]; mapa = st.session_state["mapa"]
    
    # 7.1 PREPROCESAMIENTO GLOBAL DE FECHAS
    col_fecha = mapa.get('fecha')
    if col_fecha:
        df_raw = preprocesar_fechas(df_raw, col_fecha)
        mapa['dia_semana_esp'] = 'dia_semana_esp' # Agregar al mapa para que el filtro lo encuentre

    # 7.2 APLICAR FILTROS
    df_filtered = df_raw.copy(); filter_desc = []
    for col_key, val in st.session_state.filters.items():
        col_name = mapa.get(col_key)
        # Fix: Asegurar que si filtramos por dia semana, usamos la columna creada
        if col_key == 'dia_semana_esp': col_name = 'dia_semana_esp'
        
        if col_name and val:
            df_filtered = df_filtered[df_filtered[col_name] == val]
            filter_desc.append(f"{val}")
    
    kpis_filtered = calcular_kpis(df_filtered, mapa)
    kpis_global = calcular_kpis(df_raw, mapa) 
    conflictos = auditar_calidad_datos(df_raw, mapa)
    
    c_dashboard, c_chat = st.columns([3, 1], gap="medium")
    with c_dashboard:
        if filter_desc:
            col_reset, col_text = st.columns([1, 4])
            with col_reset: 
                if st.button("🗑️ Borrar Filtros", type="primary", use_container_width=True): clear_filters()
            with col_text: st.markdown(f'<div class="active-filters">🔎 Filtros Activos: {" + ".join(filter_desc)}</div>', unsafe_allow_html=True)
        else: st.caption("👈 Haz clic en cualquier gráfico para filtrar los demás.")

        # AUDITORÍA
        n_conf = len(conflictos)
        with st.expander(f"⚠️ Auditoría: {n_conf} conflictos" if n_conf > 0 else "✅ Auditoría: Datos Limpios", expanded=False):
            if n_conf > 0:
                if st.button("✨ Ayuda IA"): solucionar_conflictos_ia(conflictos)
                for c in conflictos: st.markdown(f"- {c}")
            else: st.success("Todo en orden.")

        # METRICAS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ventas Totales", f"${kpis_filtered.get('total_ventas', 0):,.0f}")
        m2.metric("Total Clientes", kpis_filtered.get('total_clientes', 0))
        m3.metric("Clientes Nuevos", kpis_filtered.get('clientes_nuevos', 0))
        m4.metric("Promedio Diario", f"${kpis_filtered.get('promedio_dia', 0):,.0f}")
        st.markdown("---")

        with st.container(border=True):
            st.subheader("📊 Análisis Visual Interactivo")
            
            # --- FILA 1: PRODUCTOS y CLIENTES ---
            tc1, tc2 = st.columns(2)
            with tc1:
                # PRODUCTOS (Usa Global para mantener contexto, resalta selección)
                if kpis_global.get('top_productos') is not None:
                    df_p = kpis_global['top_productos'].reset_index(); df_p.columns = ['Producto', 'Total_Venta']
                    fig_p = px.bar(df_p, x='Total_Venta', y='Producto', orientation='h', title="🏆 Top 10 Productos (Global)",
                                   color_discrete_sequence=get_colors(df_p, 'Producto', 'producto_nom'))
                    fig_p.update_layout(yaxis=dict(autorange="reversed"), clickmode='event+select', showlegend=False)
                    sel_p = st.plotly_chart(fig_p, use_container_width=True, on_select="rerun", selection_mode="points", key="chart_prod")
                    if sel_p.selection.points: update_filter('producto_nom', df_p.iloc[sel_p.selection.points[0]["point_index"]]['Producto'])
                else: st.info("Sin datos de productos.")

            with tc2:
                # CLIENTES (Usa Filtered para ver detalle, o Global si no hay filtros, con Highlight)
                # Estrategia: Mostrar Global con Highlight permite ver quién es el top incluso filtrando.
                if kpis_global.get('top_clientes') is not None:
                    df_c = kpis_global['top_clientes'].reset_index(); df_c.columns = ['Cliente', 'Total_Venta']
                    fig_c = px.bar(df_c, x='Total_Venta', y='Cliente', orientation='h', title="💎 Top 10 Clientes (Global)",
                                   color_discrete_sequence=get_colors(df_c, 'Cliente', 'cliente_nom'))
                    fig_c.update_layout(yaxis=dict(autorange="reversed"), clickmode='event+select', showlegend=False)
                    sel_c = st.plotly_chart(fig_c, use_container_width=True, on_select="rerun", selection_mode="points", key="chart_cli")
                    if sel_c.selection.points: update_filter('cliente_nom', df_c.iloc[sel_c.selection.points[0]["point_index"]]['Cliente'])
                else: st.info("Sin datos de clientes.")
            
            # --- FILA 2: FIDELIDAD y CANALES ---
            tc3, tc4 = st.columns(2)
            with tc3:
                # FIDELIDAD (Misma lógica: Global con Highlight)
                if kpis_global.get('top_fieles') is not None:
                    df_f = kpis_global['top_fieles'].reset_index(); df_f.columns = ['Cliente', 'Frecuencia']
                    fig_f = px.bar(df_f, x='Frecuencia', y='Cliente', orientation='h', title="❤️ Fidelidad (Frecuencia Global)",
                                   color_discrete_sequence=get_colors(df_f, 'Cliente', 'cliente_nom'))
                    fig_f.update_layout(yaxis=dict(autorange="reversed"), clickmode='event+select', showlegend=False)
                    sel_f = st.plotly_chart(fig_f, use_container_width=True, on_select="rerun", selection_mode="points", key="chart_fiel")
                    if sel_f.selection.points: update_filter('cliente_nom', df_f.iloc[sel_f.selection.points[0]["point_index"]]['Cliente'])
                else: st.info("Falta info fidelidad.")

            with tc4:
                # CANALES
                if kpis_global.get('ventas_canal') is not None:
                    df_can = kpis_global['ventas_canal'].reset_index(); df_can.columns = ['Canal', 'Ventas']
                    selected_can = st.session_state.filters.get('canal')
                    pull_list = [0.1 if c == selected_can else 0 for c in df_can['Canal']]
                    fig_can = px.pie(df_can, values='Ventas', names='Canal', title="🌐 Canales (Global)", hole=0.4)
                    fig_can.update_traces(pull=pull_list)
                    sel_can = st.plotly_chart(fig_can, use_container_width=True, on_select="rerun", selection_mode="points", key="chart_can")
                    if sel_can.selection.points: update_filter('canal', df_can.iloc[sel_can.selection.points[0]["point_index"]]['Canal'])
                else: st.info("No se detectó canal.")

            # --- SEMANAL ---
            # Usamos kpis_global['ventas_por_dia'] que ya tiene la data agrupada
            if kpis_global.get('ventas_por_dia') is not None:
                st.markdown("#### 🗓️ Análisis Semanal (Global)")
                df_week = kpis_global['ventas_por_dia']
                fig_d = px.bar(df_week, x='dia_semana_esp', y='venta_total', 
                               color_discrete_sequence=get_colors(df_week, 'dia_semana_esp', 'dia_semana_esp'))
                fig_d.update_layout(clickmode='event+select', showlegend=False)
                sel_d = st.plotly_chart(fig_d, use_container_width=True, on_select="rerun", selection_mode="points", key="chart_dia")
                if sel_d.selection.points: update_filter('dia_semana_esp', df_week.iloc[sel_d.selection.points[0]["point_index"]]['dia_semana_esp'])

        st.markdown("---")
        with st.container(border=True):
            st.subheader("📥 Exportación")
            ex1, ex2 = st.columns(2)
            api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
            pdf_bytes = generar_pdf_reporte(kpis_filtered, st.session_state.get("last_files", ["Reporte"]), df_filtered, api_key_val)
            ex1.download_button("📄 PDF Reporte", pdf_bytes, "reporte.pdf")
            xls_bytes = generar_excel_descarga(df_filtered)
            ex2.download_button("📊 Excel Data (Filtrada)", xls_bytes, "data.xlsx")
            with st.expander("Ver Datos"): st.dataframe(df_filtered.head(50), use_container_width=True)

        with st.container(border=True):
            st.subheader("🕰️ Snapshot"); 
            if st.button("Guardar hoy"): guardar_en_memoria("Manual", kpis_filtered, datetime.now().strftime("%Y-%m-%d")); st.success("Guardado")
            hist = obtener_historia(); 
            if not hist.empty: st.dataframe(hist, use_container_width=True)
            if st.button("Borrar Historial"): borrar_historia(); st.rerun()

    with c_chat:
        st.markdown('<div class="chat-panel"><h3>🤖 Asistente</h3>', unsafe_allow_html=True)
        if "messages" not in st.session_state: st.session_state.messages = []
        chat_cont = st.container(height=500)
        with chat_cont:
            if not st.session_state.messages: st.info("Pregunta sobre tus datos filtrados.")
            for m in st.session_state.messages: st.markdown(f'<div class="chat-msg-{ "user" if m["role"]=="user" else "ai"}">{m["content"]}</div>', unsafe_allow_html=True)
        if p := st.chat_input("..."):
            st.session_state.messages.append({"role": "user", "content": p}); st.rerun()
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with chat_cont:
                with st.spinner("..."):
                    api_key_val = st.session_state.get("api_key_input") or st.secrets.get("GOOGLE_API_KEY")
                    res = agente_inteligente_langchain(df_filtered, st.session_state.messages[-1]["content"], api_key_val)
                    st.session_state.messages.append({"role": "assistant", "content": res}); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
