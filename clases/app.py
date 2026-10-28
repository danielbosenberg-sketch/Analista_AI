import os
import sys
from datetime import date

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

st.set_page_config(page_title="Mis Clases", page_icon="📚", layout="wide")
db.init_db()

ICONOS = {"Presentación": "📊", "Guía / Taller": "📝", "Lectura": "📖", "Examen / Quiz": "✅",
          "Rúbrica": "📐", "Programa / Syllabus": "🗂️", "Notas": "🗒️", "Notas de clase": "🗒️",
          "Video / Enlace": "🔗", "Otro": "📄"}


def etiqueta_curso(c):
    extra = f" · G{c['grupo']}" if c["grupo"] else ""
    cod = f"{c['codigo']} – " if c["codigo"] else ""
    return f"{c['periodo']} | {cod}{c['nombre']}{extra}"


# ---------------- Sidebar ----------------
st.sidebar.title("📚 Mis Clases")
ver_arch = st.sidebar.checkbox("Mostrar cursos archivados")
cursos = db.listar_cursos(incluir_archivados=ver_arch)

with st.sidebar.expander("➕ Nuevo curso", expanded=not cursos):
    with st.form("nuevo_curso", clear_on_submit=True):
        n_nombre = st.text_input("Materia *")
        n_periodo = st.text_input("Periodo *", value=f"{date.today().year}-{1 if date.today().month <= 6 else 2}")
        n_codigo = st.text_input("Código")
        n_grupo = st.text_input("Grupo")
        n_desc = st.text_area("Descripción")
        if st.form_submit_button("Crear curso"):
            if n_nombre.strip() and n_periodo.strip():
                db.crear_curso(n_nombre, n_periodo, n_codigo, n_grupo, n_desc)
                st.rerun()
            else:
                st.error("Materia y periodo son obligatorios.")

st.sidebar.divider()
st.sidebar.download_button("⬇️ Respaldo de todos los archivos (ZIP)", db.exportar_zip(),
                           file_name="respaldo_clases.zip", mime="application/zip")

if not cursos:
    st.title("📚 Mis Clases")
    st.info("Crea tu primer curso desde la barra lateral para empezar a organizar tus documentos.")
    st.stop()

opciones = {etiqueta_curso(c): c for c in cursos}
sel = st.sidebar.radio("Curso", list(opciones.keys()))
curso = opciones[sel]
unidades = db.listar_unidades(curso["id"])
mapa_unidades = {u["nombre"]: u["id"] for u in unidades}

st.title(f"{curso['nombre']}")
st.caption(f"Periodo {curso['periodo']}"
           + (f" · Código {curso['codigo']}" if curso["codigo"] else "")
           + (f" · Grupo {curso['grupo']}" if curso["grupo"] else ""))
if curso["descripcion"]:
    st.write(curso["descripcion"])

tab_docs, tab_subir, tab_unidades, tab_buscar, tab_cfg = st.tabs(
    ["📂 Documentos", "⬆️ Subir", "🧱 Unidades", "🔎 Buscar en todo", "⚙️ Curso"])


def tarjeta_documento(d, key_prefix, mostrar_curso=False):
    icono = ICONOS.get(d["tipo"], "📄")
    cab = f"{icono} {d['titulo']}"
    if mostrar_curso:
        cab += f"  —  {d['curso']} ({d['periodo']})"
    with st.expander(cab):
        meta = [d["tipo"] or "", d["unidad"] or "Sin unidad", d["fecha"] or ""]
        st.caption(" · ".join(m for m in meta if m))
        if d["etiquetas"]:
            st.write("🏷️ " + d["etiquetas"])
        if d["notas"]:
            st.write(d["notas"])
        if d["enlace"]:
            st.markdown(f"🔗 [{d['enlace']}]({d['enlace']})")
        if d["archivo"]:
            try:
                st.download_button(f"Descargar {d['nombre_original']}", db.leer_archivo(d["archivo"]),
                                   file_name=d["nombre_original"], key=f"{key_prefix}dl{d['id']}")
            except OSError:
                st.warning("El archivo no se encuentra en el disco.")
        c1, c2 = st.columns(2)
        with c1:
            if st.toggle("Editar", key=f"{key_prefix}ed{d['id']}"):
                with st.form(f"{key_prefix}f{d['id']}"):
                    t = st.text_input("Título", d["titulo"])
                    tp = st.selectbox("Tipo", db.TIPOS,
                                      index=db.TIPOS.index(d["tipo"]) if d["tipo"] in db.TIPOS else 0)
                    nombres = ["(sin unidad)"] + [u["nombre"] for u in db.listar_unidades(d["curso_id"])]
                    idx = nombres.index(d["unidad"]) if d["unidad"] in nombres else 0
                    un = st.selectbox("Unidad", nombres, index=idx)
                    ids = {u["nombre"]: u["id"] for u in db.listar_unidades(d["curso_id"])}
                    f = st.text_input("Fecha (AAAA-MM-DD)", d["fecha"] or "")
                    et = st.text_input("Etiquetas (separadas por coma)", d["etiquetas"] or "")
                    nt = st.text_area("Notas", d["notas"] or "")
                    en = st.text_input("Enlace", d["enlace"] or "")
                    if st.form_submit_button("Guardar cambios"):
                        db.actualizar_documento(d["id"], t, tp, ids.get(un), f or None, et, nt, en)
                        st.rerun()
        with c2:
            if st.button("🗑️ Eliminar", key=f"{key_prefix}del{d['id']}"):
                db.eliminar_documento(d["id"])
                st.rerun()


# ---------------- Documentos ----------------
with tab_docs:
    f1, f2, f3 = st.columns([2, 2, 3])
    f_unidad = f1.selectbox("Unidad", ["Todas", "Sin unidad"] + list(mapa_unidades.keys()))
    f_tipo = f2.selectbox("Tipo", ["Todos"] + db.TIPOS)
    f_texto = f3.text_input("Buscar en este curso")
    uid = None if f_unidad == "Todas" else (-1 if f_unidad == "Sin unidad" else mapa_unidades[f_unidad])
    docs = db.buscar_documentos(curso["id"], uid, None if f_tipo == "Todos" else f_tipo, f_texto)
    st.write(f"**{len(docs)}** documento(s)")
    actual = object()
    for d in docs:
        if d["unidad"] != actual:
            actual = d["unidad"]
            st.subheader(actual or "Sin unidad")
        tarjeta_documento(d, "c")
    st.download_button("⬇️ Descargar este curso (ZIP por unidades)", db.exportar_zip(curso["id"]),
                       file_name=f"{curso['periodo']}_{curso['nombre']}.zip".replace(" ", "_"),
                       mime="application/zip")

# ---------------- Subir ----------------
with tab_subir:
    with st.form("subir", clear_on_submit=True):
        archivos = st.file_uploader("Archivos (puedes subir varios)", accept_multiple_files=True)
        titulo = st.text_input("Título (si subes varios, se usa el nombre de cada archivo)")
        c1, c2, c3 = st.columns(3)
        tipo = c1.selectbox("Tipo", db.TIPOS)
        unidad = c2.selectbox("Unidad", ["(sin unidad)"] + list(mapa_unidades.keys()))
        fecha = c3.date_input("Fecha de la clase", value=date.today())
        etiquetas = st.text_input("Etiquetas (separadas por coma)")
        notas = st.text_area("Notas")
        enlace = st.text_input("Enlace externo (Drive, YouTube, etc.) — opcional")
        if st.form_submit_button("Guardar"):
            uid = mapa_unidades.get(unidad)
            if archivos:
                for a in archivos:
                    nombre = titulo if (titulo and len(archivos) == 1) else os.path.splitext(a.name)[0]
                    db.agregar_documento(curso["id"], nombre, tipo, uid, fecha.isoformat(), etiquetas,
                                         notas, enlace, a.name, a.getvalue())
                st.success(f"{len(archivos)} archivo(s) guardado(s).")
            elif enlace.strip() and titulo.strip():
                db.agregar_documento(curso["id"], titulo, tipo, uid, fecha.isoformat(), etiquetas, notas, enlace)
                st.success("Enlace guardado.")
            else:
                st.error("Sube un archivo, o indica título y enlace.")

# ---------------- Unidades ----------------
with tab_unidades:
    st.write("Unidades, temas o semanas con las que organizas el curso.")
    with st.form("nueva_unidad", clear_on_submit=True):
        nu = st.text_input("Nueva unidad (ej. 'Unidad 1 – Fundamentos')")
        if st.form_submit_button("Agregar") and nu.strip():
            db.crear_unidad(curso["id"], nu)
            st.rerun()
    for u in unidades:
        c1, c2, c3 = st.columns([5, 1, 1])
        nuevo = c1.text_input("Nombre", u["nombre"], key=f"un{u['id']}", label_visibility="collapsed")
        if c2.button("Guardar", key=f"sv{u['id']}") and nuevo.strip() and nuevo != u["nombre"]:
            db.renombrar_unidad(u["id"], nuevo)
            st.rerun()
        if c3.button("🗑️", key=f"dl{u['id']}"):
            db.eliminar_unidad(u["id"])  # los documentos quedan "sin unidad"
            st.rerun()

# ---------------- Buscar en todo ----------------
with tab_buscar:
    q = st.text_input("Buscar en todos tus cursos (título, etiquetas, notas, nombre de archivo)")
    tp = st.selectbox("Tipo", ["Todos"] + db.TIPOS, key="bt")
    if q or tp != "Todos":
        res = db.buscar_documentos(tipo=None if tp == "Todos" else tp, texto=q)
        st.write(f"**{len(res)}** resultado(s)")
        for d in res:
            tarjeta_documento(d, "b", mostrar_curso=True)

# ---------------- Config curso ----------------
with tab_cfg:
    with st.form("cfg"):
        nombre = st.text_input("Materia", curso["nombre"])
        periodo = st.text_input("Periodo", curso["periodo"])
        codigo = st.text_input("Código", curso["codigo"] or "")
        grupo = st.text_input("Grupo", curso["grupo"] or "")
        desc = st.text_area("Descripción", curso["descripcion"] or "")
        arch = st.checkbox("Archivado (oculto al terminar el semestre)", bool(curso["archivado"]))
        if st.form_submit_button("Guardar"):
            db.actualizar_curso(curso["id"], nombre, periodo, codigo, grupo, desc, arch)
            st.rerun()
    st.divider()
    confirmar = st.checkbox("Confirmo que quiero borrar este curso y todos sus archivos")
    if st.button("🗑️ Eliminar curso", disabled=not confirmar):
        db.eliminar_curso(curso["id"])
        st.rerun()
