import sqlite3
import os
import re
import uuid
import unicodedata
import zipfile
import io
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CLASES_DATA_DIR", os.path.join(BASE_DIR, "datos"))
FILES_DIR = os.path.join(DATA_DIR, "archivos")
DB_PATH = os.path.join(DATA_DIR, "clases.db")

TIPOS = ["Presentación", "Guía / Taller", "Lectura", "Examen / Quiz", "Rúbrica",
         "Programa / Syllabus", "Notas", "Notas de clase", "Video / Enlace", "Otro"]


def _conn():
    os.makedirs(FILES_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS cursos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            codigo TEXT,
            periodo TEXT NOT NULL,
            grupo TEXT,
            descripcion TEXT,
            archivado INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS unidades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            curso_id INTEGER NOT NULL REFERENCES cursos(id) ON DELETE CASCADE,
            nombre TEXT NOT NULL,
            orden INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            curso_id INTEGER NOT NULL REFERENCES cursos(id) ON DELETE CASCADE,
            unidad_id INTEGER REFERENCES unidades(id) ON DELETE SET NULL,
            titulo TEXT NOT NULL,
            tipo TEXT,
            fecha TEXT,
            etiquetas TEXT,
            notas TEXT,
            enlace TEXT,
            archivo TEXT,
            nombre_original TEXT,
            creado TEXT
        );
        """)


# ---------- Cursos ----------
def listar_cursos(incluir_archivados=False):
    q = "SELECT * FROM cursos"
    if not incluir_archivados:
        q += " WHERE archivado = 0"
    q += " ORDER BY periodo DESC, nombre"
    with _conn() as c:
        return c.execute(q).fetchall()


def crear_curso(nombre, periodo, codigo="", grupo="", descripcion=""):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO cursos (nombre, codigo, periodo, grupo, descripcion) VALUES (?,?,?,?,?)",
            (nombre.strip(), codigo.strip(), periodo.strip(), grupo.strip(), descripcion.strip()))
        return cur.lastrowid


def actualizar_curso(curso_id, nombre, periodo, codigo, grupo, descripcion, archivado):
    with _conn() as c:
        c.execute("""UPDATE cursos SET nombre=?, periodo=?, codigo=?, grupo=?, descripcion=?, archivado=?
                     WHERE id=?""",
                  (nombre.strip(), periodo.strip(), codigo.strip(), grupo.strip(),
                   descripcion.strip(), int(archivado), curso_id))


def eliminar_curso(curso_id):
    with _conn() as c:
        rutas = [r["archivo"] for r in c.execute(
            "SELECT archivo FROM documentos WHERE curso_id=? AND archivo IS NOT NULL", (curso_id,))]
        c.execute("DELETE FROM cursos WHERE id=?", (curso_id,))
    for r in rutas:
        _borrar_archivo(r)


# ---------- Unidades ----------
def listar_unidades(curso_id):
    with _conn() as c:
        return c.execute("SELECT * FROM unidades WHERE curso_id=? ORDER BY orden, id",
                         (curso_id,)).fetchall()


def crear_unidad(curso_id, nombre):
    with _conn() as c:
        n = c.execute("SELECT COALESCE(MAX(orden),0)+1 FROM unidades WHERE curso_id=?",
                      (curso_id,)).fetchone()[0]
        c.execute("INSERT INTO unidades (curso_id, nombre, orden) VALUES (?,?,?)",
                  (curso_id, nombre.strip(), n))


def renombrar_unidad(unidad_id, nombre):
    with _conn() as c:
        c.execute("UPDATE unidades SET nombre=? WHERE id=?", (nombre.strip(), unidad_id))


def eliminar_unidad(unidad_id):
    with _conn() as c:
        c.execute("DELETE FROM unidades WHERE id=?", (unidad_id,))


# ---------- Documentos ----------
def _limpiar_nombre(nombre):
    nombre = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", nombre).strip("_") or "archivo"


def _borrar_archivo(ruta_rel):
    try:
        os.remove(os.path.join(FILES_DIR, ruta_rel))
    except OSError:
        pass


def agregar_documento(curso_id, titulo, tipo, unidad_id=None, fecha=None, etiquetas="",
                      notas="", enlace="", archivo_nombre=None, archivo_bytes=None):
    ruta = None
    if archivo_bytes is not None:
        carpeta = os.path.join(FILES_DIR, str(curso_id))
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(str(curso_id), f"{uuid.uuid4().hex[:8]}_{_limpiar_nombre(archivo_nombre)}")
        with open(os.path.join(FILES_DIR, ruta), "wb") as f:
            f.write(archivo_bytes)
    with _conn() as c:
        c.execute("""INSERT INTO documentos
            (curso_id, unidad_id, titulo, tipo, fecha, etiquetas, notas, enlace, archivo, nombre_original, creado)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                  (curso_id, unidad_id, titulo.strip(), tipo, fecha, etiquetas.strip(), notas.strip(),
                   enlace.strip(), ruta, archivo_nombre, datetime.now().isoformat(timespec="seconds")))


def buscar_documentos(curso_id=None, unidad_id=None, tipo=None, texto=""):
    q = """SELECT d.*, u.nombre AS unidad, c.nombre AS curso, c.periodo, c.grupo
           FROM documentos d
           JOIN cursos c ON c.id = d.curso_id
           LEFT JOIN unidades u ON u.id = d.unidad_id WHERE 1=1"""
    p = []
    if curso_id:
        q += " AND d.curso_id=?"; p.append(curso_id)
    if unidad_id == -1:
        q += " AND d.unidad_id IS NULL"
    elif unidad_id:
        q += " AND d.unidad_id=?"; p.append(unidad_id)
    if tipo:
        q += " AND d.tipo=?"; p.append(tipo)
    if texto:
        q += " AND (d.titulo LIKE ? OR d.etiquetas LIKE ? OR d.notas LIKE ? OR d.nombre_original LIKE ?)"
        p += [f"%{texto}%"] * 4
    q += " ORDER BY COALESCE(u.orden, 9999), d.fecha DESC, d.id DESC"
    with _conn() as c:
        return c.execute(q, p).fetchall()


def actualizar_documento(doc_id, titulo, tipo, unidad_id, fecha, etiquetas, notas, enlace):
    with _conn() as c:
        c.execute("""UPDATE documentos SET titulo=?, tipo=?, unidad_id=?, fecha=?, etiquetas=?, notas=?, enlace=?
                     WHERE id=?""",
                  (titulo.strip(), tipo, unidad_id, fecha, etiquetas.strip(), notas.strip(),
                   enlace.strip(), doc_id))


def eliminar_documento(doc_id):
    with _conn() as c:
        row = c.execute("SELECT archivo FROM documentos WHERE id=?", (doc_id,)).fetchone()
        c.execute("DELETE FROM documentos WHERE id=?", (doc_id,))
    if row and row["archivo"]:
        _borrar_archivo(row["archivo"])


def leer_archivo(ruta_rel):
    with open(os.path.join(FILES_DIR, ruta_rel), "rb") as f:
        return f.read()


# ---------- Exportación ----------
def exportar_zip(curso_id=None):
    """ZIP con estructura Periodo/Curso/Unidad/archivo. Todos los cursos si curso_id es None."""
    buf = io.BytesIO()
    docs = [d for d in buscar_documentos(curso_id=curso_id) if d["archivo"]]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for d in docs:
            grupo = f"_G{d['grupo']}" if d["grupo"] else ""
            partes = [_limpiar_nombre(d["periodo"]), _limpiar_nombre(d["curso"] + grupo),
                      _limpiar_nombre(d["unidad"] or "Sin_unidad"), _limpiar_nombre(d["nombre_original"])]
            try:
                z.writestr("/".join(partes), leer_archivo(d["archivo"]))
            except OSError:
                continue
    return buf.getvalue()
