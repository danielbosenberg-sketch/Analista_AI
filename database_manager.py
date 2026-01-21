import sqlite3
import pandas as pd
from datetime import datetime

# Nombre del archivo donde se guardarán los recuerdos
DB_NAME = "memoria_negocio.db"

def init_db():
    """Crea la tabla y asegura que tenga la estructura correcta"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Crear tabla si no existe
    c.execute('''
        CREATE TABLE IF NOT EXISTS ventas_historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_carga TEXT,
            fecha_negocio TEXT,
            nombre_archivo TEXT,
            venta_total REAL,
            total_clientes INTEGER,
            promedio_dia REAL,
            mejor_producto TEXT
        )
    ''')
    
    # 2. Migración automática (por si la tabla es vieja)
    try:
        c.execute("ALTER TABLE ventas_historico ADD COLUMN fecha_negocio TEXT")
    except sqlite3.OperationalError:
        pass 
        
    conn.commit()
    conn.close()

def guardar_en_memoria(nombre_archivo, kpis, fecha_real_datos):
    """Guarda el resumen usando la fecha real del negocio"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    fecha_hoy = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mejor_prod = kpis['top_productos'].index[0] if kpis['top_productos'] is not None else "N/A"
    
    c.execute('''
        INSERT INTO ventas_historico (fecha_carga, fecha_negocio, nombre_archivo, venta_total, total_clientes, promedio_dia, mejor_producto)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (fecha_hoy, fecha_real_datos, nombre_archivo, kpis['total_ventas'], kpis['total_clientes'], kpis['promedio_dia'], mejor_prod))
    
    conn.commit()
    conn.close()
    return True

def obtener_historia():
    """Recupera la historia ordenada por la fecha del negocio"""
    conn = sqlite3.connect(DB_NAME)
    try:
        df_historia = pd.read_sql_query("SELECT * FROM ventas_historico ORDER BY fecha_negocio ASC", conn)
    except:
        df_historia = pd.DataFrame()
    conn.close()
    return df_historia

def borrar_historia():
    """⚠️ Borra TODOS los registros de la base de datos"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM ventas_historico")
    conn.commit()
    conn.close()