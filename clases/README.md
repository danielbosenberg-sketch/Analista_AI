# Mis Clases

Plataforma para organizar los documentos de las clases: Periodo → Curso → Unidad → Documento.

    pip install -r clases/requirements.txt
    streamlit run clases/app.py

Los datos quedan en `clases/datos/` (SQLite + archivos). Cámbialo con la variable `CLASES_DATA_DIR`.
Esa carpeta no se sube a git: haz respaldo con el botón ZIP de la app.
