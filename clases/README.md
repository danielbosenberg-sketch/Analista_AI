# Mis Clases

Plataforma para organizar los documentos de las clases: Periodo → Curso → Unidad → Documento.

    pip install -r clases/requirements.txt
    streamlit run clases/app.py

## Almacenamiento

- **Local (por defecto):** `clases/datos/` (SQLite + archivos). Cámbialo con `CLASES_DATA_DIR`.
- **Google Drive:** si existe la sección `[drive]` en los secrets de Streamlit, los archivos se suben a
  `Mis Clases/<Periodo>/<Curso>/` y la base de datos se respalda en `Mis Clases/clases.db` tras cada cambio.
  Si el disco local se borra, la app restaura la base desde Drive al arrancar.
  Con el scope `drive.file` la app solo ve lo que ella misma creó.

### Configurar Drive (una vez)

1. En [Google Cloud Console](https://console.cloud.google.com/): crea un proyecto, habilita **Google Drive API**.
2. *APIs y servicios → Pantalla de consentimiento OAuth*: tipo Externo, agrega tu correo como usuario de prueba.
3. *Credenciales → Crear credenciales → ID de cliente OAuth → Aplicación de escritorio*; descarga el JSON.
4. En tu computador: `pip install google-auth-oauthlib` y `python clases/obtener_token_drive.py client_secret.json`.
5. Pega la salida en `.streamlit/secrets.toml` (local) o en *Secrets* de Streamlit Cloud / Codespaces.

Nota: con la app OAuth en modo "Testing", Google caduca el refresh token a los 7 días; para que no caduque,
publica la app (*En producción*) en la pantalla de consentimiento (basta para uso personal).

Limitación: una sola persona a la vez (no hay bloqueo si dos sesiones escriben a la vez).
