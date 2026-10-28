"""Genera el refresh_token de Google Drive. Se corre UNA vez en tu computador:

    pip install google-auth-oauthlib
    python clases/obtener_token_drive.py ruta/al/client_secret.json
"""
import json
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

flujo = InstalledAppFlow.from_client_secrets_file(sys.argv[1], SCOPES)
creds = flujo.run_local_server(port=0, access_type="offline", prompt="consent")
print("\nPega esto en .streamlit/secrets.toml:\n")
print("[drive]")
print(f'client_id = {json.dumps(creds.client_id)}')
print(f'client_secret = {json.dumps(creds.client_secret)}')
print(f'refresh_token = {json.dumps(creds.refresh_token)}')
