"""Cliente mínimo de Google Drive (scope drive.file: solo ve lo que crea esta app)."""
import io

RAIZ = "Mis Clases"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
MIME_CARPETA = "application/vnd.google-apps.folder"


class DriveClient:
    def __init__(self, cfg):
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(None, refresh_token=cfg["refresh_token"], client_id=cfg["client_id"],
                            client_secret=cfg["client_secret"],
                            token_uri="https://oauth2.googleapis.com/token", scopes=SCOPES)
        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._cache = {}

    def _buscar(self, nombre, padre=None, carpeta=False):
        nombre = nombre.replace("\\", "\\\\").replace("'", "\\'")
        q = f"name = '{nombre}' and trashed = false"
        q += f" and mimeType {'=' if carpeta else '!='} '{MIME_CARPETA}'"
        if padre:
            q += f" and '{padre}' in parents"
        r = self.svc.files().list(q=q, fields="files(id)", pageSize=1).execute()
        return r["files"][0]["id"] if r["files"] else None

    def carpeta(self, partes):
        """Id de la carpeta Mis Clases/<partes...>, creándola si no existe."""
        padre = None
        for nombre in [RAIZ] + list(partes):
            clave = (padre, nombre)
            if clave not in self._cache:
                fid = self._buscar(nombre, padre, carpeta=True)
                if not fid:
                    meta = {"name": nombre, "mimeType": MIME_CARPETA}
                    if padre:
                        meta["parents"] = [padre]
                    fid = self.svc.files().create(body=meta, fields="id").execute()["id"]
                self._cache[clave] = fid
            padre = self._cache[clave]
        return padre

    def buscar(self, nombre, padre):
        return self._buscar(nombre, padre)

    def subir(self, padre, nombre, datos, file_id=None):
        from googleapiclient.http import MediaIoBaseUpload
        media = MediaIoBaseUpload(io.BytesIO(datos), mimetype="application/octet-stream", resumable=True)
        if file_id:
            return self.svc.files().update(fileId=file_id, media_body=media, fields="id").execute()["id"]
        meta = {"name": nombre, "parents": [padre]}
        return self.svc.files().create(body=meta, media_body=media, fields="id").execute()["id"]

    def descargar(self, file_id):
        return self.svc.files().get_media(fileId=file_id).execute()

    def borrar(self, file_id):
        self.svc.files().delete(fileId=file_id).execute()
