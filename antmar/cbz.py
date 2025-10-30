# antmar/cbz.py
from __future__ import annotations
import io, os, re, zipfile
from typing import Optional, Tuple, List
from PIL import Image

# --- Leer ComicInfo.xml de un CBZ ---
def read_comicinfo_from_cbz(cbz_path: str) -> Optional[str]:
    if not os.path.exists(cbz_path): return None
    try:
        with zipfile.ZipFile(cbz_path, 'r') as zf:
            # nombres exactos más comunes
            for name in zf.namelist():
                if name.lower() == 'comicinfo.xml' or name.lower().endswith('/comicinfo.xml'):
                    with zf.open(name, 'r') as f:
                        return f.read().decode('utf-8', errors='replace')
    except Exception:
        return None
    return None

# --- Inyectar ComicInfo.xml en un CBZ ---
def inject_xml_into_cbz(cbz_path: str, comicinfo_xml: str) -> bool:
    if not os.path.exists(cbz_path): return False
    tmp_path = cbz_path + '.tmp'
    try:
        with zipfile.ZipFile(cbz_path, 'r') as src, zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as dst:
            # Copia todo excepto ComicInfo.xml previo
            for item in src.infolist():
                name_lower = item.filename.lower()
                if name_lower == 'comicinfo.xml' or name_lower.endswith('/comicinfo.xml'):
                    continue
                data = src.read(item.filename)
                dst.writestr(item, data)
            # Escribe el nuevo ComicInfo.xml en raíz
            dst.writestr('ComicInfo.xml', comicinfo_xml.encode('utf-8'))
        os.replace(tmp_path, cbz_path)
        return True
    except Exception:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        return False

# --- Heurística para portada: primera imagen "pequeña" por nombre ---
_IMG_EXT = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}

def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in _IMG_EXT

def get_cover_from_cbz(cbz_path: str) -> Optional[Image.Image]:
    if not os.path.exists(cbz_path): return None
    try:
        with zipfile.ZipFile(cbz_path, 'r') as zf:
            names = [n for n in zf.namelist() if _is_image(n)]
            if not names: return None
            # orden natural (01, 2, 10…)
            def _natkey(s: str):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
            names.sort(key=_natkey)
            # abre la primera
            with zf.open(names[0]) as f:
                data = f.read()
            return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception:
        return None
