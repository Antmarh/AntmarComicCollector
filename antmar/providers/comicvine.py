# antmar/providers/comicvine.py
from __future__ import annotations
import os
import time
import requests
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Tuple

from antmar import config
from antmar.metadata import generate_comicinfo_xml

# --- Ajustes básicos ---
_API_BASE = "https://comicvine.gamespot.com/api"
# ComicVine recomienda 1 req/seg para apps gratuitas
_MIN_INTERVAL_S = 1.05
_last_call_ts = 0.0

class ComicVineError(RuntimeError):
    pass

def _throttle():
    global _last_call_ts
    now = time.time()
    if now - _last_call_ts < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - (now - _last_call_ts))
    _last_call_ts = time.time()

def _get_api_key(prompt_key=None) -> str:
    """
    Obtiene la API key de ComicVine.
    - Primero intenta leerla de config.ini (APPDATA/AntmarComicCollector/config.ini).
    - Si no existe y se pasa 'prompt_key' (callable que pida la clave al usuario),
      la solicita una vez y la guarda.
    """
    k = config.get("COMICVINE")
    if k:
        return k
    if callable(prompt_key):
        val = prompt_key("Introduce tu API key de ComicVine:")
        if not val:
            raise ComicVineError("No se configuró la API key de ComicVine.")
        config.set("COMICVINE", val)
        return val
    raise ComicVineError("Falta API key de ComicVine. Proporciónala con 'prompt_key' o guárdala en config.")

def _request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    _throttle()
    headers = {"User-Agent": "AntmarComicCollector/1.0 (+github.com/Antmarh/AntmarComicCollector)"}
    r = requests.get(f"{_API_BASE}/{endpoint}", params=params, headers=headers, timeout=15)
    if r.status_code == 401:
        raise ComicVineError("API key inválida o sin permisos (401).")
    r.raise_for_status()
    data = r.json()
    if data.get("status_code") != 1:  # 1 = OK
        raise ComicVineError(f"ComicVine error: {data.get('error')} (code {data.get('status_code')})")
    return data

# --------------------------
# BÚSQUEDA
# --------------------------
def search_issues(query: str, api_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Busca issues por texto. Devuelve una lista con datos básicos.
    """
    if not query:
        return []
    params = {
        "api_key": api_key,
        "format": "json",
        "resources": "issue",
        "query": query,
        "limit": limit,
        "field_list": ",".join([
            "id","name","issue_number","cover_date","site_detail_url",
            "volume","image","person_credits","team_credits","publisher"
        ]),
        "sort": "cover_date:desc"
    }
    data = _request("search", params)
    return data.get("results", []) or []

def search_volumes(query: str, api_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Busca volúmenes (series). Útil si prefieres elegir serie y luego issue.
    """
    if not query:
        return []
    params = {
        "api_key": api_key,
        "format": "json",
        "resources": "volume",
        "query": query,
        "limit": limit,
        "field_list": ",".join([
            "id","name","publisher","count_of_issues","start_year","site_detail_url","image"
        ]),
        "sort": "start_year:desc"
    }
    data = _request("search", params)
    return data.get("results", []) or []

# --------------------------
# DETALLES DE ISSUE
# --------------------------
def get_issue_details(issue_id: int, api_key: str) -> Dict[str, Any]:
    """
    Obtiene la ficha completa de un issue (id numérico interno de ComicVine).
    ComicVine usa el endpoint issue/4000-<id>.
    """
    params = {
        "api_key": api_key,
        "format": "json"
    }
    data = _request(f"issue/4000-{issue_id}", params)
    return data.get("results") or {}

# --------------------------
# MAPEADO A ComicInfo
# --------------------------
def _people_list(credits: Optional[List[Dict[str, Any]]], role_field="name") -> str:
    if not credits:
        return ""
    names = [c.get(role_field) for c in credits if c.get(role_field)]
    # eliminar duplicados manteniendo orden
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return ", ".join(out)

def comicvine_issue_to_comicinfo(issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte la respuesta de un issue ComicVine a un dict listo para ComicInfo.xml
    (campos comunes; ajusta según tu flujo).
    """
    volume = issue.get("volume") or {}
    publisher = volume.get("publisher") or issue.get("publisher") or {}
    # Personas por rol (ComicVine mezcla en person_credits)
    persons = issue.get("person_credits") or []
    def role(role_name: str) -> List[Dict[str, Any]]:
        return [p for p in persons if p.get("role","").lower() == role_name]

    data = {
        "Title": issue.get("name") or "",
        "Series": volume.get("name") or "",
        "Number": issue.get("issue_number") or "",
        "Year": (issue.get("cover_date") or "")[:4],
        "Month": (issue.get("cover_date") or "")[5:7],
        "Day": (issue.get("cover_date") or "")[8:10],
        "Publisher": (publisher.get("name") if isinstance(publisher, dict) else publisher) or "",
        "Web": issue.get("site_detail_url") or "",
        "Summary": "",  # si quieres, puedes usar "deck" o "description" limpio de HTML
        # Créditos (ComicInfo estándar)
        "Writer": _people_list(role("writer")),
        "Penciller": _people_list(role("penciller")),
        "Inker": _people_list(role("inker")),
        "Colorist": _people_list(role("colorist")),
        "Letterer": _people_list(role("letterer")),
        "Editor": _people_list(role("editor")),
        # Campos opcionales
        "CoverURL": ((issue.get("image") or {}).get("original_url") or ""),
    }
    # Limpieza básica
    for k, v in list(data.items()):
        if isinstance(v, str):
            data[k] = v.strip()
    return data

# --------------------------
# FUNCIÓN PRINCIPAL
# --------------------------
def get_comicvine_details(
    query: Optional[str] = None,
    issue_id: Optional[int] = None,
    *,
    prompt_key=None,
    return_xml: bool = True,
    limit_search: int = 10
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Uso:
      - Por texto: get_comicvine_details(query="Amazing Spider-Man 129")
      - Por id:    get_comicvine_details(issue_id=12345)

    Parámetros:
      prompt_key: callable(str)->str para pedir y guardar la API key si falta.
      return_xml: si True, también devuelve el XML ComicInfo.
    Devuelve: (dict_comicinfo, xml_str|None)
    """
    api_key = _get_api_key(prompt_key)

    if issue_id:
        issue = get_issue_details(int(issue_id), api_key)
        if not issue:
            raise ComicVineError("No se encontró el issue indicado.")
    else:
        results = search_issues(query or "", api_key, limit=limit_search)
        if not results:
            raise ComicVineError("Sin resultados para la búsqueda.")
        # Elegimos el primero (más reciente por cover_date desc). Ajusta si prefieres otro criterio.
        issue = get_issue_details(int(results[0]["id"]), api_key)

    data = comicvine_issue_to_comicinfo(issue)

    xml = generate_comicinfo_xml(data) if return_xml else None
    return data, xml
