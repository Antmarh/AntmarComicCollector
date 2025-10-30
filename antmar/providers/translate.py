import os
import requests
from antmar.utils import natural_sort_key   # opcional, solo si lo usas
import deepl
from antmar import config

def _get_key():
    k = config.get("DEEPL")
    if k: return k
    from legacy.metaB import require_api_key_once
    return require_api_key_once("DEEPL", "Introduce tu API key de DeepL:")


def get_translator():
    global translator
    if translator is not None: return translator
    if not DEEPL_API_KEY or "TU_CLAVE_API" in DEEPL_API_KEY: return None
    try:
        import deepl
        translator = deepl.Translator(DEEPL_API_KEY)
        print("Traductor de DeepL inicializado por primera vez.")
        return translator
    except Exception as e:
        messagebox.showerror("Error de DeepL", f"No se pudo inicializar el traductor. Verifica tu clave API y conexión.\n\nError: {e}")
        return None