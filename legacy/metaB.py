# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
import sys
import io
import os
import re
from antmar.providers.comicvine import get_comicvine_details
from antmar.providers.translate import get_translator


from antmar.cbz import inject_xml_into_cbz, read_comicinfo_from_cbz, get_cover_from_cbz  # ← si esas funciones ya existen; si no, bórralas también por ahora

from antmar.utils import (
    natural_sort_key,
    get_local_ip
)
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
from antmar.metadata import generate_comicinfo_xml

from antmar import config
from tkinter import simpledialog, messagebox

def require_api_key_once(vendor, prompt_text):
    key = config.get(vendor.upper(), "API_KEYS", "")
    if key: return key
    # pedir solo si no existe
    val = simpledialog.askstring("Configurar API", prompt_text, show="*")
    if not val:
        messagebox.showwarning("API", "No se configuró la clave. Operación cancelada.")
        return ""
    config.set(vendor.upper(), val, "API_KEYS")
    return val



class AIMetadataDialog:
    """Diálogo para configurar la generación de metadatos con IA"""
    
    def __init__(self, parent, file_path):
        self.parent = parent
        self.file_path = file_path
        self.result = None
        
        self.top = tk.Toplevel(parent)
        self.top.title("Generar Metadatos con IA")
        self.top.geometry("520x650")
        self.top.resizable(True, True)  # Permitir redimensionar
        self.top.transient(parent)
        self.top.grab_set()
        
        # Centrar la ventana
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (520 // 2)
        y = (self.top.winfo_screenheight() // 2) - (650 // 2)
        self.top.geometry(f"520x650+{x}+{y}")
        
        # Configurar tamaño mínimo
        self.top.minsize(480, 550)
        
        self._create_interface()
        
        # Esperar a que el usuario cierre el diálogo
        self.top.wait_window()
    
    def _create_interface(self):
        # Hacer scrollable todo el diálogo
        canvas = tk.Canvas(self.top)
        scrollbar = ttk.Scrollbar(self.top, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        main_frame = ttk.Frame(scrollable_frame, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="🤖 Generación de Metadatos con IA", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(anchor='w', pady=(0, 10))
        
        # Información del archivo
        info_frame = ttk.LabelFrame(main_frame, text="📁 Archivo seleccionado", padding=8)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        filename = os.path.basename(self.file_path)
        ttk.Label(info_frame, text=filename, font=('Arial', 9, 'bold'), 
                 wraplength=400).pack(anchor='w')
        
        # Descripción compacta
        desc_frame = ttk.LabelFrame(main_frame, text="ℹ️ Información", padding=8)
        desc_frame.pack(fill=tk.X, pady=(0, 10))
        
        desc_text = """Genera metadatos automáticamente usando IA para:
• Fan-edits sin metadatos online • Cómics independientes
• Proyectos personales • One-shots sin información"""
        
        desc_label = ttk.Label(desc_frame, text=desc_text, font=('Arial', 8), wraplength=400)
        desc_label.pack(anchor='w')
        
        # Opciones de configuración
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuración", padding=8)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tipo de cómic
        ttk.Label(config_frame, text="Tipo de cómic:", font=('Arial', 8, 'bold')).pack(anchor='w')
        self.comic_type = tk.StringVar(value="auto")
        
        type_frame = ttk.Frame(config_frame)
        type_frame.pack(fill=tk.X, pady=(3, 8))
        
        ttk.Radiobutton(type_frame, text="Auto", 
                       variable=self.comic_type, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(type_frame, text="Fan-edit", 
                       variable=self.comic_type, value="fanedit").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(type_frame, text="Independiente", 
                       variable=self.comic_type, value="indie").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(type_frame, text="Personal", 
                       variable=self.comic_type, value="personal").pack(side=tk.LEFT, padx=(10, 0))
        
        # Género sugerido
        ttk.Label(config_frame, text="Género (opcional):", font=('Arial', 8, 'bold')).pack(anchor='w', pady=(5, 2))
        self.genre_entry = ttk.Entry(config_frame, font=('Arial', 9))
        self.genre_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Idioma
        ttk.Label(config_frame, text="Idioma:", font=('Arial', 8, 'bold')).pack(anchor='w', pady=(5, 2))
        self.language = tk.StringVar(value="auto")
        
        lang_frame = ttk.Frame(config_frame)
        lang_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Radiobutton(lang_frame, text="Auto", variable=self.language, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(lang_frame, text="Español", variable=self.language, value="es").pack(side=tk.LEFT, padx=(15, 0))
        ttk.Radiobutton(lang_frame, text="English", variable=self.language, value="en").pack(side=tk.LEFT, padx=(15, 0))
        
        # Advertencia
        warning_frame = ttk.Frame(main_frame)
        warning_frame.pack(fill=tk.X, pady=(5, 15))
        
        warning_text = "⚠️ Los metadatos generados pueden requerir revisión manual."
        ttk.Label(warning_frame, text=warning_text, font=('Arial', 8), 
                 foreground='orange', wraplength=400).pack(anchor='w')
        
        # Botones fijos en la parte inferior
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame fijo para botones en la ventana principal
        button_frame = ttk.Frame(self.top, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(button_frame, text="🤖 Generar Metadatos", 
                  command=self._generate, bootstyle=SUCCESS).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(button_frame, text="❌ Cancelar", 
                  command=self._cancel).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Eventos
        self.top.bind("<Return>", lambda e: self._generate())
        self.top.bind("<Escape>", lambda e: self._cancel())
        
        # Scroll con rueda del ratón
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
    
    def _generate(self):
        self.result = {
            'comic_type': self.comic_type.get(),
            'preferred_genre': self.genre_entry.get().strip(),
            'language': self.language.get()
        }
        self.top.destroy()
    
    def _cancel(self):
        self.result = None
        self.top.destroy()


class IssueSelectionDialog:
    """Diálogo para seleccionar un issue específico de una lista"""
    
    def __init__(self, parent, issue_options):
        self.parent = parent
        self.selected_issue = None
        
        self.top = tk.Toplevel(parent)
        self.top.title("Seleccionar Issue")
        self.top.geometry("600x400")
        self.top.resizable(True, True)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Centrar la ventana
        self.top.update_idletasks()
        x = (self.top.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.top.winfo_screenheight() // 2) - (400 // 2)
        self.top.geometry(f"600x400+{x}+{y}")
        
        # Crear interfaz
        main_frame = ttk.Frame(self.top, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.top.grid_rowconfigure(0, weight=1)
        self.top.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Etiqueta
        label = ttk.Label(main_frame, text="Se encontraron múltiples issues. Selecciona uno:")
        label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        # Lista de issues
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        self.listbox = tk.Listbox(list_frame, font=("Segoe UI", 9),
                                 bg="white", fg="black",
                                 selectbackground="#0078d4", selectforeground="white")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Poblar lista
        self.issue_data = []
        for display_text, issue_data in issue_options:
            self.listbox.insert(tk.END, display_text)
            self.issue_data.append(issue_data)
        
        # Seleccionar primer elemento por defecto
        if self.issue_data:
            self.listbox.selection_set(0)
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Seleccionar", command=self._select_issue).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Cancelar", command=self._cancel).pack(side="left")
        
        # Eventos
        self.listbox.bind("<Double-Button-1>", lambda e: self._select_issue())
        self.top.bind("<Return>", lambda e: self._select_issue())
        self.top.bind("<Escape>", lambda e: self._cancel())
        
        # Enfocar
        self.listbox.focus_set()
    
    def _select_issue(self):
        selection = self.listbox.curselection()
        if selection:
            self.selected_issue = self.issue_data[selection[0]]
        self.top.destroy()
    
    def _cancel(self):
        self.selected_issue = None
        self.top.destroy()
import zipfile
import tempfile
import threading
import xml.etree.ElementTree as ET
from xml.dom import minidom
import requests
import html
from io import BytesIO
import configparser
import json
import time
from pathlib import Path
from collections import Counter
import webbrowser
import sqlite3
import shutil
from PIL import Image, ImageDraw, ImageTk
import traceback
import socket
from flask import Flask, jsonify, send_file, abort
from werkzeug.serving import make_server

# Importar módulo de tema moderno
try:
    from modern_theme import *
    MODERN_THEME_AVAILABLE = True
    print("✨ Módulo de tema moderno cargado")
except ImportError:
    MODERN_THEME_AVAILABLE = False
    print("⚠️ Módulo de tema moderno no disponible")

# --- Importación para la interfaz moderna ---
try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    MODERN_UI = True
    print("✨ Interfaz moderna activada (ttkbootstrap)")
except ImportError:
    print("ADVERTENCIA: ttkbootstrap no encontrado. Se usará la interfaz clásica.")
    print("Para instalarlo, ejecuta: pip install ttkbootstrap")
    import tkinter.ttk as ttkb
    MODERN_UI = False
    PRIMARY, SUCCESS, INFO, WARNING, DANGER, DARK, LIGHT, SECONDARY, DEFAULT, CENTER, INVERSE = "primary", "success", "info", "warning", "danger", "dark", "light", "secondary", "default", "center", "inverse"

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("ADVERTENCIA: OpenCV no encontrado. La función para eliminar números de página estará desactivada.")

try:
    from antmar.providers import whakoom_scraper
    WHAKOOM_AVAILABLE = True
except ImportError:
    WHAKOOM_AVAILABLE = False
    print("ADVERTENCIA: No se encontró el scraper de Whakoom. La pestaña de Whakoom estará desactivada.")


# ==============================================================================
# 2. CONFIGURACIÓN Y CONSTANTES
# ==============================================================================
COMICVINE_API_KEY = ""
DEEPL_API_KEY = ""
translator = None # Se inicializará bajo demanda

HEADERS = {'User-Agent': 'GestorDeComics/1.0'}

# Variables globales para el servidor Flask
http_server = None
server_thread = None
flask_app = None # Se inicializará bajo demanda
REQUEST_TIMEOUT = 20
DB_FILE = "comics.db"

# ==============================================================================
# 3. FUNCIONES AUXILIARES GLOBALES
# ==============================================================================
from antmar.utils import natural_sort_key

def parse_query(query):
    match = re.search(r'^(.*?)(?:#\s*|\s+)(\d+(\.\d+)?)$', query.strip())
    if match: return match.group(1).strip(), match.group(2).strip()
    return query.strip(), None








def remove_page_number(pil_image):
    if not OPENCV_AVAILABLE: return pil_image
    try:
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR); height, width, _ = open_cv_image.shape
        roi_height = int(height * 0.08); combined_mask = np.zeros((height, width), dtype=np.uint8)
        scan_areas = [(0, int(width * 0.15)), (int(width * 0.425), int(width * 0.575)), (int(width * 0.85), width)]
        for start_x, end_x in scan_areas:
            roi = open_cv_image[height - roi_height:height, start_x:end_x]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY); _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if 50 < cv2.contourArea(cnt) < 2000:
                    cv2.drawContours(combined_mask, [cnt], -1, (255), -1, offset=(start_x, height - roi_height))
        kernel = np.ones((5,5), np.uint8); dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        if np.any(dilated_mask):
            result_image = cv2.inpaint(open_cv_image, dilated_mask, 5, cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        return pil_image
    except Exception as e: print(f"Error al procesar la eliminación de números de página: {e}"); return pil_image

def get_cover_from_cbz(cbz_path, size):
    """Extrae la portada de un CBZ y retorna PIL Image con redimensionamiento correcto"""
    try:
        if not os.path.exists(cbz_path):
            print(f"❌ Archivo no existe: {cbz_path}")
            return None
        
        with zipfile.ZipFile(cbz_path, 'r') as zf:
            # Buscar archivos de imagen, priorizando formatos comunes
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
            image_list = []
            
            for file_name in zf.namelist():
                if file_name.lower().endswith(image_extensions):
                    # Filtrar archivos no deseados (thumbnails, etc.)
                    if not any(skip in file_name.lower() for skip in ['thumb', 'preview', '__macosx', '.ds_store']):
                        image_list.append(file_name)
            
            if not image_list: 
                print(f"⚠️ No hay imágenes en: {os.path.basename(cbz_path)}")
                return None
                
            # Ordenar para obtener la primera página
            image_list.sort(key=natural_sort_key)
            
            with zf.open(image_list[0]) as image_file:
                img_data = image_file.read()
                
                # Verificar que tenemos datos válidos
                if len(img_data) < 100:  # Muy pequeño para ser una imagen válida
                    print(f"❌ Archivo de imagen demasiado pequeño en {os.path.basename(cbz_path)}")
                    return None
                
                # Abrir imagen con manejo de errores mejorado
                img = Image.open(BytesIO(img_data))
                
                # Convertir a RGB si es necesario
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Crear fondo blanco y pegar la imagen encima
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Crear copia para persistencia
                img = img.copy()
                
                # Validar dimensiones
                if img.size[0] < 10 or img.size[1] < 10:
                    print(f"❌ Imagen demasiado pequeña en {os.path.basename(cbz_path)}")
                    return None
                
                # Obtener dimensiones
                img_width, img_height = img.size
                target_width, target_height = size
                
                # Validar tamaño objetivo
                if target_width <= 0 or target_height <= 0:
                    print(f"❌ Tamaño objetivo inválido: {size}")
                    return None
                
                # Calcular ratio preservando aspecto
                ratio_w = target_width / img_width
                ratio_h = target_height / img_height
                
                # Usar el ratio más pequeño para que quepa todo
                ratio = min(ratio_w, ratio_h)
                
                # Calcular nuevas dimensiones
                new_width = max(1, int(img_width * ratio))
                new_height = max(1, int(img_height * ratio))
                
                # Redimensionar con calidad alta
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                return img
                
    except zipfile.BadZipFile:
        print(f"❌ Archivo CBZ corrupto: {os.path.basename(cbz_path)}")
    except OSError as e:
        print(f"❌ Error de sistema leyendo {os.path.basename(cbz_path)}: {e}")
    except Exception as e:
        print(f"❌ Error inesperado cargando portada de {os.path.basename(cbz_path)}: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def create_circular_photo(image_path, size):
    """Crea una imagen circular a partir de una ruta, con un tamaño fijo."""
    try:
        with Image.open(image_path) as im:
            # 1. Asegurarse de que la imagen sea cuadrada, recortando desde el centro
            w, h = im.size
            short_side = min(w, h)
            left = (w - short_side) / 2
            top = (h - short_side) / 2
            right = (w + short_side) / 2
            bottom = (h + short_side) / 2
            im = im.crop((left, top, right, bottom))
            
            # 2. Redimensionar a la calidad más alta
            im = im.resize((size, size), Image.Resampling.LANCZOS)

            # 3. Crear la máscara circular
            mask = Image.new('L', (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size, size), fill=255)
            
            # 4. Aplicar la máscara
            im.putalpha(mask)
            
            # 5. Crear un fondo (opcional, por si la imagen se muestra sobre algo que no es negro)
            # output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            # output.paste(im, (0, 0), im)

            return ImageTk.PhotoImage(im)
    except Exception as e:
        print(f"Error al crear imagen circular: {e}")
        return None
    
PUBLISHER_LOGO_PATH = Path("publisher_logos")
LOGO_CACHE = {}

def load_publisher_logo(publisher_name, height):
    """Carga, redimensiona y cachea un logo de editorial."""
    if not publisher_name:
        return None
    
    # Comprobar si el logo ya está en caché para esta altura
    cache_key = (publisher_name, height)
    if cache_key in LOGO_CACHE:
        return LOGO_CACHE[cache_key]

    # Normalizar el nombre para buscar el archivo
    logo_filename = publisher_name.lower().replace(" ", "_").replace("comics", "").strip("_") + ".png"
    logo_path = PUBLISHER_LOGO_PATH / logo_filename
    
    # Búsqueda de variantes (ej. 'marvel comics' -> 'marvel.png')
    if not logo_path.exists():
        simple_name = publisher_name.split(' ')[0].lower() + ".png"
        simple_path = PUBLISHER_LOGO_PATH / simple_name
        if simple_path.exists():
            logo_path = simple_path
        else:
            LOGO_CACHE[cache_key] = None # Marcar como no encontrado para no volver a buscar
            return None
            
    try:
        with Image.open(logo_path) as img:
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            new_width = int(height * aspect_ratio)
            
            # Usar LANCZOS para el mejor redimensionado
            img_resized = img.resize((new_width, height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img_resized)
            LOGO_CACHE[cache_key] = photo # Guardar en caché
            return photo
    except Exception as e:
        print(f"Error cargando el logo para {publisher_name}: {e}")
        LOGO_CACHE[cache_key] = None
        return None
    
    # EN LA SECCIÓN 3. FUNCIONES AUXILIARES GLOBALES
# def generate_summary_with_ai - FUNCIÓN DESACTIVADA TEMPORALMENTE(cbz_path, status_var_ref):
    """Extrae texto de las primeras páginas y genera un resumen con IA."""
    if not TESSERACT_AVAILABLE or not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        messagebox.showerror("Función no disponible", "Tesseract, la librería de OpenAI o la clave API no están configurados correctamente.")
        return None

    extracted_text = ""
    try:
        status_var_ref.set("IA: Extrayendo texto de las primeras 5 páginas...")
        with zipfile.ZipFile(cbz_path, 'r') as zf:
            image_list = sorted([f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))], key=natural_sort_key)
            # Analizar hasta 5 páginas o el total si es menor
            for i in range(min(5, len(image_list))):
                with zf.open(image_list[i]) as image_file:
                    with Image.open(BytesIO(image_file.read())) as img:
                        # OCR en español e inglés
                        extracted_text += pytesseract.image_to_string(img, lang='spa+eng') + "\n\n"
    except Exception as e:
        messagebox.showerror("Error de OCR", f"No se pudo extraer el texto del cómic.\nError: {e}")
        return None
        
    if not extracted_text.strip():
        messagebox.showwarning("Sin texto", "El OCR no pudo detectar texto legible en las primeras páginas.")
        return None

    try:
        status_var_ref.set("IA: Contactando con OpenAI para generar resumen...")
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        prompt_text = (
            "Eres un experto catalogador de cómics. A partir del siguiente texto extraído mediante OCR de las primeras páginas de un cómic, "
            "escribe un resumen conciso y atractivo en español, de 2 a 4 frases. Ignora por completo los créditos, fechas, precios o texto sin sentido. "
            "Céntrate solo en la trama y los personajes. Si no puedes discernir una trama, indica que no hay suficiente información.\n\n"
            f"TEXTO EXTRAÍDO:\n---\n{extracted_text[:4000]}\n---\nRESUMEN:"
        )

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # Modelo rápido y económico
            messages=[
                {"role": "system", "content": "Eres un asistente experto en cómics que escribe resúmenes en español."},
                {"role": "user", "content": prompt_text}
            ]
        )
        summary = completion.choices[0].message.content.strip()
        status_var_ref.set("IA: ¡Resumen generado con éxito!")
        return summary
    except Exception as e:
        messagebox.showerror("Error de OpenAI", f"No se pudo generar el resumen.\nVerifica tu clave API y conexión.\nError: {e}")
        return None


# ==============================================================================
# 4. CLASES DE DIÁLOGOS Y VENTANAS
# ==============================================================================
# (Ordenadas para evitar NameErrors)

# Pega esta clase junto a las otras, por ejemplo, después de BatchTranslatorWindow


        
class VolumeSelectionDialog(tk.Toplevel):
    def __init__(self, parent, volumes):
        super().__init__(parent); self.title("Seleccionar Volumen"); self.geometry("700x500"); self.transient(parent); self.grab_set()
        self.selected_volume = None; self.all_volumes = volumes; self.currently_displayed_volumes = volumes
        ttk.Label(self, text="Se encontraron varios volúmenes. Filtra y selecciona el correcto:", justify=tk.LEFT, padx=10, pady=(10,0)).pack(anchor="w")
        filter_frame = ttk.Frame(self, padx=10, pady=5); filter_frame.pack(fill=tk.X)
        ttk.Label(filter_frame, text="Filtro:").pack(side=tk.LEFT)
        self.filter_entry = ttk.Entry(filter_frame); self.filter_entry.pack(fill=tk.X, expand=True, padx=5); self.filter_entry.bind("<KeyRelease>", self.filter_list)
        list_frame = ttk.Frame(self); list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.listbox = tk.Listbox(list_frame, font=("Arial", 10), selectmode=tk.SINGLE, 
                                 bg="white", fg="black", 
                                 selectbackground="#0078d4", selectforeground="white"); self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.listbox.config(yscrollcommand=scrollbar.set)
        self.populate_listbox(self.all_volumes); self.listbox.bind("<Double-Button-1>", self.on_select)
        button_frame = ttk.Frame(self); button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        select_btn = ttk.Button(button_frame, text="Seleccionar", command=self.on_select)
        if MODERN_UI:
            select_btn.config(bootstyle=SUCCESS)
        select_btn.pack(side=tk.RIGHT)
        self.filter_entry.focus_set(); self.wait_window()
    def populate_listbox(self, volumes_to_show):
        self.listbox.delete(0, tk.END); self.currently_displayed_volumes = volumes_to_show
        for vol in self.currently_displayed_volumes:
            publisher_name = vol.get('publisher', {}).get('name', 'N/A') if vol.get('publisher') else 'N/A'
            self.listbox.insert(tk.END, f"{vol.get('name', 'N/A')} ({vol.get('start_year', '????')}) - {publisher_name}")
    def filter_list(self, event=None):
        filter_text = self.filter_entry.get().lower()
        if not filter_text: filtered_volumes = self.all_volumes
        else:
            filtered_volumes = [vol for vol in self.all_volumes if filter_text in f"{vol.get('name','')} {vol.get('start_year','')} {vol.get('publisher', {}).get('name', '')}".lower()]
        self.populate_listbox(filtered_volumes)
    def on_select(self, event=None):
        selection = self.listbox.curselection()
        if not selection: 
            return
        self.selected_volume = self.currently_displayed_volumes[selection[0]]
        self.destroy()

# Reemplaza la clase MetadataEditorWindow completa con esta versión

class MetadataEditorWindow(tk.Toplevel):
    def __init__(self, parent, cbz_path, status_var_ref, app_instance=None, file_list_context=None, current_index_context=None):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.app = app_instance
        self.status_var_ref = status_var_ref
        self.file_list = file_list_context
        self.current_index = current_index_context if current_index_context is not None else 0
        self.cbz_path = cbz_path 
        self.metadata_fields = {}
        self.cv_issue_number = None
        self.title("Editor de Metadatos") 
        self.geometry("1200x980")
        
        paned_window_config = {'orient': tk.HORIZONTAL}
        if MODERN_UI:
            paned_window_config['bootstyle'] = LIGHT
        paned_window = ttk.PanedWindow(self, **paned_window_config)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        left_frame = ttk.Frame(paned_window, width=400)
        paned_window.add(left_frame, weight=1)
        
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)
        
        self.notebook = ttk.Notebook(left_frame)
        
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        cv_tab = ttk.Frame(self.notebook)
        self.notebook.add(cv_tab, text="Comic Vine (Original)")
        self.setup_cv_tab(cv_tab)
        
        # Pestaña de Whakoom (solo si está disponible)
        if WHAKOOM_AVAILABLE:
            whakoom_tab = ttk.Frame(self.notebook)
            self.notebook.add(whakoom_tab, text="Whakoom (Ed. Española)")
            self.setup_whakoom_tab(whakoom_tab)
        else:
            print("ℹ️ Pestaña de Whakoom deshabilitada - scraper no disponible")
        
        self.cover_label = tk.Label(right_frame, bg="black", text="Portada", fg="white", anchor="center")
        self.cover_label.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Frame para los campos de metadatos con scroll - FONDO BLANCO
        canvas = tk.Canvas(right_frame, bg="white", highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scrollable_frame = tk.Frame(canvas, bg="white")
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.config(yscrollcommand=scrollbar.set)
        
        # Asegurar que el frame interno se expanda con el canvas
        def configure_canvas_width(event):
            canvas.itemconfig(self.canvas_window, width=event.width)
        canvas.bind("<Configure>", configure_canvas_width)
        
        self.create_metadata_widgets()
        
        # Botones inferiores
        ttk.Button(bottom_frame, text="Cerrar", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        self.next_btn = ttk.Button(bottom_frame, text="Siguiente >>", command=self.go_next)
        self.next_btn.pack(side=tk.RIGHT, padx=(5,10))
        self.prev_btn = ttk.Button(bottom_frame, text="<< Anterior", command=self.go_previous)
        self.prev_btn.pack(side=tk.RIGHT, padx=5)
        
        # ¡IMPORTANTE! El botón 'Guardar' ahora llama a la función corregida.
        save_btn = ttk.Button(bottom_frame, text="Guardar Cambios en CBZ", command=self.save_metadata_sync)
        if MODERN_UI:
            save_btn.config(bootstyle=SUCCESS)
        save_btn.pack(side=tk.RIGHT)
        
        self.add_to_lib_btn = ttk.Button(
            bottom_frame, 
            text="Añadir/Actualizar en Biblioteca", 
            command=self.add_to_library
        )
        if MODERN_UI:
            self.add_to_lib_btn.config(bootstyle=PRIMARY)
        self.add_to_lib_btn.pack(side=tk.LEFT, padx=10)
        
        # Cargar datos DESPUÉS de que la ventana esté lista
        print(f"🔧 DEBUG: Ventana creada, programando carga...")
        print(f"🔧 DEBUG: file_list = {self.file_list}, current_index = {self.current_index}, cbz_path = {self.cbz_path}")
        self.after(100, self.load_initial_data)

    def _load_comic(self, index):
        print(f"🔧 DEBUG: _load_comic llamado con index={index}")
        if not self.file_list or not (0 <= index < len(self.file_list)):
            print(f"🔧 DEBUG: No hay file_list válido, cargando CBZ individual")
            self.load_initial_data()
            return
        print(f"🔧 DEBUG: Cargando cómic {index} de la lista")
        self.current_index = index
        self.cbz_path = self.file_list[self.current_index]
        self.cv_issue_number = None
        if hasattr(self, 'whakoom_url_entry'): self.whakoom_url_entry.delete(0, tk.END)
        self.title(f"Editor de Metadatos ({self.current_index + 1}/{len(self.file_list)}) - {os.path.basename(self.cbz_path)}")
        self.load_initial_data()
        self.update_navigation_buttons()

    def go_next(self):
        if self.file_list and self.current_index < len(self.file_list) - 1:
            self._load_comic(self.current_index + 1)

    def go_previous(self):
        if self.file_list and self.current_index > 0:
            self._load_comic(self.current_index - 1)

    def update_navigation_buttons(self):
        if not self.file_list or len(self.file_list) <= 1:
            self.prev_btn.pack_forget()
            self.next_btn.pack_forget()
            return
        self.prev_btn.pack(side=tk.RIGHT, padx=5)
        self.next_btn.pack(side=tk.RIGHT, padx=(5,10))
        self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_index < len(self.file_list) - 1 else tk.DISABLED)

    def load_initial_data(self):
        print(f"🔍 DEBUG: Cargando datos de {self.cbz_path}")
        self.status_var_ref.set(f"Cargando {os.path.basename(self.cbz_path)}...")

        print(f"🔍 DEBUG: Cargando portada...")
        pil_img = get_cover_from_cbz(self.cbz_path, (400, 600))
        if pil_img:
            # Convertir PIL Image a PhotoImage para Tkinter
            self.photo = ImageTk.PhotoImage(pil_img)
            print(f"✅ Portada cargada")
            self.cover_label.config(image=self.photo, text="")
        else:
            self.photo = None
            print(f"❌ No se pudo cargar la portada")
            self.cover_label.config(image=None, text="Sin portada", bg="black", fg="white")
        
        print(f"🔍 DEBUG: Leyendo metadatos del CBZ...")
        existing_metadata = read_comicinfo_from_cbz(self.cbz_path)
        print(f"🔍 DEBUG: Metadatos encontrados: {len(existing_metadata) if existing_metadata else 0} campos")
        
        # Limpiar campos
        for field, widget in self.metadata_fields.items():
            if isinstance(widget, scrolledtext.ScrolledText): 
                widget.delete(1.0, tk.END)
            else: 
                widget.delete(0, tk.END)
        
        if existing_metadata: 
            print(f"✅ Poblando campos con metadatos...")
            self.populate_fields(existing_metadata, "load")
        else:
            print(f"⚠️ No hay metadatos en el CBZ")
        
        self.cv_search_entry.delete(0, tk.END)
        self.cv_search_entry.insert(0, os.path.splitext(os.path.basename(self.cbz_path))[0])
        self.status_var_ref.set("Listo.")
        print(f"✅ Carga completada")

    # --- INICIO DE LA CORRECCIÓN DE save_metadata y add_to_library ---

    def _save_task(self, path, xml_string, show_success_message):
        """Tarea de guardado que se puede ejecutar en un hilo."""
        success = inject_xml_into_cbz(path, xml_string)
        final_status = f"Metadatos guardados en {os.path.basename(path)}." if success else f"Error al guardar en {os.path.basename(path)}."
        
        # Actualizamos la UI de forma segura desde el hilo
        self.after(0, self.status_var_ref.set, final_status)
        
        if show_success_message and success:
            self.after(0, lambda: messagebox.showinfo("Éxito", "Metadatos guardados en el archivo CBZ.", parent=self))
        elif not success:
            self.after(0, lambda: messagebox.showerror("Error", "No se pudieron guardar los metadatos en el CBZ.", parent=self))
        return success

    def save_metadata_sync(self):
        """Función para el botón 'Guardar'. Síncrona, espera a que termine."""
        self.status_var_ref.set("Guardando metadatos en el archivo CBZ...")
        metadata = {field: self.get_field_value(field) for field in self.metadata_fields if self.get_field_value(field)}
        xml_string = generate_comicinfo_xml(metadata)
        self._save_task(self.cbz_path, xml_string, show_success_message=True)

    def add_to_library(self):
        """Función para 'Añadir/Actualizar'. Guarda el CBZ en segundo plano."""
        # 1. Preparamos los datos
        metadata = {field: self.get_field_value(field) for field in self.metadata_fields if self.get_field_value(field)}
        xml_string = generate_comicinfo_xml(metadata)
        
        # 2. Lanzamos el guardado del CBZ en un hilo secundario (asíncrono)
        self.status_var_ref.set(f"Guardando {os.path.basename(self.cbz_path)} en segundo plano...")
        threading.Thread(target=self._save_task, args=(self.cbz_path, xml_string, False), daemon=True).start()
        
        # 3. Actualizamos la base de datos inmediatamente (es rápido)
        if self.app:
            self.app.add_or_update_comic_in_db(self.cbz_path, metadata=metadata)

    # --- FIN DE LA CORRECCIÓN ---

    def create_metadata_widgets(self):
        self.predefined_groups = ["", "MARVEL", "DC COMICS", "MARVEL EN ORDEN", "DC COMICS EN ORDEN", "INDEPENDIENTES", "CONAN", "EUROPEO", "CLÁSICO", "MANGA"]
        
        # Usar tk.Label con colores explícitos en lugar de ttk.Label
        tk.Label(self.scrollable_frame, text="Grupo Serie (Tags):", bg="white", fg="black", anchor='w').grid(row=0, column=0, sticky="w", padx=5, pady=2)
        combo = ttk.Combobox(self.scrollable_frame, values=self.predefined_groups)
        combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.metadata_fields["Tags"] = combo
        
        fields = ["Series", "Number", "Title", "Publisher", "Year", "Month", "Day", "Writer", "Penciller", "Inker", "Colorist", "Letterer", "CoverArtist", "Editor", "StoryArc", "Characters", "Teams", "Web", "Notes", "ScanInformation"]
        row = 1
        for field in fields:
            tk.Label(self.scrollable_frame, text=f"{field}:", bg="white", fg="black", anchor='w').grid(row=row, column=0, sticky="w", padx=5, pady=2)
            entry = tk.Entry(self.scrollable_frame, width=80, bg="white", fg="black", insertbackground="black")
            entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            self.metadata_fields[field] = entry
            row += 1
        
        tk.Label(self.scrollable_frame, text="Summary:", bg="white", fg="black", anchor='w').grid(row=row, column=0, sticky="nw", padx=5, pady=2)
        summary_text = scrolledtext.ScrolledText(self.scrollable_frame, wrap=tk.WORD, height=10, width=80, bg="white", fg="black", insertbackground="black")
        summary_text.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.metadata_fields["Summary"] = summary_text
        self.scrollable_frame.columnconfigure(1, weight=1)

    def apply_whakoom_data(self):
        """Aplica metadatos desde una URL de Whakoom"""
        url = self.whakoom_url_entry.get().strip()
        
        if not url:
            messagebox.showwarning("URL Vacía", "Por favor, pega una URL de Whakoom.", parent=self)
            return
            
        if "whakoom.com" not in url.lower():
            messagebox.showwarning("URL Inválida", "Por favor, pega una URL válida de Whakoom.\n\nEjemplo: https://www.whakoom.com/comics/...", parent=self)
            return
            
        if not WHAKOOM_AVAILABLE:
            messagebox.showerror("Scraper no disponible", "El scraper de Whakoom no está disponible.", parent=self)
            return
            
        self.status_var_ref.set("Obteniendo metadatos de Whakoom...")
        threading.Thread(target=self._get_whakoom_details_thread, args=(url,), daemon=True).start()
    def download_cover(self, url):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT); img_data = BytesIO(response.content)
            with Image.open(img_data) as img:
                img.thumbnail((400, 600), Image.Resampling.LANCZOS); photo = ImageTk.PhotoImage(img)
                self.cover_label.config(image=photo, text=""); self.cover_label.image = photo
        except Exception as e: print(f"Error descargando portada: {e}")
    def find_issue_in_volume(self, volume):
        volume_id = volume.get('id');
        if not volume_id: 
            self.status_var_ref.set("Error: No se seleccionó un volumen válido."); 
            return
        threading.Thread(target=self._find_issue_in_volume_thread, args=(volume_id, self.cv_issue_number), daemon=True).start()
    def get_field_value(self, field):
        widget = self.metadata_fields.get(field)
        if not widget: return ""
        if isinstance(widget, scrolledtext.ScrolledText): return widget.get(1.0, tk.END).strip()
        elif isinstance(widget, ttk.Combobox): return widget.get().strip()
        else: return widget.get().strip()
    def handle_found_volumes(self, volumes):
        if not volumes: self.status_var_ref.set("No se encontró ningún volumen."); messagebox.showinfo("Sin resultados", "No se encontró ningún volumen.", parent=self); return
        selected_volume = volumes[0] if len(volumes) == 1 else VolumeSelectionDialog(self, volumes).selected_volume
        if selected_volume: self.find_issue_in_volume(selected_volume)
        else: self.status_var_ref.set("Búsqueda cancelada.")
    def open_system_browser(self):
        self.status_var_ref.set("Abriendo Whakoom en tu navegador..."); webbrowser.open_new_tab('https://www.whakoom.com')
    def populate_fields(self, metadata, source):
        if not metadata: 
            self.status_var_ref.set("No se pudieron obtener detalles.")
            return
        
        # Si estamos en modo organizador, aplicar al organizador
        if hasattr(self, 'organizer_apply_mode') and self.organizer_apply_mode:
            self.organizer_apply_mode = False  # Resetear flag
            self._organizer_apply_cv_metadata(metadata)
            return
        
        if source == "whakoom":
            fields_to_populate = ['Title', 'Tags', 'Publisher', 'Year', 'Month', 'Day', 'Series', 'Number', 'Web', 'Notes', 'Summary', 'Writer', 'Penciller', 'Inker', 'Colorist']
            for field in fields_to_populate:
                if metadata.get(field): 
                    self.set_field_value(field, metadata.get(field))
            self.status_var_ref.set("Datos de Whakoom aplicados.")
            return
        elif source == "cv":
            for field, value in metadata.items():
                if field in self.metadata_fields: 
                    self.set_field_value(field, value)
            self.status_var_ref.set("Datos de Comic Vine aplicados.")
        elif source == "load":
            for field, value in metadata.items():
                if field in self.metadata_fields: 
                    self.set_field_value(field, value)
            self.status_var_ref.set("Metadatos existentes cargados.")
    
    def _organizer_apply_cv_metadata(self, metadata):
        """Aplicar metadatos de ComicVine al organizador"""
        try:
            # Limpiar campos primero
            self._organizer_clear_fields()
            
            # Aplicar datos
            for key, value in metadata.items():
                if key in self.organizer_field_widgets and value:
                    widget = self.organizer_field_widgets[key]
                    if isinstance(widget, scrolledtext.ScrolledText):
                        widget.insert(1.0, str(value))
                    else:
                        widget.insert(0, str(value))
            
            # Generar nombre automático
            self._organizer_generate_filename()
            
            self.status_var.set("✅ Datos de ComicVine aplicados")
        except Exception as e:
            print(f"Error aplicando metadatos CV: {e}")
            self.status_var.set("Error aplicando datos")

    def set_field_value(self, field, value):
        widget = self.metadata_fields.get(field)
        if not widget: 
            return
        clean_value = str(value)
        if isinstance(widget, scrolledtext.ScrolledText):
            widget.delete(1.0, tk.END)
            widget.insert(tk.END, clean_value)
        elif isinstance(widget, ttk.Combobox):
            widget.set(clean_value)
        else: 
            widget.delete(0, tk.END)
            widget.insert(0, clean_value)
    def setup_cv_tab(self, parent_tab):
        search_frame = ttk.LabelFrame(parent_tab, text="Búsqueda Precisa"); search_frame.pack(fill=tk.X, padx=5, pady=5)
        self.cv_search_entry = ttk.Entry(search_frame); self.cv_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.cv_search_entry.bind("<Return>", lambda e: self.start_precise_search())
        ttk.Button(search_frame, text="Buscar", command=self.start_precise_search).pack(side=tk.RIGHT, padx=5, pady=5)
        info_btn = ttk.Button(parent_tab, text="↓ Datos de CV se aplican solos al encontrar ↓", state="disabled")
        if MODERN_UI:
            info_btn.config(bootstyle=DARK)
        info_btn.pack(fill=tk.X, padx=5, pady=5)
    def setup_whakoom_tab(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding=5); main_frame.pack(fill=tk.BOTH, expand=True)
        browser_btn = ttk.Button(main_frame, text="Abrir Whakoom en mi Navegador", command=self.open_system_browser)
        if MODERN_UI:
            browser_btn.config(bootstyle=INFO)
        browser_btn.pack(fill=tk.X, pady=(0,10))
        url_frame = ttk.LabelFrame(main_frame, text="Pega aquí la URL del cómic de Whakoom"); url_frame.pack(fill=tk.X, pady=5)
        self.whakoom_url_entry = ttk.Entry(url_frame, font=("Arial", 10)); self.whakoom_url_entry.pack(fill=tk.X, padx=5, pady=5, ipady=3)
        apply_btn = ttk.Button(main_frame, text="↓ Aplicar Datos de la URL pegada ↓", command=self.apply_whakoom_data)
        if MODERN_UI:
            apply_btn.config(bootstyle=SUCCESS)
        apply_btn.pack(fill=tk.X, pady=10)
    def start_precise_search(self):
        query = self.cv_search_entry.get().strip()
        if not query: 
            return
        
        series_name, issue_number = parse_query(query)
        self.cv_issue_number = issue_number
        
        # Ya no es obligatorio tener número - puede ser un one-shot, TPB, etc.
        if self.cv_issue_number:
            self.status_var_ref.set(f"Buscando volúmenes para '{series_name}' #{self.cv_issue_number}...")
        else:
            self.status_var_ref.set(f"Buscando volúmenes para '{series_name}' (número único o sin número)...")
            
        threading.Thread(target=self._search_for_volumes_thread, args=(series_name,), daemon=True).start()
    def _get_whakoom_details_thread(self, details_url):
        try:
            if not WHAKOOM_AVAILABLE:
                raise ImportError("Scraper de Whakoom no disponible")
            
            metadata = whakoom_scraper.get_whakoom_details(details_url)
            self.after(0, self.populate_fields, metadata, "whakoom")
            self.after(0, self.status_var_ref.set, "Metadatos de Whakoom obtenidos correctamente.")
        except ImportError as e:
            self.after(0, self.status_var_ref.set, "Error: Scraper de Whakoom no disponible.")
            self.after(0, lambda: messagebox.showerror("Scraper no disponible", f"No se pudo cargar el scraper de Whakoom:\n{e}", parent=self))
        except ConnectionError as e:
            self.after(0, self.status_var_ref.set, "Error de conexión con Whakoom.")
            self.after(0, lambda: messagebox.showerror("Error de Conexión", f"No se pudo conectar a Whakoom:\n{e}", parent=self))
        except Exception as e:
            self.after(0, self.status_var_ref.set, "Error obteniendo detalles de Whakoom.")
            self.after(0, lambda: messagebox.showerror("Error de Whakoom", f"Error al obtener detalles:\n{e}", parent=self))
            import traceback; traceback.print_exc()
    def _find_issue_in_volume_thread(self, volume_id, issue_number):
        if not COMICVINE_API_KEY:
            self.after(0, lambda: messagebox.showerror("API Key Faltante", "La clave API de Comic Vine no está configurada.", parent=self)); return
        
        try:
            if issue_number:
                # Búsqueda específica por número
                params = {"api_key": COMICVINE_API_KEY, "format": "json", "filter": f"volume:{volume_id},issue_number:{issue_number}"}
                response = requests.get("https://comicvine.gamespot.com/api/issues/", params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                results = response.json().get('results', [])
                
                if not results: 
                    self.after(0, lambda: messagebox.showinfo("No encontrado", f"El número #{issue_number} no se encontró.", parent=self)); 
                    return
                    
                # Usar el primer resultado
                issue_guid = results[0].get('api_detail_url', '').strip('/').split('/')[-1]
                
            else:
                # Sin número específico - mostrar lista de issues del volumen para que el usuario elija
                params = {"api_key": COMICVINE_API_KEY, "format": "json", "filter": f"volume:{volume_id}", "sort": "issue_number:asc", "limit": 100}
                response = requests.get("https://comicvine.gamespot.com/api/issues/", params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                results = response.json().get('results', [])
                
                if not results:
                    self.after(0, lambda: messagebox.showinfo("Sin contenido", "Este volumen no tiene issues disponibles.", parent=self)); 
                    return
                
                # Si solo hay uno, usarlo directamente
                if len(results) == 1:
                    issue_guid = results[0].get('api_detail_url', '').strip('/').split('/')[-1]
                else:
                    # Mostrar selector de issues
                    self.after(0, lambda: self._show_issue_selector(results))
                    return
            
            # Obtener metadatos del issue
            metadata = get_comicvine_details(issue_guid, self.status_var_ref)
            self.after(0, self.populate_fields, metadata, "cv")
            if metadata.get('CoverURL'): 
                self.download_cover(metadata['CoverURL'])
                
        except requests.exceptions.RequestException as e: 
            self.after(0, lambda: messagebox.showerror("Error de Red", f"No se pudo buscar el ejemplar.\n{e}", parent=self))

    def _show_issue_selector(self, issues):
        """Muestra una ventana para seleccionar un issue específico cuando no hay número"""
        import tkinter.simpledialog as simpledialog
        
        # Crear lista de opciones
        issue_options = []
        for issue in issues:
            issue_num = issue.get('issue_number', 'Sin número')
            issue_name = issue.get('name', 'Sin título')
            cover_date = issue.get('cover_date', '')[:10] if issue.get('cover_date') else ''
            
            if issue_name and issue_name.strip():
                display_text = f"#{issue_num} - {issue_name}"
            else:
                display_text = f"#{issue_num}"
                
            if cover_date:
                display_text += f" ({cover_date})"
                
            issue_options.append((display_text, issue))
        
        # Mostrar diálogo de selección
        try:
            dialog = IssueSelectionDialog(self, issue_options)
            if dialog.selected_issue:
                issue_guid = dialog.selected_issue.get('api_detail_url', '').strip('/').split('/')[-1]
                metadata = get_comicvine_details(issue_guid, self.status_var_ref)
                self.populate_fields(metadata, "cv")
                if metadata.get('CoverURL'): 
                    self.download_cover(metadata['CoverURL'])
        except Exception as e:
            print(f"Error mostrando selector de issues: {e}")
            messagebox.showinfo("Múltiples resultados", 
                              f"Se encontraron {len(issues)} issues en este volumen.\n"
                              "Intenta ser más específico con el número.", parent=self)

    def _search_for_volumes_thread(self, series_name):
        if not COMICVINE_API_KEY:
            self.after(0, lambda: messagebox.showerror("API Key Faltante", "La clave API de Comic Vine no está configurada.", parent=self)); return
        params = {"api_key": COMICVINE_API_KEY, "format": "json", "resources": "volume", "query": series_name, "limit": 100}
        try:
            response = requests.get("https://comicvine.gamespot.com/api/search", params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT); response.raise_for_status()
            volumes = sorted([v for v in response.json().get('results', [])], key=lambda x: int(x.get('start_year', 9999)))
            self.after(0, self.handle_found_volumes, volumes)
        except requests.exceptions.RequestException as e: self.after(0, lambda: messagebox.showerror("Error de Red", f"No se pudo buscar en Comic Vine.\n{e}", parent=self))

class BatchRenamerWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent); self.title("Renombrador por Lote Visual"); self.geometry("800x800"); self.minsize(600, 600); self.transient(parent); self.grab_set()
        self.files = []; self.rename_plan = {}; self.current_index = -1; self.cover_photo = None
        main_frame = ttk.Frame(self, padding=10); main_frame.pack(fill=tk.BOTH, expand=True); main_frame.rowconfigure(1, weight=1); main_frame.columnconfigure(0, weight=1)
        ttk.Button(main_frame, text="Seleccionar Carpeta...", command=self.select_folder).grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.cover_label = ttk.Label(main_frame, background="black"); self.cover_label.grid(row=1, column=0, sticky="nsew")
        controls_frame = ttk.Frame(main_frame); controls_frame.grid(row=2, column=0, sticky="ew", pady=10); controls_frame.columnconfigure(1, weight=1)
        info_frame = ttk.Frame(controls_frame); info_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0,10)); info_frame.columnconfigure(1, weight=1)
        ttk.Label(info_frame, text="Nombre Actual:").grid(row=0, column=0, sticky="w")
        label_config = {"text": "", "anchor": "w", "wraplength": 500, "justify": "left"}
        if MODERN_UI:
            label_config["bootstyle"] = SECONDARY
        self.current_name_label = ttk.Label(info_frame, **label_config)
        self.current_name_label.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(info_frame, text="Nombre Nuevo:", font="-weight bold").grid(row=1, column=0, sticky="w", pady=(5,0)); self.new_name_entry = ttk.Entry(info_frame); self.new_name_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(5,0)); self.new_name_entry.bind("<Return>", lambda e: self.save_and_next())
        self.prev_btn = ttk.Button(controls_frame, text="<< Anterior", command=self.go_previous, state=tk.DISABLED); self.prev_btn.grid(row=1, column=0, sticky="ew", padx=2)
        self.save_next_btn = ttk.Button(controls_frame, text="Guardar y Siguiente", command=self.save_and_next, state=tk.DISABLED)
        if MODERN_UI:
            self.save_next_btn.config(bootstyle=SUCCESS)
        self.save_next_btn.grid(row=1, column=1, sticky="ew", padx=2)
        self.next_btn = ttk.Button(controls_frame, text="Siguiente >>", command=self.go_next, state=tk.DISABLED); self.next_btn.grid(row=1, column=2, sticky="ew", padx=2)
        self.execute_btn = ttk.Button(main_frame, text="Ejecutar Cambios...", command=self.execute_renames, state=tk.DISABLED)
        if MODERN_UI:
            self.execute_btn.config(bootstyle=DANGER)
        self.execute_btn.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.status_label = ttk.Label(main_frame, text="Selecciona una carpeta para empezar.", anchor="w"); self.status_label.grid(row=4, column=0, sticky="ew", pady=(5,0))
    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Selecciona la carpeta con tus archivos CBZ");
        if not folder_path: return
        self.files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.cbz')], key=natural_sort_key); self.rename_plan = {}
        if not self.files: messagebox.showinfo("Vacío", "No se encontraron archivos .cbz.", parent=self); return
        self.load_comic_at_index(0)
    def load_comic_at_index(self, index):
        if not (0 <= index < len(self.files)): return
        self.current_index = index; filepath = self.files[index]; basename = os.path.basename(filepath)
        w, h = self.cover_label.winfo_width(), self.cover_label.winfo_height();
        if w < 10 or h < 10: w, h = 800, 800
        self.cover_photo = get_cover_from_cbz(filepath, (w-10, h-10))
        if self.cover_photo: self.cover_label.config(image=self.cover_photo)
        else: self.cover_label.config(image="", text="No se pudo cargar la portada", foreground="white", font=("-size", 14))
        self.current_name_label.config(text=basename); new_name = os.path.basename(self.rename_plan.get(filepath, filepath)); self.new_name_entry.delete(0, tk.END); self.new_name_entry.insert(0, new_name); self.new_name_entry.focus_set(); self.new_name_entry.selection_range(0, tk.END)
        status_text = f"Archivo {index + 1} de {len(self.files)}";
        if filepath in self.rename_plan: status_text += " (Cambio guardado)"
        self.status_label.config(text=status_text); self.update_button_states()
    def update_button_states(self):
        has_files = bool(self.files); is_not_first = self.current_index > 0; is_not_last = self.current_index < len(self.files) - 1
        self.prev_btn.config(state=tk.NORMAL if has_files and is_not_first else tk.DISABLED); self.next_btn.config(state=tk.NORMAL if has_files and is_not_last else tk.DISABLED)
        self.save_next_btn.config(state=tk.NORMAL if has_files else tk.DISABLED); self.execute_btn.config(state=tk.NORMAL if self.rename_plan else tk.DISABLED)
    def save_and_next(self):
        old_path = self.files[self.current_index]; new_basename = self.new_name_entry.get().strip()
        if not new_basename.lower().endswith('.cbz'): new_basename += '.cbz'
        if new_basename != os.path.basename(old_path): self.rename_plan[old_path] = os.path.join(os.path.dirname(old_path), new_basename)
        elif old_path in self.rename_plan: del self.rename_plan[old_path]
        if self.current_index < len(self.files) - 1: self.load_comic_at_index(self.current_index + 1)
        else: messagebox.showinfo("Fin de la lista", "Has llegado al final.", parent=self); self.update_button_states()
    def go_next(self): self.load_comic_at_index(self.current_index + 1)
    def go_previous(self): self.load_comic_at_index(self.current_index - 1)
    def execute_renames(self):
        if not self.rename_plan: messagebox.showinfo("Nada que hacer", "No hay archivos para renombrar.", parent=self); return
        final_names = list(self.rename_plan.values())
        if len(final_names) != len(set(final_names)):
            duplicates = [os.path.basename(item) for item, count in Counter(final_names).items() if count > 1]
            messagebox.showerror("Conflicto de Nombres", f"Has asignado el mismo nombre nuevo a varios archivos:\n\n- {duplicates[0]}", parent=self); return
        plan_str = "\n".join([f"DE: {os.path.basename(k)}\nA:   {os.path.basename(v)}\n" for k, v in list(self.rename_plan.items())[:5]])
        if len(self.rename_plan) > 5: plan_str += "\n..."
        if not messagebox.askyesno("Confirmar Cambios", f"¿Renombrar {len(self.rename_plan)} archivos?\n\nEjemplo:\n{plan_str}", parent=self): return
        temp_to_final_map, renamed_in_pass1 = [], []
        try:
            for old_path, new_path in self.rename_plan.items():
                if not os.path.exists(old_path): continue
                temp_path = old_path + ".tmp_rename"; os.rename(old_path, temp_path); renamed_in_pass1.append(temp_path); temp_to_final_map.append((temp_path, new_path))
        except Exception as e:
            for path in renamed_in_pass1: os.rename(path, path.replace(".tmp_rename", "")); messagebox.showerror("Error en Fase 1", f"Error al renombrar. Cambios revertidos.\n\nError: {e}", parent=self); return
        succeeded, failed = 0, 0
        for temp_path, new_path in temp_to_final_map:
            try: os.rename(temp_path, new_path); succeeded += 1
            except Exception as e:
                failed += 1; print(f"FALLO Fase 2: {os.path.basename(temp_path)} -> {os.path.basename(new_path)}: {e}")
                try: os.rename(temp_path, temp_path.replace(".tmp_rename", ""))
                except Exception as e_revert: print(f"FALLO al revertir: {e_revert}")
        if failed > 0: messagebox.showwarning("Proceso con Fallos", f"Éxitos: {succeeded}\nFallos: {failed}", parent=self)
        else: messagebox.showinfo("Proceso Completado", f"{succeeded} archivos renombrados con éxito.", parent=self)
        self.destroy()

class BatchTranslatorWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent); self.title("Traductor de Biblioteca"); self.geometry("700x500"); self.transient(parent); self.grab_set()
        self.cbz_files = []; self.campos_a_traducir = { "Summary": tk.BooleanVar(value=True), "Title": tk.BooleanVar(value=True), "StoryArc": tk.BooleanVar(value=True), "Notes": tk.BooleanVar(value=False), "Characters": tk.BooleanVar(value=False), "Teams": tk.BooleanVar(value=False) }; self.omitir_espanol = tk.BooleanVar(value=True)
        main_frame = ttk.Frame(self, padding=10); main_frame.pack(fill=tk.BOTH, expand=True); main_frame.rowconfigure(1, weight=1); main_frame.columnconfigure(0, weight=1)
        top_frame = ttk.Frame(main_frame); top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10)); top_frame.columnconfigure(1, weight=1)
        self.select_folder_btn = ttk.Button(top_frame, text="Seleccionar Carpeta...", command=self.select_folder); self.select_folder_btn.grid(row=0, column=0, sticky="w")
        label_config = {"text": "Ninguna carpeta seleccionada", "anchor": "w"}
        if MODERN_UI:
            label_config["bootstyle"] = SECONDARY
        self.folder_path_label = ttk.Label(top_frame, **label_config)
        self.folder_path_label.grid(row=0, column=1, sticky="ew", padx=10)
        list_frame = ttk.LabelFrame(main_frame, text="Cómics Encontrados"); list_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10)); list_frame.rowconfigure(0, weight=1); list_frame.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(list_frame); self.listbox.grid(row=0, column=0, sticky="nsew"); scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview); scrollbar.grid(row=0, column=1, sticky="ns"); self.listbox.config(yscrollcommand=scrollbar.set)
        options_frame = ttk.LabelFrame(main_frame, text="Opciones de Traducción"); options_frame.grid(row=1, column=1, sticky="nsew")
        ttk.Label(options_frame, text="Traducir los siguientes campos:", font="-weight bold").pack(anchor="w", padx=5, pady=5)
        for campo, var in self.campos_a_traducir.items(): ttk.Checkbutton(options_frame, text=campo, variable=var).pack(anchor="w", padx=15)
        ttk.Separator(options_frame, orient='horizontal').pack(fill='x', pady=10, padx=5); ttk.Checkbutton(options_frame, text="Omitir cómics ya en español", variable=self.omitir_espanol).pack(anchor="w", padx=5, pady=5)
        bottom_frame = ttk.Frame(main_frame); bottom_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0)); bottom_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Listo."); ttk.Label(bottom_frame, textvariable=self.status_var, anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew")
        self.progress_bar = ttk.Progressbar(bottom_frame, orient='horizontal', mode='determinate'); self.progress_bar.grid(row=1, column=0, sticky="ew", pady=5)
        self.start_btn = ttk.Button(bottom_frame, text="Iniciar Traducción", command=self.start_translation, state=tk.DISABLED)
        if MODERN_UI:
            self.start_btn.config(bootstyle=SUCCESS)
        self.start_btn.grid(row=1, column=1, sticky="e", padx=(10, 0))
    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Selecciona la carpeta con tus CBZ");
        if not folder_path: return
        self.folder_path_label.config(text=folder_path)
        if MODERN_UI:
            self.folder_path_label.config(bootstyle=DEFAULT)
        self.listbox.delete(0, tk.END); self.cbz_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".cbz")])
        for file_path in self.cbz_files: self.listbox.insert(tk.END, os.path.basename(file_path))
        if self.cbz_files: self.start_btn.config(state=tk.NORMAL); self.status_var.set(f"{len(self.cbz_files)} cómics encontrados.")
        else: self.start_btn.config(state=tk.DISABLED); self.status_var.set("No se encontraron archivos .cbz.")
    def start_translation(self):
        if not self.cbz_files: messagebox.showwarning("Sin archivos", "No hay archivos CBZ para procesar.", parent=self); return
        if not any(var.get() for var in self.campos_a_traducir.values()): messagebox.showwarning("Sin selección", "Selecciona al menos un campo para traducir.", parent=self); return
        if not get_translator():
             messagebox.showerror("Error de API", "La clave de API de DeepL no está configurada o no es válida.", parent=self); return
        self.start_btn.config(state=tk.DISABLED); self.select_folder_btn.config(state=tk.DISABLED); self.progress_bar['maximum'] = len(self.cbz_files); self.progress_bar['value'] = 0
        campos_seleccionados = [campo for campo, var in self.campos_a_traducir.items() if var.get()]; threading.Thread(target=self._translate_thread, args=(self.cbz_files, campos_seleccionados, self.omitir_espanol.get()), daemon=True).start()
    def _translate_thread(self, files_to_process, campos, omitir_es):
        traducidos, omitidos, errores = 0, 0, 0
        for i, cbz_path in enumerate(files_to_process):
            self.after(0, self.update_status, f"Procesando ({i+1}/{len(files_to_process)}): {os.path.basename(cbz_path)}", i)
            try:
                metadata = read_comicinfo_from_cbz(cbz_path)
                if not metadata or (omitir_es and metadata.get("LanguageISO", "").lower() == "es"):
                    omitidos += 1; continue
                comic_modificado = False
                for campo in campos:
                    texto_original = metadata.get(campo)
                    if texto_original and texto_original.strip():
                        resultado = get_translator().translate_text(texto_original, target_lang="ES")
                        if resultado.text.lower() != texto_original.lower():
                            metadata[campo] = resultado.text; comic_modificado = True
                if comic_modificado:
                    metadata["LanguageISO"] = "es"; xml_string = generate_comicinfo_xml(metadata)
                    if inject_xml_into_cbz(cbz_path, xml_string): traducidos += 1
                    else: errores += 1
                else: omitidos += 1
            except Exception as e:
                print(f"Error procesando {os.path.basename(cbz_path)}: {e}"); errores += 1
        self.after(0, self.finish_translation, traducidos, omitidos, errores, len(files_to_process))
    def update_status(self, message, value): self.status_var.set(message); self.progress_bar['value'] = value
    def finish_translation(self, traducidos, omitidos, errores, total):
        self.status_var.set("Proceso completado."); self.progress_bar['value'] = total
        self.start_btn.config(state=tk.NORMAL); self.select_folder_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Proceso Finalizado", f"Traducidos: {traducidos}\nOmitidos: {omitidos}\nErrores: {errores}\nTotal: {total}", parent=self)

class ManualSplitterWindow(tk.Toplevel):
    def __init__(self, parent, files, folder):
        super().__init__(parent); self.title("Divisor de Tomos - Paso 1: Seleccionar Portadas"); self.geometry("1000x800"); self.transient(parent); self.grab_set()
        self.all_files = files; self.folder_path = folder; self.current_index = 0; self.cover_indices = {0}; self.final_indices = None; self.photo = None
        main_frame = ttk.Frame(self, padding=10); main_frame.pack(fill=tk.BOTH, expand=True); main_frame.rowconfigure(0, weight=1); main_frame.columnconfigure(0, weight=3); main_frame.columnconfigure(1, weight=1)
        preview_frame = ttk.Frame(main_frame); preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10)); preview_frame.rowconfigure(1, weight=1); preview_frame.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(preview_frame, text="", font="-weight bold"); self.status_label.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        self.preview_label = ttk.Label(preview_frame, background="black", anchor=CENTER); self.preview_label.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.prev_btn = ttk.Button(preview_frame, text="<< Anterior", command=self.go_previous)
        self.prev_btn.grid(row=2, column=0, sticky="ew", pady=10)
        self.mark_btn = ttk.Button(preview_frame, text="Marcar/Desmarcar Portada", command=self.toggle_cover_mark)
        if MODERN_UI:
            self.mark_btn.config(bootstyle=SUCCESS)
        self.mark_btn.grid(row=2, column=1, sticky="ew", padx=10, pady=10)
        self.next_btn = ttk.Button(preview_frame, text="Siguiente >>", command=self.go_next)
        self.next_btn.grid(row=2, column=2, sticky="ew", pady=10)
        list_frame = ttk.LabelFrame(main_frame, text="Portadas Marcadas"); list_frame.grid(row=0, column=1, sticky="nsew"); list_frame.rowconfigure(0, weight=1); list_frame.columnconfigure(0, weight=1)
        self.covers_listbox = tk.Listbox(list_frame); self.covers_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.covers_listbox.yview); scrollbar.grid(row=0, column=1, sticky="ns", pady=5); self.covers_listbox.config(yscrollcommand=scrollbar.set)
        label_config = {"text": "(Doble clic para quitar)"}
        if MODERN_UI:
            label_config["bootstyle"] = SECONDARY
        ttk.Label(list_frame, **label_config).grid(row=1, column=0, columnspan=2, pady=(0, 5))
        finish_btn = ttk.Button(main_frame, text="Finalizar Selección y Continuar", command=self.on_finish)
        if MODERN_UI:
            finish_btn.config(bootstyle=PRIMARY)
        finish_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10,0))
        self.after(100, lambda: self.load_page(0)); self.update_covers_list(); self.wait_window()
    def load_page(self, index):
        self.current_index = index; filepath = self.all_files[index]; self.update_idletasks(); w, h = self.preview_label.winfo_width(), self.preview_label.winfo_height()
        try:
            with Image.open(filepath) as img:
                if w > 1 and h > 1: img.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(img); self.preview_label.config(image=self.photo)
        except Exception as e: self.preview_label.config(image="", text=f"Error al cargar\n{e}")
        
        self.status_label.config(text=f"Página {index + 1} de {len(self.all_files)} - {os.path.basename(filepath)}")
        self.prev_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if index < len(self.all_files) - 1 else tk.DISABLED)
        
        # Lógica para actualizar el botón de marcar/desmarcar
        if self.current_index in self.cover_indices:
            self.mark_btn.config(text="Desmarcar Portada")
            if MODERN_UI:
                self.mark_btn.config(bootstyle=DANGER)
        else:
            self.mark_btn.config(text="Marcar como Portada")
            if MODERN_UI:
                self.mark_btn.config(bootstyle=SUCCESS)
    def go_next(self):
        if self.current_index < len(self.all_files) - 1: self.load_page(self.current_index + 1)
    def go_previous(self):
        if self.current_index > 0: self.load_page(self.current_index - 1)
    
    def toggle_cover_mark(self):
        """Marca o desmarca la página actual como portada."""
        if self.current_index in self.cover_indices:
            # No permitir desmarcar la primera página (índice 0)
            if self.current_index == 0:
                messagebox.showwarning("Acción no permitida", "La primera página (Pág. 1) debe ser siempre una portada.", parent=self)
                return
            self.cover_indices.remove(self.current_index)
            self.mark_btn.config(text="Marcar como Portada")
            if MODERN_UI:
                self.mark_btn.config(bootstyle=SUCCESS)
        else:
            self.cover_indices.add(self.current_index)
            self.mark_btn.config(text="Desmarcar Portada")
            if MODERN_UI:
                self.mark_btn.config(bootstyle=DANGER)
        
        self.update_covers_list()

    def remove_cover_from_listbox(self, event=None):
        """Elimina una portada haciendo doble clic en la lista."""
        selection = self.covers_listbox.curselection()
        if not selection:
            return
        
        selected_text = self.covers_listbox.get(selection[0])
        # Extraer el número de página del texto "Pág. X: ..."
        try:
            page_number = int(re.search(r'Pág\. (\d+):', selected_text).group(1))
            index_to_remove = page_number - 1
            
            if index_to_remove == 0:
                messagebox.showwarning("Acción no permitida", "No se puede quitar la primera portada (Pág. 1).", parent=self)
                return

            if index_to_remove in self.cover_indices:
                self.cover_indices.remove(index_to_remove)
                self.update_covers_list()
                # Actualizar el botón principal si estábamos viendo la página que acabamos de desmarcar
                if index_to_remove == self.current_index:
                    self.mark_btn.config(text="Marcar como Portada", bootstyle=SUCCESS)

        except (AttributeError, ValueError) as e:
            print(f"Error al procesar la selección de la lista: {e}")
    def update_covers_list(self):
        self.covers_listbox.delete(0, tk.END)
        for index in sorted(list(self.cover_indices)):
            self.covers_listbox.insert(tk.END, f"Pág. {index+1}: {os.path.basename(self.all_files[index])}")
    def on_finish(self):
        if len(self.cover_indices) < 1: messagebox.showwarning("Sin Selección", "Debes marcar al menos la primera página como portada.", parent=self); return
        self.final_indices = sorted(list(self.cover_indices)); self.destroy()

class BatchConfigDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, initialvalue=None):
        self.initialvalue = initialvalue or {}; super().__init__(parent, title)
    def body(self, master):
        self.title("Divisor de Tomos - Paso 2: Configuración"); ttk.Label(master, text="Nombre de la Serie:").grid(row=0, sticky="w", padx=5, pady=5)
        self.series_entry = ttk.Entry(master, width=40); self.series_entry.grid(row=0, column=1, padx=5, pady=5); self.series_entry.insert(0, self.initialvalue.get("series_name", ""))
        ttk.Label(master, text="Número del primer cómic:").grid(row=1, sticky="w", padx=5, pady=5)
        self.number_entry = ttk.Entry(master, width=10); self.number_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5); self.number_entry.insert(0, self.initialvalue.get("start_number", "1"))
        return self.series_entry
    def validate(self):
        try: int(self.number_entry.get()); return True
        except ValueError: messagebox.showerror("Error", "El número inicial debe ser un entero.", parent=self); return False
    def apply(self):
        self.result = {'series_name': self.series_entry.get().strip(), 'start_number': int(self.number_entry.get())}
        
class MarathonDialog(tk.Toplevel):
    def __init__(self, parent, all_files, indices, config):
        super().__init__(parent); self.title("Divisor de Tomos - Paso 3: Confirmar Plan"); self.geometry("800x600"); self.transient(parent); self.grab_set()
        self.results = None; main_frame = ttk.Frame(self, padding=10); main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="Se crearán los siguientes archivos CBZ. Revisa el plan y confirma.", justify=tk.LEFT).pack(anchor="w", pady=(0, 10))
        cols = ("Archivo", "Páginas", "Total Páginas"); self.tree = ttk.Treeview(main_frame, columns=cols, show='headings');
        for col in cols: self.tree.heading(col, text=col)
        self.tree.column("Archivo", width=400); self.tree.column("Páginas", width=150, anchor=CENTER); self.tree.column("Total Páginas", width=100, anchor=CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True); self.generate_plan(all_files, indices, config)
        button_frame = ttk.Frame(main_frame); button_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_frame, text="Cancelar", command=self.on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Confirmar y Crear", command=self.on_confirm).pack(side=tk.RIGHT)
        self.wait_window()
    def generate_plan(self, all_files, indices, config):
        self.results = []; series_name = config['series_name']; start_number = config['start_number']
        for i, start_idx in enumerate(indices):
            end_idx = indices[i+1] if i+1 < len(indices) else len(all_files); issue_num = start_number + i
            filename = f"{series_name} #{issue_num:03d}.cbz"; query_search = f"{series_name} #{issue_num}"
            page_range = f"{start_idx + 1} - {end_idx}"; page_count = end_idx - start_idx
            self.tree.insert("", tk.END, values=(filename, page_range, page_count))
            self.results.append({'filename': filename, 'query': query_search, 'start_index': start_idx, 'end_index': end_idx})
    def on_confirm(self): self.destroy()
    def on_cancel(self): self.results = None; self.destroy()

class ApiKeysWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent); self.title("Configuración de Claves API"); self.geometry("600x280"); self.transient(parent); self.grab_set()
        main_frame = ttk.Frame(self, padding=20); main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="Introduce tus claves API para activar las funciones online.", wraplength=550).pack(pady=(0, 15))
        cv_frame = ttk.LabelFrame(main_frame, text="Comic Vine API Key", padding=10); cv_frame.pack(fill=tk.X, pady=5)
        self.cv_key_var = tk.StringVar(); cv_entry = ttk.Entry(cv_frame, textvariable=self.cv_key_var, width=60); cv_entry.pack(fill=tk.X)
        deepl_frame = ttk.LabelFrame(main_frame, text="DeepL API Key (Free o Pro)", padding=10); deepl_frame.pack(fill=tk.X, pady=5)
        self.deepl_key_var = tk.StringVar(); deepl_entry = ttk.Entry(deepl_frame, textvariable=self.deepl_key_var, width=60, show="*"); deepl_entry.pack(fill=tk.X)
        button_frame = ttk.Frame(main_frame, padding=(0, 15, 0, 0)); button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(button_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Guardar Claves", command=self.save_keys).pack(side=tk.RIGHT)
        self.load_keys(); self.wait_window()
    def load_keys(self):
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
            if 'APIKeys' in config:
                self.cv_key_var.set(config['APIKeys'].get('comicvine_api_key', '')); self.deepl_key_var.set(config['APIKeys'].get('deepl_api_key', ''))
    def save_keys(self):
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'): 
            config.read('config.ini')
        
        if 'APIKeys' not in config: 
            config['APIKeys'] = {}
        
        cv_key = self.cv_key_var.get().strip()
        deepl_key = self.deepl_key_var.get().strip()
        
        config['APIKeys']['comicvine_api_key'] = cv_key
        config['APIKeys']['deepl_api_key'] = deepl_key
        
        try:
            with open('config.ini', 'w') as configfile: 
                config.write(configfile)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la configuración: {e}", parent=self)
            return
        
        global COMICVINE_API_KEY, DEEPL_API_KEY, translator
        COMICVINE_API_KEY = config['APIKeys']['comicvine_api_key']
        DEEPL_API_KEY = config['APIKeys']['deepl_api_key']
        translator = None
        
        cv_status = "✓" if COMICVINE_API_KEY else "✗"
        deepl_status = "✓" if DEEPL_API_KEY else "✗"
        print(f"🔑 API Keys guardadas y actualizadas - ComicVine: {cv_status}, DeepL: {deepl_status}")
        
        messagebox.showinfo("Guardado", "Claves API guardadas con éxito.\nLas nuevas claves se usarán en esta sesión.", parent=self)
        self.destroy()

class AuthorEditorWindow(tk.Toplevel):
    def __init__(self, parent, author_name):
        super().__init__(parent)
        self.author_name = author_name; self.app = parent.app if hasattr(parent, 'app') else parent; self.db_file = self.app.db_file
        self.author_images_path = self.app.author_images_path; self.new_photo_path = None
        self.title(f"Editor de Autor - {self.author_name}"); self.geometry("700x550"); self.transient(parent); self.grab_set()
        main_frame = ttk.Frame(self, padding=10); main_frame.pack(fill=tk.BOTH, expand=True); main_frame.rowconfigure(3, weight=1); main_frame.columnconfigure(1, weight=1)
        photo_panel = ttk.Frame(main_frame); photo_panel.grid(row=0, column=0, rowspan=5, sticky="ns", padx=(0, 15))
        self.photo_label = ttk.Label(photo_panel, text="Sin foto", background="black", anchor="center"); self.photo_label.pack()
        ttk.Button(photo_panel, text="Subir Foto Local...", command=self.select_photo).pack(fill=tk.X, pady=(10,0))
        ttk.Label(main_frame, text="URL de la imagen:").grid(row=0, column=1, sticky="w"); self.url_entry = ttk.Entry(main_frame); self.url_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(main_frame, text="↓ Descargar y Usar Imagen de URL ↓", command=self.download_photo_from_url).grid(row=2, column=1, sticky="ew", pady=(5,10))
        ttk.Label(main_frame, text="Biografía:").grid(row=3, column=1, sticky="nw"); self.bio_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD); self.bio_text.grid(row=4, column=1, sticky="nsew")
        button_frame = ttk.Frame(main_frame); button_frame.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(button_frame, text="Cancelar", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Guardar Cambios", command=self.save_data).pack(side=tk.RIGHT)
        self.load_data()
    def load_data(self):
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor(); cursor.execute("SELECT biography, photo_filename FROM authors WHERE name = ?", (self.author_name,)); data = cursor.fetchone(); conn.close()
        if not data: return
        self.bio_text.insert(tk.END, data[0] or "")
        if data[1]:
            photo_path = self.author_images_path / data[1]
            if photo_path.exists(): self.update_photo_preview(photo_path)
    def select_photo(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp")])
        if not path: return
        self.new_photo_path = Path(path); self.update_photo_preview(self.new_photo_path)
    def download_photo_from_url(self):
        url = self.url_entry.get().strip()
        if not url: return messagebox.showwarning("URL Vacía", "Pega una URL en el campo.", parent=self)
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True); response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                shutil.copyfileobj(response.raw, temp_file)
            self.new_photo_path = Path(temp_file.name); self.update_photo_preview(self.new_photo_path)
            messagebox.showinfo("Éxito", "Imagen descargada. Pulsa 'Guardar Cambios'.", parent=self)
        except Exception as e: messagebox.showerror("Error", f"No se pudo descargar la imagen.\nError: {e}", parent=self)
    def update_photo_preview(self, path_to_image):
        try:
            photo = create_circular_photo(path_to_image, size=200)
            if photo:
                self.photo_label.config(image=photo, text=""); self.photo_label.image = photo
            else:
                self.photo_label.config(image="", text="Error al\nprevisualizar")
        except Exception as e: self.photo_label.config(image="", text="Error al\nprevisualizar"); print(f"Error al previsualizar: {e}")
    def save_data(self):
        bio = self.bio_text.get(1.0, tk.END).strip(); photo_filename = None
        if self.new_photo_path:
            safe_name = re.sub(r'[\\/*?:"<>|]', "", self.author_name.replace(" ", "_"))
            try:
                with Image.open(self.new_photo_path) as img: extension = f".{img.format.lower() if img.format else 'jpg'}"
            except: extension = self.new_photo_path.suffix or ".jpg"
            photo_filename = f"{safe_name}{extension}"; destination = self.author_images_path / photo_filename
            shutil.copy(self.new_photo_path, destination)
            if ".tmp" in str(self.new_photo_path): os.remove(self.new_photo_path)
        else:
            conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
            cursor.execute("SELECT photo_filename FROM authors WHERE name = ?", (self.author_name,)); res = cursor.fetchone()
            if res: photo_filename = res[0]
            conn.close()
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        cursor.execute("UPDATE authors SET biography = ?, photo_filename = ? WHERE name = ?", (bio, photo_filename, self.author_name))
        conn.commit(); conn.close(); self.destroy()

class AuthorDetailWindow(tk.Toplevel):
    def __init__(self, app_instance, author_name):
        super().__init__(app_instance.root)
        self.app = app_instance
        self.author_name = author_name
        self.db_file = self.app.db_file
        self.author_images_path = self.app.author_images_path
        
        self.title(f"Ficha de Autor - {self.author_name}")
        self.geometry("800x600")
        self.transient(app_instance.root)
        self.grab_set()

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)

        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        PHOTO_SIZE = 180
        photo_frame = ttk.Frame(left_panel, width=PHOTO_SIZE, height=PHOTO_SIZE)
        photo_frame.pack(pady=(10, 0))
        photo_frame.pack_propagate(False)

        self.photo_label = ttk.Label(photo_frame, background="black")
        self.photo_label.pack(fill=tk.BOTH, expand=True)

        ttk.Button(left_panel, text="Editar este autor...", command=self.open_editor).pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(main_frame, text=self.author_name, font=('Impact', 22)).grid(row=0, column=1, sticky="w")
        
        tabs = ttk.Notebook(main_frame)
        tabs.grid(row=1, column=1, sticky="nsew", pady=(10, 0))
        
        bio_tab = ttk.Frame(tabs, padding=5)
        tabs.add(bio_tab, text="Biografía")
        self.bio_text = scrolledtext.ScrolledText(bio_tab, wrap=tk.WORD, state="disabled")
        self.bio_text.pack(fill=tk.BOTH, expand=True)
        
        works_tab = ttk.Frame(tabs)
        tabs.add(works_tab, text="Obras en la Biblioteca")

        self.works_canvas = tk.Canvas(works_tab, highlightthickness=0)
        works_scrollbar = ttk.Scrollbar(works_tab, orient=tk.VERTICAL, command=self.works_canvas.yview)
        self.works_scrollable_frame = ttk.Frame(self.works_canvas)

        self.works_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.works_canvas.configure(scrollregion=self.works_canvas.bbox("all"))
        )

        self.works_canvas.create_window((0, 0), window=self.works_scrollable_frame, anchor="nw")
        self.works_canvas.configure(yscrollcommand=works_scrollbar.set)

        self.works_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        works_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.work_thumbnails = {}

        self.load_author_data()
        self.wait_window()

    def load_author_data(self):
        conn = sqlite3.connect(self.db_file); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("SELECT * FROM authors WHERE name = ?", (self.author_name,)); author = cursor.fetchone()
        
        if not author:
            self.destroy()
            conn.close()
            return
            
        self.bio_text.config(state="normal")
        self.bio_text.delete(1.0, tk.END)
        self.bio_text.insert(tk.END, author['biography'] or "No hay biografía disponible.")
        self.bio_text.config(state="disabled")

        self.photo_label.config(image=None)
        self.photo_label.image = None
        if author['photo_filename']:
            photo_path = self.author_images_path / author['photo_filename']
            if photo_path.exists():
                circular_photo = create_circular_photo(photo_path, size=180)
                if circular_photo:
                    self.photo_label.config(image=circular_photo)
                    self.photo_label.image = circular_photo
        
        for widget in self.works_scrollable_frame.winfo_children():
            widget.destroy()
        
        cursor.execute("""
            SELECT c.path, c.series, c.number, ca.role FROM comics c 
            JOIN comic_authors ca ON c.id = ca.comic_id
            WHERE ca.author_id = ? ORDER BY c.series, CAST(c.number as REAL)
        """, (author['id'],))
        
        works = cursor.fetchall()
        conn.close()

        self.update_idletasks() # Forzar a que el canvas tenga tamaño antes de calcular columnas
        THUMB_WIDTH = 120
        container_width = self.works_canvas.winfo_width()
        cols = max(1, container_width // (THUMB_WIDTH + 10))
        
        for i, work in enumerate(works):
            row, col = divmod(i, cols)
            
            path, series, number, role = work
            
            frame = ttk.Frame(self.works_scrollable_frame, padding=5)
            frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
            
            cover_label = ttk.Label(frame, text="Cargando...", anchor=CENTER)
            cover_label.pack(fill=tk.X)
            
            title_text = f"{series or '?'} #{number or '?'}"
            title_label = ttk.Label(frame, text=title_text, anchor=CENTER, wraplength=THUMB_WIDTH)
            title_label.pack(fill=tk.X, pady=(5,0))
            
            role_label = ttk.Label(frame, text=f"({role})", bootstyle=SECONDARY, anchor=CENTER)
            role_label.pack(fill=tk.X)

            threading.Thread(
                target=self.load_work_thumbnail,
                args=(path, cover_label),
                daemon=True
            ).start()
            
            for widget in (frame, cover_label, title_label, role_label):
                widget.bind("<Button-1>", lambda e, p=path: self.on_comic_click(p))
                
        self.update_idletasks()

    def load_work_thumbnail(self, path, cover_label):
        if path in self.work_thumbnails:
            photo = self.work_thumbnails[path]
        else:
            photo = get_cover_from_cbz(path, size=(120, 180))
            if photo:
                self.work_thumbnails[path] = photo
        
        if photo and self.winfo_exists():
            self.after(0, lambda: cover_label.config(image=photo, text=""))

    def on_comic_click(self, comic_path):
        self.app.notebook.select(self.app.library_tab)
        
        if self.app.library_view_mode.get() != "list":
            self.app._toggle_library_view()
            self.app.root.update_idletasks()
            
        for group_node in self.app.library_tree.get_children():
            for comic_node in self.app.library_tree.get_children(group_node):
                if self.app.library_tree.item(comic_node, 'values')[0] == comic_path:
                    self.app.library_tree.selection_set(comic_node)
                    self.app.library_tree.focus(comic_node)
                    self.app.library_tree.see(comic_node)
                    self.app.on_comic_selected(comic_path)
                    self.destroy()
                    return

    def open_editor(self):
        editor = AuthorEditorWindow(self, self.author_name)
        self.wait_window(editor)
        self.load_author_data()

class AuthorManagementWindow(tk.Toplevel):
    def __init__(self, app_instance):
        super().__init__(app_instance.root); self.app = app_instance; self.db_file = self.app.db_file; self.all_authors = []
        self.title("Gestionar Autores"); self.geometry("600x500"); self.transient(app_instance.root); self.grab_set()
        main_frame = ttk.Frame(self, padding=10); main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_frame); top_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top_frame, text="Buscar autor:").pack(side=tk.LEFT, padx=(0, 5)); self.search_entry = ttk.Entry(top_frame); self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True); self.search_entry.bind("<KeyRelease>", self.filter_authors)
        ttk.Button(top_frame, text="Añadir Nuevo Autor...", command=self.add_new_author).pack(side=tk.RIGHT, padx=(10, 0))
        list_frame = ttk.Frame(main_frame); list_frame.pack(fill=tk.BOTH, expand=True)
        self.authors_listbox = tk.Listbox(list_frame); self.authors_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.authors_listbox.yview); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.authors_listbox.config(yscrollcommand=scrollbar.set)
        self.authors_listbox.bind("<Double-Button-1>", self.on_select); self.load_all_authors(); self.wait_window()
    def load_all_authors(self):
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor(); cursor.execute("SELECT name FROM authors ORDER BY name")
        self.all_authors = [row[0] for row in cursor.fetchall()]; conn.close(); self.filter_authors()
    def populate_listbox(self, authors_to_display):
        self.authors_listbox.delete(0, tk.END)
        for author in authors_to_display: self.authors_listbox.insert(tk.END, author)
    def filter_authors(self, event=None):
        query = self.search_entry.get().lower().strip()
        filtered_authors = self.all_authors if not query else [name for name in self.all_authors if query in name.lower()]
        self.populate_listbox(filtered_authors)
    def on_select(self, event=None):
        selection = self.authors_listbox.curselection()
        if not selection: return
        author_name = self.authors_listbox.get(selection[0]); AuthorDetailWindow(self.app, author_name); self.load_all_authors()
    def add_new_author(self):
        name = simpledialog.askstring("Nuevo Autor", "Nombre del nuevo autor:", parent=self)
        if not name or not name.strip(): return
        name = name.strip(); conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO authors (name) VALUES (?)", (name,)); conn.commit()
            messagebox.showinfo("Éxito", f"Autor '{name}' añadido. Ahora puedes editar sus detalles.", parent=self); self.load_all_authors()
            editor = AuthorEditorWindow(self, name); self.wait_window(editor); self.load_all_authors()
        except sqlite3.IntegrityError: messagebox.showwarning("Duplicado", f"El autor '{name}' ya existe.", parent=self)
        finally: conn.close()

class ComicReaderWindow(tk.Toplevel):
    def __init__(self, parent, cbz_path, reading_order_context=None):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.cbz_path = cbz_path
        
        self.reading_order_context = reading_order_context
        self.parent_app = parent.app if hasattr(parent, 'app') else app 
        
        self.image_files = []
        self.current_page_index = 0
        
        self.pil_image = None
        self.tk_photo = None
        self.zoom_level = 1.0
        
        self.title(f"Lector - {os.path.basename(cbz_path)}")
        self.configure(bg="black")
        self.attributes('-fullscreen', True)

        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._load_image_list()
        if self.image_files:
            self._load_page(0)
        else:
            self.destroy()
            messagebox.showerror("Error", "Este archivo CBZ no contiene imágenes.", parent=parent)
            return

        self.bind("<KeyPress-Escape>", self._close)
        self.bind("<KeyPress-Right>", self._on_next_page_event)
        self.bind("<KeyPress-Left>", self._on_prev_page_event)
        self.canvas.bind("<Button-1>", self._on_tap)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)
        self.bind("<Configure>", self._on_resize)

    def _load_image_list(self):
        try:
            with zipfile.ZipFile(self.cbz_path, 'r') as zf:
                supported = ('.jpg', '.jpeg', '.png', '.webp')
                self.image_files = sorted(
                    [f for f in zf.namelist() if f.lower().endswith(supported)],
                    key=natural_sort_key
                )
        except Exception as e:
            print(f"Error al leer el archivo CBZ: {e}")

    def _load_page(self, page_index):
        if page_index >= len(self.image_files):
            if self.reading_order_context and hasattr(self.parent_app, 'start_reading_order'):
                self.parent_app.start_reading_order(self.reading_order_context['order_id'], self.reading_order_context['current_index'] + 1)
                self.destroy()
                return
            else:
                return
        
        if page_index < 0:
            if self.reading_order_context and hasattr(self.parent_app, 'start_reading_order'):
                self.parent_app.start_reading_order(self.reading_order_context['order_id'], self.reading_order_context['current_index'] - 1)
                self.destroy()
                return
            else:
                return

        self.current_page_index = page_index
        try:
            with zipfile.ZipFile(self.cbz_path, 'r') as zf:
                with zf.open(self.image_files[page_index]) as image_file:
                    self.pil_image = Image.open(BytesIO(image_file.read()))
                    if self.pil_image.mode != 'RGB':
                        self.pil_image = self.pil_image.convert('RGB')
            
            self.zoom_level = 1.0
            self.canvas.delete("all")
            self._update_display()
        except Exception as e:
            print(f"Error al cargar la página {page_index}: {e}")
            
    def _update_display(self):
        if not self.pil_image: return
        canvas_w = self.canvas.winfo_width(); canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return
        img_w, img_h = self.pil_image.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        display_w = int(img_w * ratio * self.zoom_level)
        display_h = int(img_h * ratio * self.zoom_level)
        
        if display_w < 1 or display_h < 1: return

        resized_img = self.pil_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_photo = ImageTk.PhotoImage(resized_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w / 2, canvas_h / 2, image=self.tk_photo, anchor="center")
        
        title_text = f"Pág {self.current_page_index + 1}/{len(self.image_files)} - {os.path.basename(self.cbz_path)}"
        if self.reading_order_context:
            order_info = f" | Orden: {self.reading_order_context['order_name']} ({self.reading_order_context['current_index'] + 1}/{self.reading_order_context['total_comics']})"
            title_text += order_info
        self.title(title_text)

    def _on_resize(self, event): self._update_display()
        
    def _on_tap(self, event):
        if self.zoom_level > 1.0:
            self.zoom_level = 1.0; self._update_display(); return
        width = self.winfo_width()
        if event.x > width * 0.6: self._on_next_page()
        elif event.x < width * 0.4: self._on_prev_page()
            
    def _on_pan_start(self, event): self.canvas.scan_mark(event.x, event.y)
    def _on_pan_move(self, event): self.canvas.scan_dragto(event.x, event.y, gain=1)
        
    def _on_zoom(self, event):
        factor = 1.1 if (event.num == 4 or event.delta > 0) else 0.9 if (event.num == 5 or event.delta < 0) else 0
        if factor:
            self.zoom_level = max(1.0, min(self.zoom_level * factor, 5.0))
            self._update_display()

    def _on_next_page(self): self._load_page(self.current_page_index + 1)
    def _on_prev_page(self): self._load_page(self.current_page_index - 1)
    def _on_next_page_event(self, event): self._on_next_page()
    def _on_prev_page_event(self, event): self._on_prev_page()
    def _close(self, event=None): self.destroy()

class ReadingOrderManagerWindow(tk.Toplevel):
    def __init__(self, app_instance):
        super().__init__(app_instance.root)
        self.app = app_instance
        self.db_file = self.app.db_file
        
        self.title("Gestionar Órdenes de Lectura")
        self.geometry("1000x700")
        self.transient(app_instance.root)
        self.grab_set()

        paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(paned_window, padding=5)
        paned_window.add(left_panel, weight=1)
        
        left_controls = ttk.Frame(left_panel)
        left_controls.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(left_controls, text="Nueva Orden...", bootstyle=SUCCESS, command=self.create_new_order).pack(side=tk.LEFT)
        ttk.Button(left_controls, text="Eliminar Orden", bootstyle=DANGER, command=self.delete_selected_order).pack(side=tk.RIGHT)

        list_frame = ttk.LabelFrame(left_panel, text="Órdenes de Lectura")
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.orders_listbox = tk.Listbox(list_frame, exportselection=False)
        self.orders_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.orders_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.orders_listbox.config(yscrollcommand=scrollbar.set)
        self.orders_listbox.bind("<<ListboxSelect>>", self.on_order_selected)

        right_panel = ttk.Frame(paned_window, padding=5)
        paned_window.add(right_panel, weight=3)
        
        right_controls = ttk.Frame(right_panel)
        right_controls.pack(fill=tk.X, pady=(0, 5))
        
        self.start_reading_btn = ttk.Button(right_controls, text="Empezar a Leer", bootstyle=SUCCESS, state="disabled", command=self.start_reading)
        self.start_reading_btn.pack(side=tk.LEFT)
        self.edit_order_btn = ttk.Button(
            right_controls, 
            text="Editar Orden...", 
            state="disabled", 
            command=self.open_order_editor
        )
        self.edit_order_btn.pack(side=tk.LEFT, padx=5)

        cols = ("#", "Serie", "Número", "Título")
        self.comics_tree = ttk.Treeview(right_panel, columns=cols, show='headings')
        for col in cols: self.comics_tree.heading(col, text=col)
        
        self.comics_tree.column("#", width=40, anchor=CENTER, stretch=False)
        self.comics_tree.column("Serie", width=250)
        self.comics_tree.column("Número", width=80, anchor=CENTER)
        self.comics_tree.column("Título", width=300)
        
        self.comics_tree.pack(fill=tk.BOTH, expand=True)

        self.load_reading_orders()
        self.wait_window()

    def load_reading_orders(self):
        self.orders_listbox.delete(0, tk.END)
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM reading_orders ORDER BY name")
        self.order_data = {name: order_id for order_id, name in cursor.fetchall()}
        conn.close()
        
        for name in sorted(self.order_data.keys()):
            self.orders_listbox.insert(tk.END, name)
            
    def on_order_selected(self, event=None):
        selection = self.orders_listbox.curselection()
        if not selection:
            self.start_reading_btn.config(state="disabled")
            self.edit_order_btn.config(state="disabled")
            for item in self.comics_tree.get_children(): self.comics_tree.delete(item)
            return

        order_name = self.orders_listbox.get(selection[0])
        order_id = self.order_data.get(order_name)
        if not order_id: return

        for item in self.comics_tree.get_children(): self.comics_tree.delete(item)

        conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        cursor.execute("""
            SELECT roi.sequence_number, c.series, c.number, c.title
            FROM reading_order_items roi JOIN comics c ON roi.comic_id = c.id
            WHERE roi.order_id = ? ORDER BY roi.sequence_number
        """, (order_id,))
        comics = cursor.fetchall(); conn.close()
        
        if comics:
            self.start_reading_btn.config(state="normal")
            for comic in comics: self.comics_tree.insert("", tk.END, values=comic)
        else:
            self.start_reading_btn.config(state="disabled")

        self.edit_order_btn.config(state="normal")

    def create_new_order(self):
        name = simpledialog.askstring("Nueva Orden de Lectura", "Nombre para la nueva orden:", parent=self)
        if not name or not name.strip(): return
        name = name.strip()
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO reading_orders (name) VALUES (?)", (name,))
            conn.commit()
            messagebox.showinfo("Éxito", f"Orden de lectura '{name}' creada.", parent=self)
            self.load_reading_orders()
            order_id = cursor.lastrowid
            self.open_order_editor_by_id(order_id, name)
        except sqlite3.IntegrityError:
            messagebox.showwarning("Duplicado", f"Ya existe una orden de lectura con el nombre '{name}'.", parent=self)
        finally: conn.close()

    def delete_selected_order(self):
        selection = self.orders_listbox.curselection()
        if not selection:
            messagebox.showwarning("Sin selección", "Selecciona una orden de lectura para eliminar.", parent=self)
            return
        order_name = self.orders_listbox.get(selection[0])
        if not messagebox.askyesno("Confirmar Eliminación", f"¿Estás seguro de que quieres eliminar la orden '{order_name}'?", parent=self): return
            
        order_id = self.order_data.get(order_name)
        if not order_id: return
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        cursor.execute("DELETE FROM reading_orders WHERE id = ?", (order_id,)); conn.commit(); conn.close()
        for item in self.comics_tree.get_children(): self.comics_tree.delete(item)
        self.load_reading_orders()
        self.on_order_selected()

    def open_order_editor(self):
        selection = self.orders_listbox.curselection()
        if not selection: return
        order_name = self.orders_listbox.get(selection[0])
        order_id = self.order_data.get(order_name)
        self.open_order_editor_by_id(order_id, order_name)

    def open_order_editor_by_id(self, order_id, order_name):
        if order_id:
            editor = ReadingOrderEditorWindow(self, order_id, order_name)
            self.wait_window(editor)
            self.on_order_selected()

    def start_reading(self):
        selection = self.orders_listbox.curselection()
        if not selection: return
        order_name = self.orders_listbox.get(selection[0])
        order_id = self.order_data.get(order_name)
        if order_id:
            self.app.start_reading_order(order_id, start_index=0)

class ReadingOrderEditorWindow(tk.Toplevel):
    def __init__(self, parent, order_id, order_name):
        super().__init__(parent)
        self.placeholder_img = tk.PhotoImage(width=120, height=180)
        self.manager_window = parent
        self.app = parent.app
        self.db_file = self.app.db_file
        self.order_id = order_id
        
        self.title(f"Editando Orden: {order_name}")
        self.geometry("1300x800")
        self.transient(parent)
        self.grab_set()

        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="Nombre:").pack(side=tk.LEFT, padx=(0,5))
        self.name_var = tk.StringVar(value=order_name)
        ttk.Entry(top_frame, textvariable=self.name_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(top_frame, text="Descripción:").pack(side=tk.LEFT, padx=(10,5))
        self.desc_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.desc_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_panel = ttk.LabelFrame(paned_window, text="Cómics en esta Orden", padding=5)
        paned_window.add(left_panel, weight=2)
        
        self.order_comics_map = {}
        
        cols_order = ("#", "Serie", "Número", "Título")
        self.order_tree = ttk.Treeview(left_panel, columns=cols_order, show='headings')
        for col in cols_order: self.order_tree.heading(col, text=col)
        self.order_tree.column("#", width=40, anchor=CENTER, stretch=False)
        self.order_tree.column("Serie", width=200)
        self.order_tree.column("Número", width=60, anchor=CENTER)
        self.order_tree.column("Título", width=250)
        self.order_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        order_scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.order_tree.yview)
        order_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.order_tree.config(yscrollcommand=order_scrollbar.set)

        left_buttons = ttk.Frame(left_panel)
        left_buttons.pack(fill=tk.X, pady=5)
        ttk.Button(left_buttons, text="Subir", command=self.move_up).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(left_buttons, text="Bajar", command=self.move_down).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(left_buttons, text="Quitar", bootstyle=DANGER, command=self.remove_comic).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        center_panel = ttk.Frame(paned_window)
        paned_window.add(center_panel, weight=0)
        ttk.Button(center_panel, text="< Añadir", command=self.add_comic).pack(expand=True, padx=10)

        right_panel = ttk.LabelFrame(paned_window, text="Mi Biblioteca (Selecciona un cómic para añadir)", padding=5)
        paned_window.add(right_panel, weight=3)

        self.library_canvas = tk.Canvas(right_panel, highlightthickness=0)
        library_scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.library_canvas.yview)
        self.library_scrollable_frame = ttk.Frame(self.library_canvas)

        self.library_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.library_canvas.configure(scrollregion=self.library_canvas.bbox("all"))
        )

        self.library_canvas.create_window((0, 0), window=self.library_scrollable_frame, anchor="nw")
        self.library_canvas.configure(yscrollcommand=library_scrollbar.set)
        
        self.library_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        library_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Diccionarios para gestionar las miniaturas
        self.library_thumbnails = {}  # Cache de imágenes ya cargadas
        self.library_widgets = {}     # Widgets de cada miniatura
        self.selected_comic_id_in_lib = None
        self.selected_comic_frame = None
        
        bottom_frame = ttk.Frame(self, padding=10)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(bottom_frame, text="Cancelar", command=self.on_close).pack(side=tk.RIGHT)
        ttk.Button(bottom_frame, text="Guardar Cambios", bootstyle=SUCCESS, command=self.save_changes).pack(side=tk.RIGHT, padx=10)
        self.library_canvas.bind_all("<MouseWheel>", lambda e: self.after(100, self.lazy_load_library_thumbnails))

        self.load_data()
        
    def on_close(self):
        # Desvincula el evento global para evitar que se siga ejecutando
        self.library_canvas.unbind_all("<MouseWheel>")
        self.destroy()    

    def load_data(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Cargar descripción de la orden
        cursor.execute("SELECT description FROM reading_orders WHERE id = ?", (self.order_id,))
        desc = cursor.fetchone()
        if desc and desc[0]:
            self.desc_var.set(desc[0])
        
        # Cargar cómics ya en la orden (panel izquierdo)
        cursor.execute("""
            SELECT roi.comic_id, c.series, c.number, c.title
            FROM reading_order_items roi JOIN comics c ON roi.comic_id = c.id
            WHERE roi.order_id = ? ORDER BY roi.sequence_number
        """, (self.order_id,))
        for comic_id, series, number, title in cursor.fetchall():
            self.add_comic_to_order_view(comic_id, series, number, title)

        # Cargar todos los cómics de la biblioteca (para el panel derecho)
        cursor.execute("SELECT id, series, number, title, series_group FROM comics ORDER BY COALESCE(series_group, 'zzzz'), series, CAST(number AS REAL), number")
        all_comics = cursor.fetchall()
        conn.close()

        self.update_idletasks()  # Asegura que el canvas tenga dimensiones
        THUMB_WIDTH = 120
        container_width = self.library_canvas.winfo_width()
        cols = max(1, container_width // (THUMB_WIDTH + 20)) # +20 para padding

        # Agrupar cómics por 'series_group'
        comics_by_group = {}
        for comic_id, series, number, title, group in all_comics:
            group_name = group or "Sin Grupo"
            if group_name not in comics_by_group:
                comics_by_group[group_name] = []
            comics_by_group[group_name].append((comic_id, series, number, title))

        # Crear los widgets de miniaturas
        current_row = 0
        for group_name in sorted(comics_by_group.keys()):
            # Etiqueta de grupo
            group_label = ttk.Label(self.library_scrollable_frame, text=group_name, font="-weight bold")
            group_label.grid(row=current_row, column=0, columnspan=cols, sticky="ew", pady=(10, 5), padx=5)
            current_row += 1

            group_comics = comics_by_group[group_name]
            for i, comic_data in enumerate(group_comics):
                comic_id, series, number, title = comic_data
                
                row, col = divmod(i, cols)
                
                frame = ttk.Frame(self.library_scrollable_frame, padding=5)
                frame.grid(row=current_row + row, column=col, sticky='nsew')
                
                cover_label = ttk.Label(frame, image=self.placeholder_img, anchor=CENTER)
                cover_label.pack(fill=tk.X)
                
                title_text = f"{series or '?'} #{number or '?'}"
                title_label = ttk.Label(frame, text=title_text, anchor=CENTER, wraplength=THUMB_WIDTH)
                title_label.pack(fill=tk.X, pady=(5,0))
                
                self.library_widgets[comic_id] = {'frame': frame, 'cover': cover_label}
                
                # Asignar evento de click a todos los widgets del frame
                for widget in (frame, cover_label, title_label):
                    widget.bind("<Button-1>", lambda e, c_id=comic_id: self.on_library_comic_selected(c_id))
            
            # Actualizar la fila actual para el siguiente grupo
            current_row += (len(group_comics) + cols - 1) // cols
        
        # Iniciar la carga perezosa de miniaturas después de un breve retraso
        self.after(200, self.lazy_load_library_thumbnails)

    def add_comic_to_order_view(self, comic_id, series, number, title):
        if comic_id in self.order_comics_map.values(): return
        count = len(self.order_tree.get_children()) + 1
        values = (count, series, number, title)
        item_id = self.order_tree.insert("", tk.END, values=values)
        self.order_comics_map[item_id] = comic_id
        
    def add_comic(self):
        selection = self.library_tree.selection()
        if not selection or not self.library_tree.parent(selection[0]):
            messagebox.showwarning("Selección inválida", "Por favor, selecciona un cómic individual, no un grupo de series.", parent=self)
            return
        comic_id, series, number, title = self.library_tree.item(selection[0], 'values')
        self.add_comic_to_order_view(int(comic_id), series, number, title)

    def remove_comic(self):
        selection = self.order_tree.selection()
        if not selection: return
        del self.order_comics_map[selection[0]]
        self.order_tree.delete(selection[0])
        self.update_sequence_numbers()

    def update_sequence_numbers(self):
        for i, item_id in enumerate(self.order_tree.get_children()):
            self.order_tree.item(item_id, values=(i + 1,) + self.order_tree.item(item_id, 'values')[1:])

    def move_up(self):
        selection = self.order_tree.selection()
        if not selection: return
        self.order_tree.move(selection[0], self.order_tree.parent(selection[0]), self.order_tree.index(selection[0]) - 1)
        self.update_sequence_numbers()

    def move_down(self):
        selection = self.order_tree.selection()
        if not selection: return
        self.order_tree.move(selection[0], self.order_tree.parent(selection[0]), self.order_tree.index(selection[0]) + 1)
        self.update_sequence_numbers()

    def save_changes(self):
        new_name = self.name_var.get().strip()
        if not new_name:
            messagebox.showerror("Error", "El nombre de la orden no puede estar vacío.", parent=self)
            return
        
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        try:
            cursor.execute("UPDATE reading_orders SET name = ?, description = ? WHERE id = ?", (new_name, self.desc_var.get().strip(), self.order_id))
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", f"El nombre '{new_name}' ya está en uso por otra orden.", parent=self)
            conn.close(); return
            
        cursor.execute("DELETE FROM reading_order_items WHERE order_id = ?", (self.order_id,))

        comic_sequence = [(self.order_id, self.order_comics_map[item_id], seq + 1) for seq, item_id in enumerate(self.order_tree.get_children())]
        if comic_sequence:
            cursor.executemany("INSERT INTO reading_order_items (order_id, comic_id, sequence_number) VALUES (?, ?, ?)", comic_sequence)
        
        conn.commit(); conn.close()
        
        messagebox.showinfo("Guardado", "La orden de lectura se ha guardado con éxito.", parent=self)
        self.manager_window.load_reading_orders()
        self.destroy()

# ==============================================================================
# 5. CLASE PRINCIPAL DE LA APLICACIÓN
# ==============================================================================

class ToolTip:
    """Clase para mostrar tooltips de ayuda"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=self.text, 
                        background="#ffffe0", relief=tk.SOLID, 
                        borderwidth=1, font=("Segoe UI", 9))
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class GestorApp:
    def __init__(self, root):
        self.root = root
        self.is_running = True
        self.server_running = False
        self.db_file = DB_FILE
        self.author_images_path = Path("author_images")
        
        self.root.title("ANTMAR COMICS COLLECTOR")
        self.root.geometry("1600x900")
        
        # Configurar icono si existe
        icon_path = Path("icono.ico")
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except:
                pass
        
        # Intentar maximizar la ventana, si no funciona, no pasa nada
        try:
            self.root.state('zoomed')
        except tk.TclError:
            print("No se pudo maximizar la ventana (puede ser normal en algunos S.O.)")

        # Configurar estilo moderno
        if MODERN_UI:
            self.style = ttkb.Style()
        else:
            self.style = ttk.Style()
        self.setup_styles()
        
        # Crear contenedor principal con padding
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Título de la aplicación con estilo moderno
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, text="ANTMAR COMICS COLLECTOR" if MODERN_UI else None
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        subtitle_label = ttk.Label(
            header_frame, text="Tu biblioteca digital de cómics" if MODERN_UI else None
        )
        subtitle_label.pack(side=tk.LEFT, padx=10)
        
        self.notebook = ttk.Notebook(main_container)

        # ========== BARRA DE MENÚS ==========
        self.create_menu_bar()
        
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de Biblioteca (principal)
        self.library_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.library_tab, text="📚 MI BIBLIOTECA")
        self.setup_library_tab(self.library_tab)
        
        # Pestaña de Organizador
        self.organizer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.organizer_tab, text="📥 ORGANIZADOR")
        self.setup_organizer_tab(self.organizer_tab)
        
        self.tools_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tools_tab, text="🛠️ HERRAMIENTAS")
        self.setup_tools_tab(self.tools_tab)
        
        # Pestaña de Lector
        self.reader_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reader_tab, text="🔍 LECTOR")
        self.setup_reader_tab(self.reader_tab)
        
        # Barra de estado moderna
        status_frame = ttk.Frame(root, relief=tk.SUNKEN, borderwidth=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="✨ Listo para gestionar tu colección")
        self.status_bar = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            anchor='w',
            bootstyle=INFO if MODERN_UI else None,
            font=('Segoe UI', 9)
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)
        
        self.load_settings()
        self.load_api_keys_from_config()
        # API keys se cargarán cuando se necesiten
        self.setup_database()
        self.refresh_library_view()
        self.load_window_geometry()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Control-f>", lambda event: self.filter_value_entry.focus_set())

    def run(self):
        self.root.mainloop()

    def setup_styles(self):
        """Configura estilos modernos y atractivos para la interfaz"""
        try:
            # Aplicar estilos del módulo moderno si está disponible
            if MODERN_THEME_AVAILABLE:
                apply_modern_styles(self)
            
            # Estilos para frames y contenedores
            self.style.configure('Comic.TLabelframe', borderwidth=2, relief='groove')
            self.style.configure('Comic.TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
            
            # Estilos para títulos modernos con gradientes visuales
            self.style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'))
            self.style.configure('Subtitle.TLabel', font=('Segoe UI', 12))
            self.style.configure('BigTitle.TLabel', font=('Segoe UI', 28, 'bold'))
            
            # Estilos para botones modernos con mejor padding
            self.style.configure('Modern.TButton', font=('Segoe UI', 10, 'bold'), padding=12)
            self.style.configure('BigButton.TButton', font=('Segoe UI', 12, 'bold'), padding=18)
            
            # Frame seleccionado con color primario elegante
            self.style.configure("Selected.TFrame", background=self.style.colors.primary if hasattr(self.style, 'colors') else "#6366f1")
            
            # Estilos para el notebook (pestañas) más espaciosas
            self.style.configure('TNotebook.Tab', font=('Segoe UI', 11, 'bold'), padding=[25, 12])
            
            # Estilos para treeview (listas)
            self.style.configure('Treeview', rowheight=30, font=('Segoe UI', 10))
            self.style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))
            
            print("✨ Estilos personalizados aplicados correctamente")
            
        except Exception as e:
            print(f"⚠️ Warning: No se pudieron aplicar todos los estilos: {e}")
            # Fallback para estilos básicos
            self.style.configure("Selected.TFrame", background="#6366f1")

    def toggle_streaming_server(self):
        global http_server, server_thread
        if not self.server_running:
            try:
                ip_address = get_local_ip()
                port = 5000
                server_thread = ServerThread(flask_app, host='0.0.0.0', port=port)
                server_thread.start()
                self.server_running = True
                self.stream_btn.config(text="Detener Streaming", bootstyle=DANGER)
                messagebox.showinfo("Servidor Iniciado",
                                    f"Servidor de streaming iniciado.\n\n"
                                    f"En el otro ordenador, introduce esta dirección en el lector:\n"
                                    f"http://{ip_address}:{port}",
                                    parent=self.root)
                self.status_var.set(f"Streaming activo en http://{ip_address}:{port}")
            except Exception as e:
                 messagebox.showerror("Error", f"No se pudo iniciar el servidor.\n¿El puerto 5000 está en uso?\n\nError: {e}")
        else:
            if server_thread:
                server_thread.shutdown()
                server_thread.join()
            self.server_running = False
            self.stream_btn.config(text="Iniciar Streaming", bootstyle=SUCCESS)
            self.status_var.set("Servidor de streaming detenido.")

    def on_closing(self):
        if self.server_running:
            self.toggle_streaming_server()
        self.save_window_geometry()
        self.is_running = False
        self.save_settings()
        self.root.destroy()
        
    def setup_tools_tab(self, parent_tab):
        """Configurar pestaña de herramientas - expandida para mejor rendimiento"""
        
        # Inicializar variables
        self.image_files = []
        self.resolution_profiles = {
            "Original": (9999, 9999), 
            "Tablet HD (1920p)": (1200, 1920), 
            "Tablet 2K (2560p)": (1600, 2560), 
            "Personalizado": None
        }
        
        # Frame principal
        main_frame = ttk.Frame(parent_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de controles superior
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Sección de creación
        creation_frame = ttk.LabelFrame(controls_frame, text="Crear CBZ", padding=5)
        creation_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(creation_frame, text="Crear Individual", 
                  command=self.select_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(creation_frame, text="Dividir Tomo Manual", 
                  command=self.start_volume_splitting).pack(side=tk.LEFT, padx=(0, 5))
        
        self.lbl_folder_path = ttk.Label(creation_frame, text="Ninguna carpeta seleccionada")
        self.lbl_folder_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Sección de herramientas
        metadata_frame = ttk.LabelFrame(controls_frame, text="Editar y Herramientas", padding=5)
        metadata_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        self.stream_btn = ttk.Button(metadata_frame, text="Iniciar Streaming",
                                   command=self.toggle_streaming_server)
        self.stream_btn.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Button(metadata_frame, text="Configurar API Keys", 
                  command=self.open_api_keys_window).pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(metadata_frame, text="Editar Metadatos CBZ...", 
                  command=self.open_existing_cbz_for_metadata).pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Button(metadata_frame, text="Re-optimizar CBZ...", 
                  command=self.reoptimize_cbz).pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Button(metadata_frame, text="Descomprimir CBZ...", 
                  command=self.decompress_cbz).pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Button(metadata_frame, text="Traductor de Biblioteca", 
                  command=self.open_batch_translator).pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Button(metadata_frame, text="Renombrador por Lote", 
                  command=self.open_batch_renamer).pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        ttk.Button(metadata_frame, text="Editar en Lote...", 
                  command=self.open_batch_metadata_editor).pack(side=tk.LEFT, padx=(0, 10))
        
        # Área de visualización con previsualización más grande
        viewer_frame = ttk.Frame(main_frame, padding=(0, 5))
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.columnconfigure(1, weight=2)  # Espacio balanceado para la previsualización
        viewer_frame.rowconfigure(0, weight=1)
        
        # Lista de archivos
        list_frame = ttk.Frame(viewer_frame)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, 
                                 bg="white", fg="black", 
                                 selectbackground="blue", selectforeground="white")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        self.listbox.bind('<<ListboxSelect>>', self.show_preview)
        
        # También bind para doble clic por si acaso
        self.listbox.bind("<Double-Button-1>", self.show_preview)
        
        # Previsualización con tamaño balanceado
        preview_frame = tk.Frame(viewer_frame, bg="black")
        preview_frame.grid(row=0, column=1, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        
        self.lbl_preview = tk.Label(preview_frame, text="Previsualización\n\nSelecciona una imagen de la lista\npara ver la previsualización aquí", 
                                   anchor=tk.CENTER, bg="black", fg="white")
        self.lbl_preview.grid(row=0, column=0, sticky="nsew")
        
        # Tamaño balanceado que permita ver las opciones de abajo
        preview_frame.configure(width=400, height=300)
        preview_frame.grid_propagate(False)  # Evitar que se redimensione automáticamente
        
        # Bind para actualizar previsualización cuando cambie el tamaño
        self.lbl_preview.bind("<Configure>", self._on_preview_resize)
        
        # Opciones
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_frame, text="Opciones:", font="-weight bold").pack(anchor='w', pady=(10, 5))
        
        options_grid = ttk.Frame(options_frame)
        options_grid.pack(fill=tk.X)
        options_grid.columnconfigure(0, weight=1)
        options_grid.columnconfigure(1, weight=1)
        
        # Checkboxes de opciones
        self.protect_double_pages_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_grid, text="Proteger páginas dobles", 
                       variable=self.protect_double_pages_var, 
                       bootstyle="round-toggle").grid(row=0, column=0, sticky='w')
        
        self.remove_page_numbers_var = tk.BooleanVar(value=False)
        self.remove_page_numbers_cb = ttk.Checkbutton(options_grid, 
                                                     text="[EXPERIMENTAL] Intentar eliminar números de página", 
                                                     variable=self.remove_page_numbers_var, 
                                                     command=self.check_opencv, 
                                                     bootstyle="round-toggle")
        self.remove_page_numbers_cb.grid(row=1, column=0, sticky='w', columnspan=2, pady=(5, 0))
        
        # Configuración de tamaño
        size_frame = ttk.Frame(options_frame)
        size_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(size_frame, text="Resolución:").pack(side=tk.LEFT)
        
        self.profile_combo = ttk.Combobox(size_frame, values=list(self.resolution_profiles.keys()), state="readonly")
        self.profile_combo.pack(side=tk.LEFT, padx=5)
        self.profile_combo.set("Tablet 2K (2560p)")
        self.profile_combo.bind("<<ComboboxSelected>>", self.update_resolution_fields)
        
        ttk.Label(size_frame, text="Ancho:").pack(side=tk.LEFT, padx=(10, 0))
        self.entry_max_width = ttk.Entry(size_frame, width=8)
        self.entry_max_width.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_frame, text="Alto:").pack(side=tk.LEFT)
        self.entry_max_height = ttk.Entry(size_frame, width=8)
        self.entry_max_height.pack(side=tk.LEFT, padx=5)
        
        self.update_resolution_fields()
        
        # Control de calidad
        quality_frame = ttk.Frame(options_frame)
        quality_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(quality_frame, text="Calidad WebP:").pack(side=tk.LEFT)
        
        self.quality_slider = ttk.Scale(quality_frame, from_=1, to=100, 
                                       orient=tk.HORIZONTAL, length=200)
        self.quality_slider.set(90)
        self.quality_slider.pack(side=tk.LEFT, padx=5)
        
        # Botón de creación
        ttk.Button(main_frame, text="¡Crear CBZ desde Selección!", 
                  bootstyle="success", command=self.start_cbz_creation_thread).pack(fill=tk.X, pady=5)

    # ==============================================================================
    # ORGANIZADOR DE DESCARGAS
    # ==============================================================================
    def setup_organizer_tab(self, parent_tab):
        """Pestaña para organizar cómics descargados automáticamente"""
        self.organizer_files = []
        self.organizer_current_file = None
        self.organizer_current_index = -1
        self.organizer_destination = ""
        
        # Panel principal
        main_container = ttk.Frame(parent_tab, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # ========== BARRA SUPERIOR ==========
        toolbar = ttk.Frame(main_container)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="📂 Seleccionar Carpeta", 
                  command=self.organizer_select_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        self.organizer_status_label = ttk.Label(toolbar, text="Selecciona una carpeta para comenzar", 
                                                 font=('Arial', 10), bootstyle=INFO)
        self.organizer_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ========== ÁREA DE CONTENIDO DIVIDIDA ==========
        content = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)
        
        # ===== LISTA DE ARCHIVOS (Izquierda) =====
        left_panel = ttk.Frame(content)
        content.add(left_panel, weight=1)
        
        ttk.Label(left_panel, text="📚 Archivos", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 5))
        
        list_container = ttk.Frame(left_panel)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.organizer_tree = ttk.Treeview(list_container, columns=('tipo',), show='tree', 
                                           selectmode='browse', height=20)
        self.organizer_tree.column('#0', width=350)
        self.organizer_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tree_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.organizer_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.organizer_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.organizer_tree.bind('<<TreeviewSelect>>', self.organizer_on_file_select)
        
        # ===== PANEL DERECHO: Previsualización y Edición CON SCROLL =====
        right_panel_container = ttk.Frame(content)
        content.add(right_panel_container, weight=2)
        
        # Crear canvas con scroll para todo el panel derecho
        right_canvas = tk.Canvas(right_panel_container, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_panel_container, orient="vertical", command=right_canvas.yview)
        right_panel = ttk.Frame(right_canvas)
        
        right_panel.bind("<Configure>", 
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        
        right_canvas.create_window((0, 0), window=right_panel, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        # Scroll con rueda del ratón
        def _on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        right_canvas.bind("<MouseWheel>", _on_right_mousewheel)
        right_panel.bind("<MouseWheel>", _on_right_mousewheel)
        
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Portada - contenedor flexible para mostrar portadas
        cover_container = ttk.LabelFrame(right_panel, text="🖼️ Portada", padding=10)
        cover_container.pack(fill=tk.X, pady=(0, 10))
        
        # Crear frame interno para controlar mejor el layout
        cover_frame = tk.Frame(cover_container, bg="#2b2b2b")
        cover_frame.pack(fill=tk.BOTH, expand=True)
        
        self.organizer_cover_label = tk.Label(cover_frame, 
                                               text="Selecciona un archivo\npara ver la portada", 
                                               bg="#2b2b2b", fg="#888888", 
                                               compound='center',
                                               justify='center',
                                               wraplength=300,
                                               font=('Arial', 10))
        self.organizer_cover_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.organizer_cover_label.image = None  # Inicializar referencia
        
        # Configurar altura mínima del contenedor
        cover_container.configure(height=200)
        
        # Info rápida
        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.organizer_filename_label = ttk.Label(info_frame, text="", 
                                                   font=('Arial', 10, 'bold'), 
                                                   wraplength=500)
        self.organizer_filename_label.pack(anchor='w')
        
        # Búsqueda de metadatos
        search_frame = ttk.LabelFrame(right_panel, text="🔍 Buscar Metadatos Online", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        search_row = ttk.Frame(search_frame)
        search_row.pack(fill=tk.X)
        
        self.organizer_search_entry = ttk.Entry(search_row, font=('Arial', 10))
        self.organizer_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.organizer_search_entry.bind('<Return>', lambda e: self.organizer_search_comicvine())
        
        ttk.Button(search_row, text="ComicVine", 
                  command=self.organizer_search_comicvine, 
                  bootstyle=WARNING, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(search_row, text="Whakoom", 
                  command=self.organizer_search_whakoom, 
                  bootstyle=SUCCESS, width=12).pack(side=tk.LEFT, padx=2)
        
        # Segunda fila para IA
        ai_row = ttk.Frame(search_frame)
        ai_row.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(ai_row, text="🤖 Generar con IA", 
                  command=self.organizer_generate_ai_metadata, 
                  bootstyle=INFO, width=25).pack(side=tk.LEFT)
        
        ttk.Label(ai_row, text="(Para fan-edits sin metadatos online)", 
                 font=('Arial', 8), foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Contenedor scrollable para metadatos - altura más pequeña
        fields_container = ttk.LabelFrame(right_panel, text="✏️ Metadatos", padding=5)
        fields_container.pack(fill=tk.X, pady=(0, 10))
        
        canvas = tk.Canvas(fields_container, bg="white", highlightthickness=0, height=180)
        scrollbar = ttk.Scrollbar(fields_container, orient=tk.VERTICAL, command=canvas.yview)
        
        self.organizer_fields_frame = tk.Frame(canvas, bg="white")
        
        canvas.create_window((0, 0), window=self.organizer_fields_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.organizer_fields_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Scroll con rueda del ratón en el organizador
        def _on_organizer_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_organizer_mousewheel)
        self.organizer_fields_frame.bind("<MouseWheel>", _on_organizer_mousewheel)
        
        # Crear campos
        self.organizer_field_widgets = {}
        self._create_organizer_fields()
        
        # Acciones finales - SIEMPRE VISIBLE
        actions_frame = ttk.LabelFrame(right_panel, text="💾 Guardar y Organizar", padding=10)
        actions_frame.pack(fill=tk.X)
        
        # Carpeta destino
        dest_row = ttk.Frame(actions_frame)
        dest_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(dest_row, text="Destino:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.organizer_dest_label = ttk.Label(dest_row, text="[Sin seleccionar]", wraplength=300)
        self.organizer_dest_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dest_row, text="📁", command=self.organizer_select_destination, 
                  width=3).pack(side=tk.RIGHT)
        
        # Nombre final
        name_row = ttk.Frame(actions_frame)
        name_row.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(name_row, text="Nombre:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.organizer_final_name_entry = ttk.Entry(name_row, font=('Arial', 9))
        self.organizer_final_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Botones acción
        btn_row = ttk.Frame(actions_frame)
        btn_row.pack(fill=tk.X)
        
        ttk.Button(btn_row, text="✅ Guardar y Siguiente", 
                  command=self.organizer_save_and_next).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(btn_row, text="⏭️ Saltar", 
                  command=self.organizer_skip).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        ttk.Button(btn_row, text="🗑️ Eliminar", 
                  command=self.organizer_delete).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
    
    def _create_organizer_fields(self):
        """Crear campos de entrada de metadatos"""
        fields = [
            'Series', 'Number', 'Title', 'Publisher', 'Year', 'Month', 'Day',
            'Writer', 'Penciller', 'Inker', 'Colorist', 'Letterer', 'CoverArtist', 'Editor',
            'Genre', 'Language', 'Format', 'Web'
        ]
        
        for idx, field in enumerate(fields):
            tk.Label(self.organizer_fields_frame, text=f"{field}:", 
                    bg="white", fg="black", font=('Arial', 9, 'bold'), 
                    anchor='w').grid(row=idx, column=0, sticky='w', padx=5, pady=3)
            
            entry = tk.Entry(self.organizer_fields_frame, width=45, 
                           bg="white", fg="black", insertbackground="black",
                           font=('Arial', 9))
            entry.grid(row=idx, column=1, sticky='ew', padx=5, pady=3)
            
            self.organizer_field_widgets[field] = entry
        
        # Summary más grande
        row = len(fields)
        tk.Label(self.organizer_fields_frame, text="Summary:", 
                bg="white", fg="black", font=('Arial', 9, 'bold'), 
                anchor='nw').grid(row=row, column=0, sticky='nw', padx=5, pady=3)
        
        summary_text = scrolledtext.ScrolledText(self.organizer_fields_frame, 
                                                 wrap=tk.WORD, height=5, width=45,
                                                 bg="white", fg="black", 
                                                 insertbackground="black",
                                                 font=('Arial', 9))
        summary_text.grid(row=row, column=1, sticky='ew', padx=5, pady=3)
        self.organizer_field_widgets['Summary'] = summary_text
        
        # Notes más grande
        row += 1
        tk.Label(self.organizer_fields_frame, text="Notes:", 
                bg="white", fg="black", font=('Arial', 9, 'bold'), 
                anchor='nw').grid(row=row, column=0, sticky='nw', padx=5, pady=3)
        
        notes_text = scrolledtext.ScrolledText(self.organizer_fields_frame, 
                                                 wrap=tk.WORD, height=3, width=45,
                                                 bg="white", fg="black", 
                                                 insertbackground="black",
                                                 font=('Arial', 9))
        notes_text.grid(row=row, column=1, sticky='ew', padx=5, pady=3)
        self.organizer_field_widgets['Notes'] = notes_text
        
        self.organizer_fields_frame.columnconfigure(1, weight=1)
    
    def organizer_select_folder(self):
        """Seleccionar carpeta de descargas"""
        print("🔧 organizer_select_folder llamada")
        folder = filedialog.askdirectory(title="Selecciona la carpeta de descargas")
        print(f"📁 Carpeta seleccionada: {folder}")
        if not folder:
            print("❌ No se seleccionó carpeta")
            return
        
        # Limpiar árbol
        for item in self.organizer_tree.get_children():
            self.organizer_tree.delete(item)
        
        self.organizer_files = []
        
        try:
            print(f"🔍 Buscando en carpeta: {folder}")
            all_files = os.listdir(folder)
            print(f"📂 Total archivos en carpeta: {len(all_files)}")
            
            files = [f for f in all_files if f.lower().endswith(('.cbz', '.cbr'))]
            print(f"📘 Archivos CBZ/CBR encontrados: {len(files)}")
            
            if files:
                print(f"   Primeros 5: {files[:5]}")
            
            if not files:
                self.organizer_status_label.config(text=f"❌ No se encontraron archivos CBZ/CBR en la carpeta")
                messagebox.showinfo("Sin archivos", 
                                  "No hay archivos CBZ o CBR en esta carpeta.", 
                                  parent=self.root)
                return
            
            # Agregar al árbol
            for filename in sorted(files):
                full_path = os.path.join(folder, filename)
                self.organizer_files.append(full_path)
                
                # Icono según extensión
                icon = "📕" if filename.lower().endswith('.cbr') else "📘"
                self.organizer_tree.insert('', 'end', text=f"{icon} {filename}", 
                                          values=(full_path,))
            
            self.organizer_status_label.config(
                text=f"✅ {len(self.organizer_files)} archivos encontrados en: {folder}")
            
            # Seleccionar el primero automáticamente
            if self.organizer_files:
                first_item = self.organizer_tree.get_children()[0]
                self.organizer_tree.selection_set(first_item)
                self.organizer_tree.focus(first_item)
                self.organizer_on_file_select()
                
        except Exception as e:
            self.organizer_status_label.config(text=f"❌ Error al leer la carpeta")
            messagebox.showerror("Error", f"Error al leer la carpeta:\n{e}", parent=self.root)
            traceback.print_exc()
    
    def organizer_on_file_select(self, event=None):
        """Cuando se selecciona un archivo del árbol"""
        print("📋 organizer_on_file_select llamada")
        
        selection = self.organizer_tree.selection()
        if not selection:
            print("❌ No hay selección en el árbol")
            return
        
        # Obtener índice
        item = selection[0]
        all_items = self.organizer_tree.get_children()
        self.organizer_current_index = all_items.index(item)
        self.organizer_current_file = self.organizer_files[self.organizer_current_index]
        
        print(f"📂 Archivo seleccionado: {self.organizer_current_file}")
        print(f"📊 Índice: {self.organizer_current_index}/{len(self.organizer_files)-1}")
        
        # Verificar que el archivo existe
        if not os.path.exists(self.organizer_current_file):
            print(f"❌ El archivo no existe: {self.organizer_current_file}")
            return
        
        # Mostrar nombre
        filename = os.path.basename(self.organizer_current_file)
        self.organizer_filename_label.config(text=f"📄 {filename}")
        print(f"📝 Nombre mostrado: {filename}")
        
        # Limpiar campos
        print("🧹 Limpiando campos...")
        self._organizer_clear_fields()
        
        # Rellenar campo de búsqueda
        search_text = os.path.splitext(filename)[0]
        self.organizer_search_entry.delete(0, tk.END)
        self.organizer_search_entry.insert(0, search_text)
        
        # Sugerir nombre final
        self.organizer_final_name_entry.delete(0, tk.END)
        self.organizer_final_name_entry.insert(0, search_text + '.cbz')
        
        print(f"🚀 Iniciando carga de datos para: {filename}")
        
        # Cargar portada y metadatos en segundo plano
        threading.Thread(target=self._organizer_load_file_data, 
                        args=(self.organizer_current_file,), daemon=True).start()
    
    def _organizer_load_file_data(self, file_path):
        """Cargar portada y metadatos en hilo separado"""
        try:
            print(f"🔍 Cargando datos de: {file_path}")
            
            # Si es CBR, convertir temporalmente a CBZ
            temp_file = None
            working_path = file_path
            
            if file_path.lower().endswith('.cbr'):
                print("📦 Archivo CBR detectado, convirtiendo...")
                try:
                    temp_file = self._convert_cbr_to_cbz(file_path)
                    if temp_file:
                        working_path = temp_file
                        print("✅ Conversión CBR completada")
                    else:
                        print("❌ No se pudo convertir CBR")
                        def show_error():
                            if hasattr(self, 'organizer_cover_label') and self.organizer_cover_label.winfo_exists():
                                self.organizer_cover_label.config(
                                    text="Error: No se pudo convertir CBR", 
                                    fg='white', bg='#2b2b2b', image='')
                                self.organizer_cover_label.image = None
                        try:
                            self.root.after(0, show_error)
                        except RuntimeError:
                            pass  # Ventana cerrada
                        return
                except Exception as e:
                    print(f"❌ Error convirtiendo CBR: {e}")
                    def show_error():
                        if hasattr(self, 'organizer_cover_label') and self.organizer_cover_label.winfo_exists():
                            self.organizer_cover_label.config(
                                text=f"Error CBR: {str(e)[:30]}", 
                                fg='white', bg='#2b2b2b', image='')
                            self.organizer_cover_label.image = None
                    try:
                        self.root.after(0, show_error)
                    except RuntimeError:
                        pass  # Ventana cerrada
                    return
            
            # ===== CARGAR PORTADA =====
            print("🖼️ Extrayendo portada...")
            # Tamaño más apropiado para el organizador
            pil_image = get_cover_from_cbz(working_path, (200, 300))
            
            def update_cover_in_ui():
                """Actualizar la portada en el hilo principal"""
                try:
                    # Verificar que la ventana principal existe
                    if not hasattr(self, 'root') or not self.root.winfo_exists():
                        print("❌ Ventana principal no existe o fue destruida")
                        return
                    
                    # Verificar que el widget existe y es válido
                    if not hasattr(self, 'organizer_cover_label'):
                        print("❌ organizer_cover_label no existe")
                        return
                    
                    if not self.organizer_cover_label.winfo_exists():
                        print("❌ organizer_cover_label fue destruido")
                        return
                    
                    if pil_image:
                        # Convertir PIL a PhotoImage de forma segura
                        try:
                            print(f"📸 Creando PhotoImage desde PIL: {pil_image.size}, modo: {pil_image.mode}")
                            # Usar self.root como master para evitar problemas de timing
                            photo = ImageTk.PhotoImage(pil_image, master=self.root)
                            print(f"📸 PhotoImage creado exitosamente con master=self.root")
                            
                            # Limpiar referencia anterior si existe
                            if hasattr(self.organizer_cover_label, 'image') and self.organizer_cover_label.image:
                                # Liberar imagen anterior
                                old_photo = self.organizer_cover_label.image
                                self.organizer_cover_label.image = None
                                del old_photo
                                print("🧹 Imagen anterior limpiada")
                            
                            # Aplicar nueva imagen al label
                            self.organizer_cover_label.config(
                                image=photo,
                                text="",  # Limpiar texto
                                bg='#2b2b2b',
                                fg='white',
                                compound='center',
                                width=0,  # Permitir que se ajuste al tamaño de la imagen
                                height=0
                            )
                            
                            # CRÍTICO: Mantener múltiples referencias para evitar garbage collection
                            self.organizer_cover_label.image = photo
                            
                            # Backup adicional en el objeto principal
                            if not hasattr(self, 'organizer_cover_cache'):
                                self.organizer_cover_cache = {}
                            self.organizer_cover_cache[self.organizer_current_file] = photo
                            
                            print(f"✅ Portada cargada y mostrada ({pil_image.size}) para {os.path.basename(self.organizer_current_file)}")
                            print(f"🎯 Estado del label: image={self.organizer_cover_label.cget('image')}")
                            
                        except Exception as photo_err:
                            print(f"❌ Error creando PhotoImage: {photo_err}")
                            import traceback
                            traceback.print_exc()
                            
                            # Mostrar imagen de error
                            self.organizer_cover_label.config(
                                text="Error cargando\nimagen",
                                fg='#ff6666',
                                bg='#2b2b2b',
                                image='',
                                compound='center',
                                width=30,
                                height=10
                            )
                            self.organizer_cover_label.image = None
                    else:
                        # Limpiar imagen anterior
                        if hasattr(self.organizer_cover_label, 'image') and self.organizer_cover_label.image:
                            old_photo = self.organizer_cover_label.image
                            self.organizer_cover_label.image = None
                            del old_photo
                        
                        # Mostrar mensaje de sin imágenes
                        self.organizer_cover_label.config(
                            text="Sin imágenes\nen el archivo",
                            fg='#888888',
                            bg='#2b2b2b',
                            image='',
                            compound='center'
                        )
                        self.organizer_cover_label.image = None
                        print("⚠️ No se encontró portada en el archivo")
                
                except Exception as e:
                    print(f"❌ Error actualizando portada en UI: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Mostrar error en la UI
                    try:
                        self.organizer_cover_label.config(
                            text="Error interno",
                            fg='red',
                            bg='#2b2b2b',
                            image=''
                        )
                        self.organizer_cover_label.image = None
                    except:
                        pass  # Si no se puede ni mostrar el error, ignorar
            
            # Actualizar UI de forma segura
            print("📅 Programando actualización de portada...")
            self._organizer_safe_ui_update(update_cover_in_ui)
            
            # ===== CARGAR METADATOS =====
            print("📋 Leyendo metadatos...")
            metadata = read_comicinfo_from_cbz(working_path)
            
            if metadata:
                print(f"✅ Metadatos encontrados: {len(metadata)} campos")
                
                def update_metadata_fields():
                    """Actualizar campos de metadatos en el hilo principal"""
                    try:
                        for key, value in metadata.items():
                            if key in self.organizer_field_widgets and value:
                                widget = self.organizer_field_widgets[key]
                                
                                if isinstance(widget, scrolledtext.ScrolledText):
                                    widget.delete(1.0, tk.END)
                                    widget.insert(1.0, str(value))
                                else:
                                    widget.delete(0, tk.END)
                                    widget.insert(0, str(value))
                        
                        print(f"✅ {len(metadata)} campos actualizados")
                    except Exception as e:
                        print(f"❌ Error actualizando campos: {e}")
                
                # Actualizar metadatos de forma segura
                print("📅 Programando actualización de metadatos...")
                self._organizer_safe_ui_update(update_metadata_fields)
            else:
                print("⚠️ No se encontraron metadatos en el archivo")
            
            # Limpiar archivo temporal si se creó
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print("🗑️ Archivo temporal eliminado")
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            
            def show_error():
                try:
                    if hasattr(self, 'organizer_cover_label') and self.organizer_cover_label.winfo_exists():
                        self.organizer_cover_label.config(
                            text=f"Error: {str(e)[:50]}",
                            fg='red',
                            bg='#2b2b2b',
                            image=''
                        )
                        self.organizer_cover_label.image = None
                except:
                    pass
            
            # Mostrar error de forma segura
            self._organizer_safe_ui_update(show_error)
    
    
    def _organizer_safe_ui_update(self, update_func):
        """Ejecuta una actualización de UI de forma segura desde cualquier hilo"""
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                # Si estamos en el hilo principal, ejecutar directamente
                try:
                    import threading
                    if threading.current_thread() == threading.main_thread():
                        print("🏠 Ejecutando actualización UI en hilo principal")
                        update_func()
                    else:
                        print("⏰ Programando actualización UI desde hilo secundario")
                        self.root.after(0, update_func)
                except Exception as e:
                    print(f"⚠️ Error determinando hilo, usando after: {e}")
                    self.root.after(0, update_func)
            else:
                print("⚠️ Ventana no disponible para actualización UI")
        except Exception as e:
            print(f"❌ Error en _organizer_safe_ui_update: {e}")

    def _organizer_clear_fields(self):
        """Limpiar todos los campos"""
        print("🧹 Limpiando campos del organizador")
        
        # Limpiar campos de texto
        for field, widget in self.organizer_field_widgets.items():
            try:
                if isinstance(widget, scrolledtext.ScrolledText):
                    widget.delete(1.0, tk.END)
                else:
                    widget.delete(0, tk.END)
            except Exception as e:
                print(f"⚠️ Error limpiando campo {field}: {e}")
        
        # Limpiar portada de forma segura
        if hasattr(self, 'organizer_cover_label'):
            try:
                # Liberar imagen anterior si existe
                if hasattr(self.organizer_cover_label, 'image') and self.organizer_cover_label.image:
                    old_photo = self.organizer_cover_label.image
                    self.organizer_cover_label.image = None
                    del old_photo
                    print("🧹 Imagen anterior eliminada")
                
                # Configurar estado de carga
                self.organizer_cover_label.config(
                    image='', 
                    text="Cargando portada...", 
                    bg='#2b2b2b', 
                    fg='white',
                    compound='center',
                    width=30,
                    height=10
                )
                
                print("✅ Portada limpiada, mostrando estado de carga")
            except Exception as e:
                print(f"⚠️ Error limpiando portada: {e}")
                import traceback
                traceback.print_exc()
        
        # Limpiar cache específico del archivo actual si existe
        if hasattr(self, 'organizer_cover_cache') and hasattr(self, 'organizer_current_file'):
            if self.organizer_current_file in self.organizer_cover_cache:
                try:
                    del self.organizer_cover_cache[self.organizer_current_file]
                except:
                    pass
    
    
    def organizer_search_comicvine(self):
        """Buscar en ComicVine y aplicar metadatos al organizador"""
        if not COMICVINE_API_KEY:
            messagebox.showerror("API Key faltante", 
                               "Configura tu API Key de ComicVine primero.\n\n"
                               "Ve a Herramientas → Configurar API Keys", 
                               parent=self.root)
            return
        
        if not self.organizer_current_file:
            messagebox.showwarning("Sin archivo", "Selecciona un archivo primero", parent=self.root)
            return
        
        query = self.organizer_search_entry.get().strip()
        if not query:
            messagebox.showwarning("Búsqueda vacía", "Escribe algo para buscar", parent=self.root)
            return
        
        self.organizer_status_label.config(text="🔍 Buscando en ComicVine...")
        threading.Thread(target=self._organizer_comicvine_thread, args=(query,), daemon=True).start()
    
    def _organizer_comicvine_thread(self, query):
        """Buscar en ComicVine en hilo separado"""
        try:
            # Buscar volúmenes
            params = {
                "api_key": COMICVINE_API_KEY,
                "format": "json",
                "resources": "volume",
                "query": query,
                "limit": 20
            }
            
            response = requests.get("https://comicvine.gamespot.com/api/search", 
                                  params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if not results:
                self.root.after(0, lambda: messagebox.showinfo("Sin resultados", 
                    "No se encontraron resultados en ComicVine", parent=self.root))
                try:
                    self.root.after(0, lambda: self.organizer_status_label.config(text="❌ Sin resultados"))
                except RuntimeError:
                    pass  # Ventana cerrada
                return
            
            # Crear diálogo de selección
            def show_selection():
                dialog = tk.Toplevel(self.root)
                dialog.title("Seleccionar Volumen - ComicVine")
                dialog.geometry("700x500")
                dialog.transient(self.root)
                dialog.grab_set()
                
                main_frame = ttk.Frame(dialog, padding=10)
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                ttk.Label(main_frame, text=f"Se encontraron {len(results)} volúmenes:", 
                         font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))
                
                # Lista de resultados
                listbox_frame = ttk.Frame(main_frame)
                listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
                
                scrollbar = ttk.Scrollbar(listbox_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, 
                                   font=('Arial', 9), height=15)
                listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=listbox.yview)
                
                volume_data = []
                for vol in results:
                    name = vol.get('name', 'Sin nombre')
                    year = vol.get('start_year', 'N/A')
                    publisher = vol.get('publisher', {}).get('name', 'N/A') if vol.get('publisher') else 'N/A'
                    count = vol.get('count_of_issues', 0)
                    
                    display = f"{name} ({year}) - {publisher} - {count} issues"
                    listbox.insert(tk.END, display)
                    volume_data.append(vol)
                
                def on_select():
                    selection = listbox.curselection()
                    if not selection:
                        messagebox.showwarning("Sin selección", "Selecciona un volumen", parent=dialog)
                        return
                    
                    selected_vol = volume_data[selection[0]]
                    volume_id = selected_vol.get('id')
                    
                    # Pedir número de issue
                    issue_num = simpledialog.askstring("Número de Issue", 
                        "Introduce el número del issue:", parent=dialog)
                    
                    if not issue_num:
                        return
                    
                    dialog.destroy()
                    self.organizer_status_label.config(text=f"🔍 Buscando issue #{issue_num}...")
                    threading.Thread(target=self._organizer_fetch_issue, 
                                   args=(volume_id, issue_num), daemon=True).start()
                
                btn_frame = ttk.Frame(main_frame)
                btn_frame.pack(fill=tk.X)
                
                ttk.Button(btn_frame, text="✅ Seleccionar", command=on_select).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
                ttk.Button(btn_frame, text="❌ Cancelar", command=dialog.destroy).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
            
            try:
                self.root.after(0, show_selection)
            except RuntimeError:
                pass  # Ventana cerrada
            try:
                self.root.after(0, lambda: self.status_var.set(f"✅ {len(results)} resultados encontrados"))
            except RuntimeError:
                pass  # Ventana cerrada
            
        except Exception as e:
            print(f"❌ Error buscando en ComicVine: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"Error buscando en ComicVine:\n{e}", parent=self.root))
            try:
                self.root.after(0, lambda: self.status_var.set("❌ Error en búsqueda"))
            except RuntimeError:
                pass  # Ventana cerrada
    
    def _organizer_fetch_issue(self, volume_id, issue_number):
        """Obtener detalles de un issue específico"""
        try:
            params = {
                "api_key": COMICVINE_API_KEY,
                "format": "json",
                "filter": f"volume:{volume_id},issue_number:{issue_number}"
            }
            
            response = requests.get("https://comicvine.gamespot.com/api/issues/", 
                                  params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if not results:
                self.root.after(0, lambda: messagebox.showinfo("Sin resultados", 
                    f"No se encontró el issue #{issue_number}", parent=self.root))
                try:
                    self.root.after(0, lambda: self.status_var.set("❌ Issue no encontrado"))
                except RuntimeError:
                    pass  # Ventana cerrada
                return
            
            issue = results[0]
            issue_id = issue.get('id')
            
            # Obtener detalles completos
            try:
                self.root.after(0, lambda: self.status_var.set("📥 Descargando metadatos..."))
            except RuntimeError:
                pass  # Ventana cerrada
            
            detail_url = f"https://comicvine.gamespot.com/api/issue/{issue_id}/"
            detail_response = requests.get(detail_url, 
                params={"api_key": COMICVINE_API_KEY, "format": "json"}, 
                headers=HEADERS, timeout=REQUEST_TIMEOUT)
            detail_response.raise_for_status()
            issue_data = detail_response.json().get('results', {})
            
            # Procesar créditos
            person_credits = issue_data.get('person_credits', [])
            credits = {
                'Writer': set(), 'Penciller': set(), 'Inker': set(), 
                'Colorist': set(), 'Letterer': set(), 'CoverArtist': set(), 'Editor': set()
            }
            
            role_map = {
                'writer': 'Writer', 'penciller': 'Penciller', 'penciler': 'Penciller',
                'inker': 'Inker', 'colorist': 'Colorist', 'colorer': 'Colorist',
                'letterer': 'Letterer', 'cover': 'CoverArtist', 'cover artist': 'CoverArtist',
                'editor': 'Editor'
            }
            
            for person in person_credits:
                name = person.get('name')
                if not name:
                    continue
                for role in person.get('role', '').lower().split(', '):
                    if role.strip() in role_map:
                        credits[role_map[role.strip()]].add(name)
            
            # Aplicar metadatos al organizador
            def apply_metadata():
                if 'Series' in self.organizer_field_widgets:
                    series = issue_data.get('volume', {}).get('name', '')
                    if series:
                        self.organizer_field_widgets['Series'].delete(0, tk.END)
                        self.organizer_field_widgets['Series'].insert(0, series)
                
                if 'Number' in self.organizer_field_widgets:
                    number = str(issue_data.get('issue_number', ''))
                    if number:
                        self.organizer_field_widgets['Number'].delete(0, tk.END)
                        self.organizer_field_widgets['Number'].insert(0, number)
                
                if 'Title' in self.organizer_field_widgets:
                    title = issue_data.get('name', '')
                    if title:
                        self.organizer_field_widgets['Title'].delete(0, tk.END)
                        self.organizer_field_widgets['Title'].insert(0, title)
                
                if 'Publisher' in self.organizer_field_widgets:
                    publisher = issue_data.get('volume', {}).get('publisher', {}).get('name', '')
                    if publisher:
                        self.organizer_field_widgets['Publisher'].delete(0, tk.END)
                        self.organizer_field_widgets['Publisher'].insert(0, publisher)
                
                # Fecha
                cover_date = issue_data.get('cover_date', '')
                if cover_date:
                    try:
                        parts = cover_date.split('T')[0].split('-')
                        if 'Year' in self.organizer_field_widgets:
                            self.organizer_field_widgets['Year'].delete(0, tk.END)
                            self.organizer_field_widgets['Year'].insert(0, parts[0])
                        if 'Month' in self.organizer_field_widgets:
                            self.organizer_field_widgets['Month'].delete(0, tk.END)
                            self.organizer_field_widgets['Month'].insert(0, parts[1])
                        if 'Day' in self.organizer_field_widgets:
                            self.organizer_field_widgets['Day'].delete(0, tk.END)
                            self.organizer_field_widgets['Day'].insert(0, parts[2])
                    except:
                        pass
                
                # Créditos
                for credit_type in ['Writer', 'Penciller', 'Inker', 'Colorist', 'Letterer', 'CoverArtist', 'Editor']:
                    if credit_type in self.organizer_field_widgets and credits[credit_type]:
                        self.organizer_field_widgets[credit_type].delete(0, tk.END)
                        self.organizer_field_widgets[credit_type].insert(0, ', '.join(sorted(credits[credit_type])))
                
                # Web
                if 'Web' in self.organizer_field_widgets:
                    web = issue_data.get('site_detail_url', '')
                    if web:
                        self.organizer_field_widgets['Web'].delete(0, tk.END)
                        self.organizer_field_widgets['Web'].insert(0, web)
                
                self.status_var.set("✅ Metadatos aplicados desde ComicVine")
                messagebox.showinfo("Éxito", "Metadatos aplicados correctamente", parent=self.root)
            
            try:
                self.root.after(0, apply_metadata)
            except RuntimeError:
                pass  # Ventana cerrada
            
        except Exception as e:
            print(f"❌ Error obteniendo issue: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"Error obteniendo issue:\n{e}", parent=self.root))
            try:
                self.root.after(0, lambda: self.status_var.set("❌ Error obteniendo metadatos"))
            except RuntimeError:
                pass  # Ventana cerrada
    
    
    def organizer_search_whakoom(self):
        """Abrir ventana para pegar URL de Whakoom"""
        if not WHAKOOM_AVAILABLE:
            messagebox.showerror("Módulo no disponible", 
                               "El módulo 'whakoom_scraper.py' no está disponible.\n\n"
                               "Colócalo en la misma carpeta que este programa.", 
                               parent=self.root)
            return
        
        if not self.organizer_current_file:
            messagebox.showwarning("Sin archivo", "Selecciona un archivo primero", parent=self.root)
            return
        
        # Crear ventana emergente simple
        dialog = tk.Toplevel(self.root)
        dialog.title("Buscar en Whakoom")
        dialog.geometry("600x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Pega la URL del cómic de Whakoom:", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))
        
        url_entry = ttk.Entry(main_frame, font=('Arial', 10))
        url_entry.pack(fill=tk.X, pady=(0, 10))
        url_entry.focus()
        
        ttk.Button(main_frame, text="Abrir Whakoom en navegador", 
                  command=lambda: webbrowser.open_new_tab('https://www.whakoom.com'), 
                  bootstyle=INFO).pack(fill=tk.X, pady=(0, 10))
        
        def apply_url():
            url = url_entry.get().strip()
            if not url or "whakoom.com" not in url.lower():
                messagebox.showwarning("URL Inválida", "Pega una URL válida de Whakoom", parent=dialog)
                return
            
            dialog.destroy()
            self.status_var.set("🔍 Obteniendo datos de Whakoom...")
            threading.Thread(target=self._organizer_whakoom_thread, args=(url,), daemon=True).start()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="✅ Aplicar", command=apply_url).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(button_frame, text="❌ Cancelar", command=dialog.destroy).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        url_entry.bind('<Return>', lambda e: apply_url())
    
    def _organizer_whakoom_thread(self, url):
        """Obtener datos de Whakoom y aplicarlos al organizador"""
        try:
            print(f"🔍 Obteniendo datos de Whakoom: {url}")
            details = whakoom_scraper.get_whakoom_details(url)
            
            if not details:
                print("❌ No se obtuvieron datos de Whakoom")
                self.root.after(0, lambda: messagebox.showerror("Error", 
                    "No se pudieron obtener los datos de Whakoom", parent=self.root))
                try:
                    self.root.after(0, lambda: self.organizer_status_label.config(text="❌ Error obteniendo datos"))
                except RuntimeError:
                    pass  # Ventana cerrada
                return
            
            print(f"✅ Datos obtenidos: {list(details.keys())}")
            
            # Mapear campos de Whakoom a los del organizador
            field_mapping = {
                'Series': 'Series',
                'Number': 'Number',
                'Title': 'Title',
                'Publisher': 'Publisher',
                'Year': 'Year',
                'Month': 'Month',
                'Day': 'Day',
                'Summary': 'Summary',
                'Genre': 'Genre',
                'Web': 'Web',
                'Writer': 'Writer',
                'Penciller': 'Penciller',
                'Inker': 'Inker',
                'Colorist': 'Colorist',
                'Letterer': 'Letterer',
                'CoverArtist': 'CoverArtist',
                'Editor': 'Editor',
                'Language': 'LanguageISO',
                'Format': 'Format',
                'Notes': 'Notes',
                'ScanInformation': 'ScanInformation'
            }
            
            def update_fields():
                print("📝 Aplicando datos a los campos del organizador...")
                
                # Aplicar datos
                count = 0
                for whakoom_key, widget_key in field_mapping.items():
                    value = details.get(whakoom_key)
                    if value and widget_key in self.organizer_field_widgets:
                        widget = self.organizer_field_widgets[widget_key]
                        try:
                            if isinstance(widget, scrolledtext.ScrolledText):
                                widget.delete(1.0, tk.END)
                                widget.insert(1.0, str(value))
                            else:
                                widget.delete(0, tk.END)
                                widget.insert(0, str(value))
                            count += 1
                            print(f"   ✓ {widget_key} = {value[:50] if len(str(value)) > 50 else value}")
                        except Exception as e:
                            print(f"   ✗ Error aplicando {widget_key}: {e}")
                
                # Generar nombre automático
                self._organizer_generate_filename()
                
                print(f"✅ {count} campos aplicados correctamente")
                self.organizer_status_label.config(text=f"✅ {count} campos actualizados de Whakoom")
            
            try:
                self.root.after(0, update_fields)
            except RuntimeError:
                pass  # Ventana cerrada
            
        except Exception as e:
            print(f"Error obteniendo datos de Whakoom: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"Error al obtener datos:\n{e}", parent=self.root))
            try:
                self.root.after(0, lambda: self.organizer_status_label.config(text="❌ Error"))
            except RuntimeError:
                pass  # Ventana cerrada
    
    def organizer_generate_ai_metadata(self):
        """Generar metadatos usando IA para fan-edits y cómics sin metadatos online"""
        print("🤖 organizer_generate_ai_metadata llamada")
        
        if not self.organizer_current_file:
            print("❌ No hay archivo seleccionado")
            messagebox.showwarning("Sin archivo", "Selecciona un archivo primero", parent=self.root)
            return
        
        print(f"📁 Archivo actual: {self.organizer_current_file}")
        
        try:
            # Mostrar mensaje de confirmación primero
            result = messagebox.askyesno(
                "Generar metadatos con IA", 
                f"¿Generar metadatos automáticamente para:\n{os.path.basename(self.organizer_current_file)}?\n\n"
                "Se usará configuración automática.",
                parent=self.root
            )
            
            if result:
                print("🚀 Usuario confirmó. Iniciando análisis IA...")
                # Usar configuración por defecto
                config = {
                    'comic_type': 'auto',
                    'preferred_genre': '',
                    'language': 'auto'
                }
                
                self.organizer_status_label.config(text="🤖 Analizando cómic con IA...")
                threading.Thread(target=self._organizer_ai_metadata_thread, args=(config,), daemon=True).start()
            else:
                print("❌ Usuario canceló")
                
        except Exception as e:
            print(f"❌ Error en IA: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error en análisis IA:\n{e}", parent=self.root)
    
    def _organizer_ai_metadata_thread(self, config):
        """Thread para generar metadatos con IA"""
        try:
            print(f"🧵 Thread IA iniciado con config: {config}")
            file_path = self.organizer_current_file
            
            # Analizar el archivo
            print(f"🔍 Analizando archivo: {file_path}")
            analysis_result = self._analyze_comic_with_ai(file_path, config)
            print(f"📊 Resultado análisis: {len(analysis_result) if analysis_result else 0} campos")
            
            if analysis_result:
                def update_fields():
                    # Aplicar los metadatos generados
                    count = 0
                    for field_name, value in analysis_result.items():
                        if field_name in self.organizer_field_widgets and value:
                            widget = self.organizer_field_widgets[field_name]
                            try:
                                if isinstance(widget, scrolledtext.ScrolledText):
                                    widget.delete(1.0, tk.END)
                                    widget.insert(1.0, str(value))
                                else:
                                    widget.delete(0, tk.END)
                                    widget.insert(0, str(value))
                                count += 1
                            except Exception as e:
                                print(f"Error aplicando {field_name}: {e}")
                    
                    # Generar nombre automático
                    self._organizer_generate_filename()
                    
                    self.organizer_status_label.config(text=f"🤖 IA: {count} campos generados")
                
                self.root.after(0, update_fields)
            else:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error de IA", 
                    "No se pudieron generar metadatos con IA. Verifica el archivo.", 
                    parent=self.root
                ))
                self.root.after(0, lambda: self.organizer_status_label.config(text="❌ Error IA"))
                
        except Exception as e:
            print(f"Error en análisis IA: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Error durante análisis IA:\n{e}", parent=self.root
            ))
            self.root.after(0, lambda: self.organizer_status_label.config(text="❌ Error IA"))
    
    def _analyze_comic_with_ai(self, file_path, config):
        """Analizar un cómic usando IA y generar metadatos"""
        try:
            print(f"🤖 Iniciando análisis IA de: {os.path.basename(file_path)}")
            print(f"🔧 Configuración: {config}")
            
            # Extraer información básica del archivo
            basic_info = self._extract_basic_comic_info(file_path)
            
            # Analizar portada si está disponible
            cover_analysis = self._analyze_cover_with_ai(file_path, config)
            
            # Analizar contenido (primeras páginas)
            content_analysis = self._analyze_content_with_ai(file_path, config)
            
            # Combinar resultados
            ai_metadata = self._combine_ai_analysis(basic_info, cover_analysis, content_analysis, config)
            
            return ai_metadata
            
        except Exception as e:
            print(f"Error en análisis IA: {e}")
            return None
    
    def _extract_basic_comic_info(self, file_path):
        """Extraer información básica del nombre del archivo y estructura"""
        try:
            filename = os.path.basename(file_path)
            filename_no_ext = os.path.splitext(filename)[0]
            
            # Intentar detectar patrones en el nombre
            info = {
                'filename': filename_no_ext,
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(filename)[1].lower()
            }
            
            # Patrones comunes
            patterns = {
                'issue_number': r'#?(\d+)',
                'year': r'(19|20)\d{2}',
                'series_name': r'^([^#\d]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, filename_no_ext)
                if match:
                    info[key] = match.group(1).strip()
            
            return info
            
        except Exception as e:
            print(f"Error extrayendo info básica: {e}")
            return {}
    
    def _analyze_cover_with_ai(self, file_path, config):
        """Analizar la portada del cómic usando IA (simulado por ahora)"""
        try:
            # Obtener la portada
            cover_image = self.get_cover_from_cbz(file_path)
            
            if cover_image:
                # Por ahora, simular análisis de IA
                # En una implementación real, aquí se usaría un servicio de IA como:
                # - OpenAI Vision API
                # - Google Vision AI
                # - Azure Computer Vision
                # - Modelo local como BLIP o similares
                
                # Usar género preferido si se especificó
                preferred_genre = config.get('preferred_genre')
                estimated_genre = preferred_genre if preferred_genre else 'superhero'  # Default
                
                analysis = {
                    'has_cover': True,
                    'cover_style': 'modern',  # Placeholder
                    'dominant_colors': ['blue', 'red'],  # Placeholder
                    'estimated_genre': estimated_genre,
                }
                
                print("🎨 Análisis de portada completado (simulado)")
                return analysis
            else:
                return {'has_cover': False}
                
        except Exception as e:
            print(f"Error analizando portada: {e}")
            return {}
    
    def _analyze_content_with_ai(self, file_path, config):
        """Analizar el contenido del cómic usando IA (simulado por ahora)"""
        try:
            # Simular análisis de contenido
            # En una implementación real, se analizarían las primeras páginas
            
            # Usar configuración del usuario
            detected_lang = config.get('language', 'auto')
            if detected_lang == 'auto':
                detected_lang = 'es'  # Default para España
            
            comic_type = config.get('comic_type', 'auto')
            content_type = 'fanedit' if comic_type == 'fanedit' else 'story'
            
            analysis = {
                'estimated_page_count': 22,  # Placeholder
                'content_type': content_type,
                'language_detected': detected_lang,
                'art_style': 'realistic',  # realistic, cartoon, manga, etc.
                'comic_type': comic_type
            }
            
            print("📖 Análisis de contenido completado (simulado)")
            return analysis
            
        except Exception as e:
            print(f"Error analizando contenido: {e}")
            return {}
    
    def _combine_ai_analysis(self, basic_info, cover_analysis, content_analysis, config):
        """Combinar todos los análisis en metadatos finales"""
        try:
            # Generar metadatos inteligentes basados en el análisis
            metadata = {}
            
            # Información de archivo
            if 'series_name' in basic_info:
                metadata['Series'] = basic_info['series_name'].title()
            elif 'filename' in basic_info:
                # Limpiar nombre de archivo como serie
                clean_name = re.sub(r'[#\d\-_\.]', ' ', basic_info['filename'])
                metadata['Series'] = ' '.join(clean_name.split()).title()
            
            if 'issue_number' in basic_info:
                metadata['Number'] = basic_info['issue_number']
            
            if 'year' in basic_info:
                metadata['Year'] = basic_info['year']
            
            # Información de portada
            if cover_analysis.get('estimated_genre'):
                metadata['Genre'] = cover_analysis['estimated_genre'].title()
            
            # Información de contenido
            if content_analysis.get('language_detected'):
                lang_map = {'es': 'Spanish', 'en': 'English', 'fr': 'French'}
                metadata['LanguageISO'] = lang_map.get(content_analysis['language_detected'], 'Unknown')
            
            if content_analysis.get('estimated_page_count'):
                metadata['PageCount'] = str(content_analysis['estimated_page_count'])
            
            # Generar resumen automático basado en tipo de cómic
            genre = metadata.get('Genre', 'Unknown')
            series = metadata.get('Series', 'Comic')
            number = metadata.get('Number', '1')
            comic_type = config.get('comic_type', 'auto')
            
            if comic_type == 'fanedit':
                metadata['Summary'] = f"Fan-edit del cómic {series}. Versión editada por fans con contenido modificado o mejorado."
                metadata['Notes'] = 'Fan-edit - Metadatos generados automáticamente con IA'
            elif comic_type == 'indie':
                metadata['Summary'] = f"{series} - Cómic independiente de género {genre.lower()}. Obra autoeditada."
                metadata['Notes'] = 'Cómic independiente - Metadatos generados automáticamente con IA'
            elif comic_type == 'personal':
                metadata['Summary'] = f"Proyecto personal: {series}. Obra de creación propia."
                metadata['Notes'] = 'Proyecto personal - Metadatos generados automáticamente con IA'
            else:
                metadata['Summary'] = f"Issue #{number} of {series}. A {genre.lower()} comic analyzed with AI."
                metadata['Notes'] = 'Metadatos generados automáticamente con IA'
            
            # Información técnica
            metadata['Format'] = 'cbz' if basic_info.get('format') == '.cbz' else 'cbr'
            metadata['ScanInformation'] = 'Metadata generated automatically with AI analysis'
            
            print(f"🤖 Metadatos IA generados: {len(metadata)} campos")
            return metadata
            
        except Exception as e:
            print(f"Error combinando análisis IA: {e}")
            return {}
    
    def _organizer_display_metadata(self, metadata):
        """Ya no se usa - Los metadatos se aplican directamente desde populate_fields"""
        pass
    
    def _organizer_generate_filename(self, metadata=None):
        """Generar nombre de archivo automáticamente"""
        if not metadata:
            # Leer de los campos
            metadata = {}
            for key, widget in self.organizer_field_widgets.items():
                if isinstance(widget, scrolledtext.ScrolledText):
                    val = widget.get(1.0, tk.END).strip()
                else:
                    val = widget.get().strip()
                if val:
                    metadata[key] = val
        
        parts = []
        
        if metadata.get('Series'):
            parts.append(metadata['Series'])
        
        if metadata.get('Number'):
            parts.append(f"#{metadata['Number'].zfill(3)}")
        
        if metadata.get('Year'):
            parts.append(f"({metadata['Year']})")
        
        if parts:
            filename = ' '.join(parts) + '.cbz'
            filename = re.sub(r'[<>:"/\\|?*]', '', filename)
            self.organizer_final_name_entry.delete(0, tk.END)
            self.organizer_final_name_entry.insert(0, filename)
    
    
    def organizer_select_destination(self):
        """Seleccionar carpeta de destino"""
        folder = filedialog.askdirectory(title="Selecciona la carpeta de destino")
        if folder:
            self.organizer_destination = folder
            short_path = folder if len(folder) < 50 else "..." + folder[-47:]
            self.organizer_dest_label.config(text=short_path, bootstyle=DEFAULT)
    
    def organizer_save_and_next(self):
        """Guardar archivo actual y pasar al siguiente"""
        if not self.organizer_current_file:
            messagebox.showwarning("Sin archivo", "No hay ningún archivo seleccionado", parent=self.root)
            return
        
        if not self.organizer_destination:
            messagebox.showwarning("Sin destino", "Selecciona una carpeta de destino primero", parent=self.root)
            return
        
        final_name = self.organizer_final_name_entry.get().strip()
        if not final_name:
            messagebox.showwarning("Sin nombre", "Escribe un nombre para el archivo", parent=self.root)
            return
        
        if not final_name.endswith('.cbz'):
            final_name += '.cbz'
        
        dest_path = os.path.join(self.organizer_destination, final_name)
        
        if os.path.exists(dest_path):
            if not messagebox.askyesno("Archivo existe", 
                                      f"'{final_name}' ya existe.\n¿Sobrescribir?", 
                                      parent=self.root):
                return
        
        # Guardar en hilo separado
        self.organizer_status_label.config(text=f"💾 Guardando {final_name}...")
        
        # Recopilar metadatos de los campos
        metadata = {}
        for key, widget in self.organizer_field_widgets.items():
            if isinstance(widget, scrolledtext.ScrolledText):
                val = widget.get(1.0, tk.END).strip()
            else:
                val = widget.get().strip()
            if val:
                metadata[key] = val
        
        threading.Thread(target=self._organizer_save_thread, 
                        args=(self.organizer_current_file, dest_path, metadata), 
                        daemon=True).start()
    
    def organizer_skip(self):
        """Saltar al siguiente archivo sin guardar"""
        if not self.organizer_files:
            return
        
        # Eliminar de la lista y pasar al siguiente
        item = self.organizer_tree.selection()[0]
        all_items = self.organizer_tree.get_children()
        current_idx = all_items.index(item)
        
        self.organizer_tree.delete(item)
        self.organizer_files.pop(current_idx)
        
        if self.organizer_files:
            remaining = self.organizer_tree.get_children()
            if remaining:
                next_item = remaining[min(current_idx, len(remaining)-1)]
                self.organizer_tree.selection_set(next_item)
                self.organizer_tree.focus(next_item)
                self.organizer_on_file_select()
            self.organizer_status_label.config(text=f"⏭️ Saltado. {len(self.organizer_files)} archivos restantes")
        else:
            self.organizer_status_label.config(text="🎉 ¡Todos los archivos procesados!")
            self._organizer_clear_fields()
            self.organizer_cover_label.config(image=None, text="")
            self.organizer_filename_label.config(text="")
    
    def organizer_delete(self):
        """Eliminar archivo físicamente"""
        if not self.organizer_current_file:
            return
        
        filename = os.path.basename(self.organizer_current_file)
        response = messagebox.askyesno("⚠️ Eliminar archivo", 
                                      f"¿Eliminar permanentemente '{filename}'?\n\n"
                                      "Esta acción NO se puede deshacer.", 
                                      parent=self.root)
        if not response:
            return
        
        try:
            os.remove(self.organizer_current_file)
            
            # Eliminar de la lista
            item = self.organizer_tree.selection()[0]
            all_items = self.organizer_tree.get_children()
            current_idx = all_items.index(item)
            
            self.organizer_tree.delete(item)
            self.organizer_files.pop(current_idx)
            
            if self.organizer_files:
                remaining = self.organizer_tree.get_children()
                if remaining:
                    next_item = remaining[min(current_idx, len(remaining)-1)]
                    self.organizer_tree.selection_set(next_item)
                    self.organizer_tree.focus(next_item)
                    self.organizer_on_file_select()
                self.organizer_status_label.config(text=f"🗑️ Eliminado. {len(self.organizer_files)} archivos restantes")
            else:
                self.organizer_status_label.config(text="🎉 ¡Todos los archivos procesados!")
                self._organizer_clear_fields()
                self.organizer_cover_label.config(image=None, text="")
                self.organizer_filename_label.config(text="")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar el archivo:\n{e}", parent=self.root)
    
    def _organizer_save_thread(self, source_path, dest_path, metadata):
        """Procesar y guardar archivo en hilo separado"""
        try:
            temp_cbz = source_path
            is_converted = False
            temp_dir_to_cleanup = None
            
            # 1. Si es CBR, convertir a CBZ
            if source_path.lower().endswith('.cbr'):
                try:
                    self.root.after(0, lambda: self.status_var.set("🔄 Convirtiendo CBR a CBZ..."))
                except RuntimeError:
                    pass  # Ventana cerrada
                temp_cbz = self._convert_cbr_to_cbz(source_path)
                if not temp_cbz:
                    raise Exception("No se pudo convertir CBR a CBZ")
                is_converted = True
                temp_dir_to_cleanup = os.path.dirname(temp_cbz)
            
            # 2. Si ya es CBZ pero necesita metadatos, crear copia temporal
            elif metadata and source_path.lower().endswith('.cbz'):
                temp_dir = tempfile.mkdtemp()
                temp_dir_to_cleanup = temp_dir
                temp_cbz = os.path.join(temp_dir, os.path.basename(source_path))
                shutil.copy2(source_path, temp_cbz)
                is_converted = True
            
            # 3. Aplicar metadatos si los hay
            if metadata:
                try:
                    self.root.after(0, lambda: self.status_var.set("📝 Aplicando metadatos..."))
                except RuntimeError:
                    pass  # Ventana cerrada
                
                # Generar XML
                xml_string = generate_comicinfo_xml(metadata)
                
                # Inyectar en el CBZ
                success = inject_xml_into_cbz(temp_cbz, xml_string)
                if not success:
                    print("⚠️ No se pudieron aplicar metadatos (continuando...)")
            
            # 4. Copiar/mover al destino
            try:
                self.root.after(0, lambda: self.status_var.set("📦 Copiando archivo..."))
            except RuntimeError:
                pass  # Ventana cerrada
            
            # Crear directorio destino si no existe
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            if is_converted:
                # Si se modificó, mover el temp
                shutil.move(temp_cbz, dest_path)
                # Limpiar directorio temporal
                if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
                    try:
                        shutil.rmtree(temp_dir_to_cleanup)
                    except Exception as e:
                        print(f"⚠️ No se pudo limpiar directorio temporal: {e}")
            else:
                # Si no se modificó nada, copiar directamente
                shutil.copy2(source_path, dest_path)
            
            # 5. Éxito - Preguntar si eliminar original
            def ask_delete():
                response = messagebox.askyesno("¡Guardado!", 
                    f"✅ Archivo guardado en:\n{dest_path}\n\n¿Eliminar el archivo original de la carpeta de descargas?",
                    parent=self.root)
                if response:
                    try:
                        # Esperar un momento antes de eliminar para asegurar que el archivo no esté bloqueado
                        time.sleep(1.0)
                        
                        # Intentar varias veces si el archivo está bloqueado
                        deleted = False
                        for attempt in range(5):
                            try:
                                os.remove(source_path)
                                deleted = True
                                print(f"🗑️ Archivo original eliminado: {source_path}")
                                break
                            except PermissionError:
                                if attempt < 4:
                                    print(f"⏳ Archivo bloqueado, reintentando ({attempt + 1}/5)...")
                                    time.sleep(0.5)
                                else:
                                    raise
                        
                        if not deleted:
                            raise Exception("No se pudo eliminar el archivo después de 5 intentos")
                        
                        # Buscar y eliminar de la lista
                        try:
                            item = self.organizer_tree.selection()[0]
                            all_items = self.organizer_tree.get_children()
                            index = all_items.index(item)
                            
                            self.organizer_tree.delete(item)
                            self.organizer_files.pop(index)
                            
                            # Seleccionar siguiente si existe
                            if self.organizer_files:
                                remaining = self.organizer_tree.get_children()
                                if remaining:
                                    next_item = remaining[min(index, len(remaining)-1)]
                                    self.organizer_tree.selection_set(next_item)
                                    self.organizer_tree.focus(next_item)
                                    self.organizer_on_file_select()
                            else:
                                messagebox.showinfo("Completado", "¡Todos los archivos procesados!", parent=self.root)
                                self._organizer_clear_fields()
                                self.organizer_cover_label.config(image=None, text="¡Todo listo!")
                        except ValueError:
                            pass
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"No se pudo eliminar el original:\n{e}", parent=self.root)
                else:
                    print(f"ℹ️ Usuario decidió mantener el original: {source_path}")
                
                self.status_var.set(f"✅ Guardado - {len(self.organizer_files)} archivos restantes")
            
            try:
                self.root.after(0, ask_delete)
            except RuntimeError:
                pass  # Ventana cerrada
            
        except Exception as e:
            print(f"Error guardando: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"Error al guardar:\n{e}", parent=self.root))
            try:
                self.root.after(0, lambda: self.status_var.set("❌ Error al guardar"))
            except RuntimeError:
                pass  # Ventana cerrada
    
    def _convert_cbr_to_cbz(self, cbr_path):
        """Convertir CBR a CBZ temporal"""
        try:
            import rarfile
            
            # Configurar UnRAR.exe si está disponible
            unrar_path = os.path.join(os.path.dirname(__file__), 'UnRAR.exe')
            if os.path.exists(unrar_path):
                rarfile.UNRAR_TOOL = unrar_path
            
            temp_dir = tempfile.mkdtemp()
            extract_dir = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_dir, exist_ok=True)
            temp_cbz = os.path.join(temp_dir, os.path.splitext(os.path.basename(cbr_path))[0] + '.cbz')
            
            print(f"🔄 Convirtiendo CBR a CBZ: {cbr_path}")
            
            # Extraer CBR
            with rarfile.RarFile(cbr_path) as rf:
                rf.extractall(extract_dir)
            
            # Crear CBZ
            with zipfile.ZipFile(temp_cbz, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(extract_dir):
                    for file in sorted(files):  # Ordenar archivos
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, extract_dir)
                            zf.write(file_path, arcname)
                            print(f"  ✅ Añadido: {arcname}")
            
            print(f"✅ CBZ creado: {temp_cbz}")
            return temp_cbz
            
        except ImportError:
            self.root.after(0, lambda: messagebox.showerror("Módulo faltante", 
                "Para convertir CBR necesitas instalar:\n\npip install rarfile", parent=self.root))
            return None
        except Exception as e:
            print(f"❌ Error convirtiendo CBR: {e}")
            traceback.print_exc()
            return None

    def create_menu_bar(self):
        """Crea la barra de menús de la aplicación"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📁 Archivo", menu=file_menu)
        file_menu.add_command(label="Escanear Carpeta (Completo)", 
                             command=lambda: self.scan_library_folder('full'))
        file_menu.add_command(label="Sincronizar Biblioteca", 
                             command=lambda: self.scan_library_folder('sync'))
        file_menu.add_separator()
        file_menu.add_command(label="Configurar API Keys", 
                             command=self.open_api_keys_window)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.on_closing)
        
        # Menú Ver
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="👁️ Ver", menu=view_menu)
        view_menu.add_command(label="Cambiar Vista (Lista ↔ Miniaturas)", 
                             command=self._toggle_library_view)
        view_menu.add_separator()
        view_menu.add_command(label="Actualizar Biblioteca", 
                             command=self.refresh_library_view)
        
        # Menú Herramientas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🛠️ Herramientas", menu=tools_menu)
        tools_menu.add_command(label="Gestión de Autores", 
                              command=self.open_author_manager)
        tools_menu.add_command(label="Órdenes de Lectura", 
                              command=self.open_reading_order_manager)
        tools_menu.add_command(label="Editor por Lotes", 
                              command=self.open_batch_metadata_editor)
        tools_menu.add_separator()
        tools_menu.add_command(label="Servidor de Streaming", 
                              command=self.toggle_streaming_server)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="❓ Ayuda", menu=help_menu)
        help_menu.add_command(label="Acerca de", command=self.show_about)
    
    def show_about(self):
        """Muestra la ventana Acerca de"""
        about_text = """
ANTMAR COMICS COLLECTOR
Versión 2.0

Tu biblioteca digital de cómics profesional

© 2025 ANTMAR
Desarrollado con ❤️ para los amantes del cómic
        """
        messagebox.showinfo("Acerca de ANTMAR COMICS COLLECTOR", 
                          about_text, parent=self.root)
    
    def setup_library_tab(self, parent_tab):
        self.library_data = []; self.thumbnail_cache = {}; self.thumbnail_widgets = {}; self.selected_comic_path = None; self.library_view_mode = tk.StringVar(value="list"); self.placeholder_image = tk.PhotoImage(width=150, height=225)
        # Stack view state
        self.stack_view_var = tk.BooleanVar(value=False)
        self.stack_navigation = []  # Stack of saved states [(library_data, group_name), ...]
        
        paned_window = ttk.PanedWindow(parent_tab, orient=tk.HORIZONTAL); paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.library_frame = ttk.LabelFrame(paned_window, text="Biblioteca", style='Comic.TLabelframe'); self.library_frame.rowconfigure(2, weight=1); self.library_frame.columnconfigure(0, weight=1); paned_window.add(self.library_frame, weight=2)
        
        top_controls_frame = ttk.Frame(self.library_frame); top_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.top_controls_frame = top_controls_frame  # Save reference for back button
        scan_menu_button = ttk.Menubutton(top_controls_frame, text="Escanear...")
        scan_menu_button.pack(side=tk.LEFT, padx=(0, 5))
        scan_menu = tk.Menu(scan_menu_button, tearoff=False)
        scan_menu.add_command(
            label="Sincronizar Biblioteca Completa (Limpia obsoletos)",
            command=lambda: self.scan_library_folder(sync_mode='full')
        )
        scan_menu.add_command(
            label="Añadir/Actualizar Carpeta (No borra nada)",
            command=lambda: self.scan_library_folder(sync_mode='additive')
        )
        scan_menu_button["menu"] = scan_menu
        self.scan_btn_ref = scan_menu_button

        ttk.Button(top_controls_frame, text="Gestionar Autores...", command=self.open_author_manager).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(top_controls_frame, text="Órdenes de Lectura...", command=self.open_reading_order_manager).pack(side=tk.LEFT, padx=(0,10))
        self.view_toggle_btn = ttk.Checkbutton(top_controls_frame, text="Vista de Miniaturas", command=self._toggle_library_view, bootstyle="round-toggle"); self.view_toggle_btn.pack(side=tk.RIGHT)
        self.stack_view_btn = ttk.Checkbutton(top_controls_frame, text="Vista de Pilas", variable=self.stack_view_var, command=self._on_stack_view_toggle, bootstyle="round-toggle"); self.stack_view_btn.pack(side=tk.RIGHT, padx=(0, 5))

        filter_frame = ttk.LabelFrame(self.library_frame, text="Filtros y Agrupación", padding=5); filter_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        filter_frame.columnconfigure(3, weight=1)
        filter_frame.columnconfigure(4, weight=2)
        
        ttk.Label(filter_frame, text="Agrupar por:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.grouping_options = { 'Grupo (defecto)': 'series_group', 'Serie': 'series', 'Editorial': 'publisher', 'Año': 'year', 'Guionista': 'writer' }
        self.group_by_combo = ttk.Combobox(filter_frame, values=list(self.grouping_options.keys()), state="readonly", width=15)
        self.group_by_combo.grid(row=0, column=1, padx=(0, 10), sticky="ew")
        self.group_by_combo.set(list(self.grouping_options.keys())[0])
        self.group_by_combo.bind("<<ComboboxSelected>>", self.refresh_library_view)
        
        ttk.Label(filter_frame, text="Filtrar por:").grid(row=0, column=2, padx=(10, 5), sticky="w")
        self.filtering_options = {
            'Cualquier Campo': 'all', 'Serie': 'series', 'Título': 'title', 'Guionista': 'writer',
            'Dibujante': 'penciller', 'Editorial': 'publisher', 'Arco Argumental': 'storyarc',
            'Personajes': 'characters', 'Equipos': 'teams'
        }
        self.filter_field_combo = ttk.Combobox(filter_frame, values=list(self.filtering_options.keys()), state="readonly")
        self.filter_field_combo.grid(row=0, column=3, padx=(0, 5), sticky="ew")
        self.filter_field_combo.set(list(self.filtering_options.keys())[0])
        self.filter_value_entry = ttk.Entry(filter_frame); self.filter_value_entry.grid(row=0, column=4, padx=(0, 5), sticky="ew")
        self.filter_value_entry.bind("<KeyRelease>", self.refresh_library_view)
        ttk.Button(filter_frame, text="Limpiar", command=self.clear_filter).grid(row=0, column=5, sticky="e")
        
        self.list_view_frame = ttk.Frame(self.library_frame); self.list_view_frame.grid(row=2, column=0, sticky="nsew")
        cols = ("path", "Serie", "Número", "Título", "Año"); display_cols = ("Serie", "Número", "Título", "Año")
        self.library_tree = ttk.Treeview(self.list_view_frame, columns=cols, displaycolumns=display_cols, show='tree headings');
        for col in display_cols: self.library_tree.heading(col, text=col)
        self.library_tree.column("#0", width=200); self.library_tree.column("Serie", width=250); self.library_tree.column("Número", width=80, anchor=CENTER); self.library_tree.column("Título", width=300); self.library_tree.column("Año", width=80, anchor=CENTER)
        self.library_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar = ttk.Scrollbar(self.list_view_frame, orient=tk.VERTICAL, command=self.library_tree.yview); list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.library_tree.configure(yscrollcommand=list_scrollbar.set); self.library_tree.bind('<<TreeviewSelect>>', self.on_list_item_selected)
        self.library_tree.bind('<Double-1>', self.read_selected_comic_from_event)
        
        self.thumb_view_frame = ttk.Frame(self.library_frame); self.thumb_canvas = tk.Canvas(self.thumb_view_frame, highlightthickness=0); thumb_scrollbar = ttk.Scrollbar(self.thumb_view_frame, orient=tk.VERTICAL, command=self.thumb_canvas.yview)
        self.thumb_scrollable_frame = ttk.Frame(self.thumb_canvas); self.thumb_scrollable_frame.bind("<Configure>", lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))); self.thumb_canvas.create_window((0, 0), window=self.thumb_scrollable_frame, anchor="nw"); self.thumb_canvas.configure(yscrollcommand=thumb_scrollbar.set)
        self.thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); thumb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.thumb_canvas.bind('<Configure>', self._repopulate_thumbnail_view); self.thumb_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.thumb_canvas.bind("<Destroy>", lambda e: self.thumb_canvas.unbind_all("<MouseWheel>")); thumb_scrollbar.bind("<B1-Motion>", lambda e: self.root.after(100, self._lazy_load_thumbnails))
        
        right_panel = ttk.LabelFrame(paned_window, text="Detalles", style='Comic.TLabelframe'); paned_window.add(right_panel, weight=3); right_panel.rowconfigure(0, weight=1); right_panel.columnconfigure(0, weight=1)
        details_canvas = tk.Canvas(right_panel, highlightthickness=0); details_scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=details_canvas.yview)
        self.details_scrollable_frame = ttk.Frame(details_canvas, padding=10); self.details_scrollable_frame.bind("<Configure>", lambda e: details_canvas.configure(scrollregion=details_canvas.bbox("all"))); details_canvas.create_window((0, 0), window=self.details_scrollable_frame, anchor="nw"); details_canvas.configure(yscrollcommand=details_scrollbar.set)
        details_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        container = self.details_scrollable_frame; container.columnconfigure(0, weight=1)
        
        # Frame contenedor para la portada con tamaño fijo
        cover_container = tk.Frame(container, bg="#1a1a1a", width=350, height=525)
        cover_container.grid(row=0, column=0, pady=(0, 10))
        cover_container.grid_propagate(False)  # Mantener tamaño fijo
        
        self.detail_cover_label = tk.Label(cover_container, 
                                           bg="#1a1a1a", 
                                           fg="#888888",
                                           text="Selecciona un cómic",
                                           compound='center',
                                           anchor=CENTER)
        self.detail_cover_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detail_title_label = ttk.Label(container, text="Selecciona un cómic", style='Title.TLabel', anchor=CENTER, wraplength=550); self.detail_title_label.grid(row=1, column=0, sticky="ew")
        self.detail_series_label = ttk.Label(container, text="", font="-weight bold", anchor=CENTER); self.detail_series_label.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        
        def create_detail_row(parent, row_index, label_text):
            frame = ttk.Frame(parent)
            frame.grid(row=row_index, column=0, sticky="ew", pady=1)
            frame.columnconfigure(1, weight=1)
            ttk.Label(frame, text=f"{label_text}:", font="-weight bold", width=15).grid(row=0, column=0, sticky="nw")
            value_frame = ttk.Frame(frame)
            value_frame.grid(row=0, column=1, sticky="ew", padx=5)
            return value_frame

        r = 3
        self.detail_publisher_frame = create_detail_row(container, r, "Editorial")
        self.detail_date_label_frame = create_detail_row(container, r + 1, "Fecha Pub.")
        
        ttk.Separator(container, orient='horizontal').grid(row=r+2, column=0, sticky='ew', pady=10)
        self.writer_frame = create_detail_row(container, r+3, "Guion"); self.penciller_frame = create_detail_row(container, r+4, "Dibujo")
        self.inker_frame = create_detail_row(container, r+5, "Tinta"); self.colorist_frame = create_detail_row(container, r+6, "Color"); self.coverartist_frame = create_detail_row(container, r+7, "Portada")
        ttk.Separator(container, orient='horizontal').grid(row=r+8, column=0, sticky='ew', pady=10)

        self.detail_storyarc_label = ttk.Label(create_detail_row(container, r+9, "Arco"), text="N/A", wraplength=450); self.detail_storyarc_label.pack(anchor="w")
        self.detail_characters_label = ttk.Label(create_detail_row(container, r+10, "Personajes"), text="N/A", wraplength=450); self.detail_characters_label.pack(anchor="w")
        self.detail_teams_label = ttk.Label(create_detail_row(container, r+11, "Equipos"), text="N/A", wraplength=450); self.detail_teams_label.pack(anchor="w")
        self.detail_web_label = ttk.Label(create_detail_row(container, r+12, "Web"), text="N/A", wraplength=450); self.detail_web_label.pack(anchor="w")

        ttk.Separator(container, orient='horizontal').grid(row=r+13, column=0, sticky='ew', pady=10); ttk.Label(container, text="Resumen:", font="-weight bold").grid(row=r+14, column=0, sticky="nw", pady=(5,2))
        self.detail_summary_text = tk.Text(container, wrap=tk.WORD, height=10, state="disabled", relief="flat", highlightthickness=0); self.detail_summary_text.grid(row=r+15, column=0, sticky="nsew")
        
        button_frame = ttk.Frame(container); button_frame.grid(row=r+16, column=0, sticky="ew", pady=(15,0))
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1); button_frame.columnconfigure(2, weight=1)
        self.detail_read_btn = ttk.Button(button_frame, text="Leer Cómic", state="disabled", command=self.read_selected_comic); self.detail_read_btn.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self.detail_edit_btn = ttk.Button(button_frame, text="Editar Metadatos", state="disabled", command=self.edit_selected_comic); self.detail_edit_btn.grid(row=0, column=1, sticky="ew", padx=(5,5))
        ToolTip(self.detail_read_btn, "Modifica los metadatos del cómic seleccionado")
        self.detail_open_folder_btn = ttk.Button(button_frame, text="Abrir Ubicación", state="disabled", command=self.open_selected_comic_location); self.detail_open_folder_btn.grid(row=0, column=2, sticky="ew", padx=(5,0))

        self.thumb_view_frame.grid_remove()

    def save_window_geometry(self):
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
        if 'Window' not in config:
            config['Window'] = {}
        
        if self.root.state() == 'zoomed':
            config['Window']['state'] = 'zoomed'
        else:
            config['Window']['state'] = 'normal'
            config['Window']['geometry'] = self.root.geometry()

        with open('config.ini', 'w') as configfile:
            config.write(configfile)

    def load_window_geometry(self):
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
            if 'Window' in config:
                if config['Window'].get('state') == 'zoomed':
                    try:
                        self.root.state('zoomed')
                    except tk.TclError:
                        pass
                elif config['Window'].get('geometry'):
                    self.root.geometry(config['Window']['geometry'])
        
    def read_selected_comic_from_event(self, event=None):
        self.read_selected_comic()    
       
    def add_or_update_comic_in_db(self, cbz_path, metadata):
        if not metadata:
            messagebox.showerror("Error", "No se proporcionaron metadatos válidos.", parent=self.root)
            return

        conn = None
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")

            cursor.execute("SELECT id FROM comics WHERE path = ?", (cbz_path,))
            existing_comic = cursor.fetchone()
            
            action = "actualizado" if existing_comic else "añadido"
            if existing_comic:
                if not messagebox.askyesno("Cómic Existente", f"'{os.path.basename(cbz_path)}' ya está en tu biblioteca.\n\n¿Quieres SOBREESCRIBIR sus datos?", parent=self.root):
                    conn.rollback()
                    conn.close()
                    self.status_var.set("Operación cancelada.")
                    return

            data_tuple = (
                metadata.get('Tags'), metadata.get('Series'), metadata.get('Number'), metadata.get('Title'),
                metadata.get('Publisher'), metadata.get('Year'), metadata.get('Month'), metadata.get('Day'),
                metadata.get('Writer'), metadata.get('Penciller'), metadata.get('Inker'), metadata.get('Colorist'),
                metadata.get('Letterer'), metadata.get('CoverArtist'), metadata.get('Editor'), metadata.get('Summary'),
                metadata.get('StoryArc'), metadata.get('Characters'), metadata.get('Teams'), metadata.get('Web'),
                cbz_path
            )
            
            comic_id = -1
            if action == "actualizado":
                comic_id = existing_comic[0]
                update_tuple = data_tuple[:-1] + (cbz_path,)
                cursor.execute("UPDATE comics SET series_group=?, series=?, number=?, title=?, publisher=?, year=?, month=?, day=?, writer=?, penciller=?, inker=?, colorist=?, letterer=?, coverartist=?, editor=?, summary=?, storyarc=?, characters=?, teams=?, web=? WHERE path=?", update_tuple)
            elif action == "añadido":
                cursor.execute("INSERT INTO comics (series_group, series, number, title, publisher, year, month, day, writer, penciller, inker, colorist, letterer, coverartist, editor, summary, storyarc, characters, teams, web, path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_tuple)
                comic_id = cursor.lastrowid

            if comic_id != -1:
                cursor.execute("DELETE FROM comic_authors WHERE comic_id = ?", (comic_id,))
                
                roles = ['Writer', 'Penciller', 'Inker', 'Colorist', 'Letterer', 'CoverArtist', 'Editor']
                autores_a_insertar = []
                for role in roles:
                    authors_str = metadata.get(role)
                    if not authors_str: continue
                    for name in [name.strip() for name in re.split('[,;]', authors_str) if name.strip()]:
                        autores_a_insertar.append((name,))

                if autores_a_insertar:
                    cursor.executemany("INSERT OR IGNORE INTO authors (name) VALUES (?)", autores_a_insertar)
                    
                    for role in roles:
                        authors_str = metadata.get(role)
                        if not authors_str: continue
                        for name in [name.strip() for name in re.split('[,;]', authors_str) if name.strip()]:
                            cursor.execute("SELECT id FROM authors WHERE name = ?", (name,))
                            author_id_res = cursor.fetchone()
                            if author_id_res:
                                cursor.execute("INSERT OR IGNORE INTO comic_authors (comic_id, author_id, role) VALUES (?, ?, ?)", (comic_id, author_id_res[0], role))
            
            conn.commit()
            messagebox.showinfo("Biblioteca Actualizada", f"¡El cómic se ha {action} a la biblioteca con éxito!", parent=self.root)
            self.status_var.set(f"Biblioteca actualizada para '{os.path.basename(cbz_path)}'.")
            self.refresh_library_view()

        except Exception as e:
            if conn:
                conn.rollback()
            messagebox.showerror("Error de Base de Datos", f"Ocurrió un error al actualizar la biblioteca:\n{e}")
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

    def add_or_update_comic_in_db_batch(self, cbz_path, metadata):
        if not metadata or not cbz_path: return
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("SELECT id FROM comics WHERE path = ?", (cbz_path,))
            existing_comic = cursor.fetchone()
            action = "actualizado" if existing_comic else "añadido"
            data_tuple = (
                metadata.get('Tags'), metadata.get('Series'), metadata.get('Number'), metadata.get('Title'),
                metadata.get('Publisher'), metadata.get('Year'), metadata.get('Month'), metadata.get('Day'),
                metadata.get('Writer'), metadata.get('Penciller'), metadata.get('Inker'), metadata.get('Colorist'),
                metadata.get('Letterer'), metadata.get('CoverArtist'), metadata.get('Editor'), metadata.get('Summary'),
                metadata.get('StoryArc'), metadata.get('Characters'), metadata.get('Teams'), metadata.get('Web'),
                cbz_path
            )
            comic_id = -1
            if action == "actualizado":
                comic_id = existing_comic[0]
                update_tuple = data_tuple[:-1] + (cbz_path,)
                cursor.execute("UPDATE comics SET series_group=?, series=?, number=?, title=?, publisher=?, year=?, month=?, day=?, writer=?, penciller=?, inker=?, colorist=?, letterer=?, coverartist=?, editor=?, summary=?, storyarc=?, characters=?, teams=?, web=? WHERE path=?", update_tuple)
            elif action == "añadido":
                cursor.execute("INSERT INTO comics (series_group, series, number, title, publisher, year, month, day, writer, penciller, inker, colorist, letterer, coverartist, editor, summary, storyarc, characters, teams, web, path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_tuple)
                comic_id = cursor.lastrowid
            if comic_id != -1:
                cursor.execute("DELETE FROM comic_authors WHERE comic_id = ?", (comic_id,))
                roles = ['Writer', 'Penciller', 'Inker', 'Colorist', 'Letterer', 'CoverArtist', 'Editor']
                autores_a_insertar = []
                for role in roles:
                    authors_str = metadata.get(role)
                    if not authors_str: continue
                    for name in [name.strip() for name in re.split('[,;]', authors_str) if name.strip()]:
                        autores_a_insertar.append((name,))
                if autores_a_insertar:
                    cursor.executemany("INSERT OR IGNORE INTO authors (name) VALUES (?)", autores_a_insertar)
                    for role in roles:
                        authors_str = metadata.get(role)
                        if not authors_str: continue
                        for name in [name.strip() for name in re.split('[,;]', authors_str) if name.strip()]:
                            cursor.execute("SELECT id FROM authors WHERE name = ?", (name,))
                            author_id_res = cursor.fetchone()
                            if author_id_res:
                                cursor.execute("INSERT OR IGNORE INTO comic_authors (comic_id, author_id, role) VALUES (?, ?, ?)", (comic_id, author_id_res[0], role))
            conn.commit()
        except Exception as e:
            if conn: conn.rollback()
            print(f"Error en actualización por lote para {cbz_path}: {e}")
            traceback.print_exc()
        finally:
            if conn: conn.close()

    def load_api_keys_from_config(self):
        global COMICVINE_API_KEY, DEEPL_API_KEY
        config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            config.read('config.ini')
            if 'APIKeys' in config:
                COMICVINE_API_KEY = config['APIKeys'].get('comicvine_api_key', '')
                DEEPL_API_KEY = config['APIKeys'].get('deepl_api_key', '')
                if COMICVINE_API_KEY or DEEPL_API_KEY:
                    cv_status = "✓" if COMICVINE_API_KEY else "✗"
                    deepl_status = "✓" if DEEPL_API_KEY else "✗"
                    print(f"🔑 API Keys cargadas - ComicVine: {cv_status}, DeepL: {deepl_status}")

        # NO solicitar automáticamente - el usuario lo hará si lo necesita
    
    def open_api_keys_window(self): ApiKeysWindow(self.root)
    def open_author_manager(self): AuthorManagementWindow(self)
    def open_reading_order_manager(self): ReadingOrderManagerWindow(self)
    def open_batch_metadata_editor(self): BatchMetadataEditorWindow(self.root, self)

    def read_selected_comic(self):
        path = self.get_selected_comic_path()
        if path and os.path.exists(path):
            ComicReaderWindow(self.root, path)
        elif path:
            messagebox.showerror("Error", f"El archivo no se encuentra en la ruta:\n{path}", parent=self.root)

    def open_external_cbz_editor(self):
        """Abre editor para un CBZ externo (no en biblioteca)"""
        cbz_path = filedialog.askopenfilename(
            title="Selecciona un archivo CBZ",
            filetypes=[("Archivos CBZ", "*.cbz"), ("Todos los archivos", "*.*")],
            parent=self.root
        )
        if cbz_path:
            print(f"🔍 Abriendo editor para: {cbz_path}")
            MetadataEditorWindow(self.root, cbz_path=cbz_path)
    
    def create_individual_cbz(self):
        """Crea un CBZ desde una carpeta de imágenes"""
        folder = filedialog.askdirectory(
            title="Selecciona carpeta con imágenes",
            parent=self.root
        )
        if not folder:
            return
        
        # Buscar imágenes
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
        
        if not images:
            messagebox.showwarning("Sin imágenes", "No se encontraron imágenes en la carpeta seleccionada", parent=self.root)
            return
        
        images.sort()
        
        # Solicitar nombre del CBZ
        output_name = simpledialog.askstring(
            "Nombre del CBZ",
            "Introduce el nombre del archivo (sin extensión):",
            parent=self.root
        )
        
        if not output_name:
            return
        
        # Solicitar ubicación de guardado
        output_path = filedialog.asksaveasfilename(
            title="Guardar CBZ como",
            defaultextension=".cbz",
            initialfile=f"{output_name}.cbz",
            filetypes=[("Archivos CBZ", "*.cbz")],
            parent=self.root
        )
        
        if not output_path:
            return
        
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                for i, img in enumerate(images, 1):
                    img_path = os.path.join(folder, img)
                    # Renombrar a formato page_001.ext
                    ext = os.path.splitext(img)[1]
                    arcname = f"page_{i:03d}{ext}"
                    cbz.write(img_path, arcname)
                    print(f"  Agregando {arcname}...")
            
            messagebox.showinfo("Éxito", f"CBZ creado exitosamente:\n{output_path}", parent=self.root)
            
            # Preguntar si quiere editar metadatos
            if messagebox.askyesno("Editar metadatos", "¿Deseas agregar metadatos al CBZ creado?", parent=self.root):
                MetadataEditorWindow(self.root, cbz_path=output_path)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear CBZ:\n{e}", parent=self.root)
            print(f"❌ Error: {e}")

    def start_reading_order(self, order_id, start_index=0):
        conn = sqlite3.connect(self.db_file); cursor = conn.cursor()
        cursor.execute("SELECT name FROM reading_orders WHERE id = ?", (order_id,))
        order_name_res = cursor.fetchone()
        if not order_name_res: conn.close(); return

        cursor.execute("""
            SELECT c.path FROM reading_order_items roi
            JOIN comics c ON roi.comic_id = c.id
            WHERE roi.order_id = ? ORDER BY roi.sequence_number
        """, (order_id,))
        comic_paths = [row[0] for row in cursor.fetchall()]; conn.close()

        if not comic_paths:
            messagebox.showinfo("Orden Vacía", "Esta orden de lectura no tiene cómics.", parent=self.root)
            return
        if not (0 <= start_index < len(comic_paths)):
            messagebox.showinfo("Fin de la Orden", "Has llegado al final de la orden de lectura.", parent=self.root)
            return
            
        comic_to_read_path = comic_paths[start_index]
        context = {
            "order_id": order_id, "order_name": order_name_res[0],
            "current_index": start_index, "total_comics": len(comic_paths)
        }

        if os.path.exists(comic_to_read_path):
            ComicReaderWindow(self.root, comic_to_read_path, reading_order_context=context)
        else:
            messagebox.showerror("Archivo no encontrado", 
                                 f"No se pudo encontrar el cómic:\n{comic_to_read_path}", parent=self.root)

    def edit_selected_comic(self):
        path = self.get_selected_comic_path()
        if not path:
            messagebox.showwarning("Sin selección", "No hay ningún cómic seleccionado en la biblioteca.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Archivo no encontrado", f"El archivo del cómic no se encuentra en la ruta especificada:\n\n{path}")
            return
        
        file_list_context = [comic['path'] for comic in self.library_data]
        try:
            current_index_context = file_list_context.index(path)
        except ValueError:
            file_list_context = [path]
            current_index_context = 0
            
        self.open_metadata_editor(path, file_list_context=file_list_context, current_index_context=current_index_context)
         
    def open_selected_comic_location(self):
        path = self.get_selected_comic_path()
        if path and os.path.exists(path): webbrowser.open(f'file:///{os.path.realpath(os.path.dirname(path))}')
        elif path: messagebox.showerror("Error", f"La carpeta del archivo no se encuentra en la ruta:\n{path}")
    
    def setup_database(self):
        try:
            self.author_images_path.mkdir(exist_ok=True)
        except (PermissionError, FileExistsError) as e:
            print(f"⚠️ No se pudo crear directorio author_images: {e}")
            # Intentar con ruta alternativa
            self.author_images_path = Path("./author_imgs")
            try:
                self.author_images_path.mkdir(exist_ok=True)
            except Exception as e2:
                print(f"⚠️ Usando directorio temporal: {e2}")
                import tempfile
                self.author_images_path = Path(tempfile.gettempdir()) / "author_images"
                self.author_images_path.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("CREATE TABLE IF NOT EXISTS comics (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, series_group TEXT, series TEXT, number TEXT, title TEXT, publisher TEXT, year INTEGER, month INTEGER, day INTEGER, writer TEXT, penciller TEXT, inker TEXT, colorist TEXT, letterer TEXT, coverartist TEXT, editor TEXT, summary TEXT, storyarc TEXT, characters TEXT, teams TEXT, web TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS authors (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, biography TEXT, photo_filename TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS comic_authors (comic_id INTEGER, author_id INTEGER, role TEXT NOT NULL, FOREIGN KEY (comic_id) REFERENCES comics (id) ON DELETE CASCADE, FOREIGN KEY (author_id) REFERENCES authors (id) ON DELETE CASCADE, PRIMARY KEY (comic_id, author_id, role))")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reading_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reading_order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                comic_id INTEGER NOT NULL,
                sequence_number INTEGER NOT NULL,
                FOREIGN KEY (order_id) REFERENCES reading_orders (id) ON DELETE CASCADE,
                FOREIGN KEY (comic_id) REFERENCES comics (id) ON DELETE CASCADE,
                UNIQUE(order_id, comic_id),
                UNIQUE(order_id, sequence_number)
            )
        """)
        
        try:
            cursor.execute("ALTER TABLE comics ADD COLUMN series_group TEXT")
        except sqlite3.OperationalError: pass
        conn.commit(); conn.close(); print("Base de datos lista.")

    def scan_library_folder(self, sync_mode='full'):
        title_map = {
            'full': "Selecciona la carpeta RAÍZ de tu biblioteca para sincronizar",
            'additive': "Selecciona una carpeta para AÑADIR a tu biblioteca"
        }
        folder_path = filedialog.askdirectory(title=title_map.get(sync_mode, "Selecciona una carpeta"))
        if not folder_path: return
        
        # Preguntar si quiere resetear los grupos (Tags/series_group)
        reset_groups = messagebox.askyesno(
            "Grupos de Series",
            "¿Quieres IGNORAR los grupos existentes en los metadatos?\n\n"
            "• SÍ: Dejará los grupos vacíos (puedes editarlos manualmente después)\n"
            "• NO: Conservará los grupos que ya tengan los cómics en sus metadatos\n\n"
            "Recomendado: SÍ si los grupos actuales están incorrectos",
            parent=self.root
        )
            
        self.status_var.set(f"Buscando archivos .cbz en {folder_path}...")
        self.scan_btn_ref.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
        from pathlib import Path
        folder_path_obj = Path(folder_path)
        self.status_var.set(f"Listando archivos en {folder_path_obj}...")
        self.root.update_idletasks()
        
        cbz_files = []
        try:
            print(f"🔍 Iniciando escaneo en: {folder_path}")
            all_files = list(folder_path_obj.glob('**/*.cbz'))
            print(f"📁 Archivos CBZ encontrados: {len(all_files)}")
            
            self.status_var.set(f"Encontrados {len(all_files)} archivos. Preparando para procesar...")
            self.root.update_idletasks()
            
            for file_path in all_files:
                cbz_path = str(file_path)
                print(f"📖 Archivo encontrado: {cbz_path}")
                cbz_files.append(cbz_path)
                
        except Exception as e:
            print(f"ERROR CRÍTICO durante el escaneo de archivos con pathlib: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error de Escaneo", f"No se pudo acceder a la ruta de la biblioteca.\n\n{e}", parent=self.root)
            self.status_var.set("Error de escaneo.")
            self.scan_btn_ref.config(state=tk.NORMAL)
            return

        print(f"📊 Total de archivos CBZ para procesar: {len(cbz_files)}")
        
        if not cbz_files:
            messagebox.showinfo("Finalizado", f"No se encontraron archivos .cbz en:\n{folder_path}\n\nVerifica que:\n• La carpeta contiene archivos .cbz\n• Tienes permisos de lectura\n• Los archivos no están en uso", parent=self.root)
            self.status_var.set("Listo.")
            self.scan_btn_ref.config(state=tk.NORMAL)
            return
        threading.Thread(target=self._scan_thread, args=(cbz_files, sync_mode, reset_groups), daemon=True).start()

    def _scan_thread(self, cbz_files, sync_mode, reset_groups=False):
        conn = None
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            
            print("Optimizando: Cargando paths existentes de la base de datos en memoria...")
            db_paths = {row[0] for row in cursor.execute("SELECT path FROM comics")}
            print(f"Encontrados {len(db_paths)} cómics en la base de datos.")

            if sync_mode == 'full':
                file_paths_set = set(cbz_files)
                paths_to_delete = db_paths - file_paths_set
                if paths_to_delete:
                    print(f"Sincronización completa: Eliminando {len(paths_to_delete)} cómics obsoletos...")
                    cursor.executemany("DELETE FROM comics WHERE path = ?", [(p,) for p in paths_to_delete])
                    conn.commit()
                    try:
                        self.root.after(0, self.clear_thumbnail_cache, paths_to_delete)
                    except RuntimeError:
                        pass  # Ventana cerrada
                    db_paths -= paths_to_delete

            comics_a_procesar = []
            for path in cbz_files:
                if path not in db_paths:
                    comics_a_procesar.append(path)
            
            total_a_procesar = len(comics_a_procesar)
            print(f"📈 Estadísticas del escaneo:")
            print(f"   • Total de archivos CBZ: {len(cbz_files)}")
            print(f"   • Ya en base de datos: {len(db_paths)}")
            print(f"   • Nuevos para procesar: {total_a_procesar}")
            
            if total_a_procesar == 0:
                print("✅ No se encontraron cómics nuevos para añadir.")
                if self.is_running: self.root.after(0, self.finish_scan)
                conn.close()
                return

            print(f"🚀 Iniciando procesamiento de {total_a_procesar} cómics nuevos...")
            
            commit_counter = 0
            cursor.execute("BEGIN TRANSACTION")

            for i, cbz_path in enumerate(comics_a_procesar):
                if not self.is_running: break
                
                if (i % 25 == 0) or (i == total_a_procesar - 1):
                    try:
                        self.root.after(0, self.status_var.set, f"Escaneando cómics nuevos... {i+1}/{total_a_procesar}")
                    except RuntimeError:
                        pass  # Ventana cerrada

                print(f"📚 Procesando [{i+1}/{total_a_procesar}]: {os.path.basename(cbz_path)}")
                
                # Leer metadatos del CBZ
                raw_metadata = read_comicinfo_from_cbz(cbz_path)
                print(f"   📄 Metadatos raw: {type(raw_metadata)} - {len(str(raw_metadata)) if raw_metadata else 0} chars")
                metadata = {}
                
                # Procesar metadatos si existen
                if raw_metadata:
                    if isinstance(raw_metadata, (str, bytes)):
                        try:
                            from antmar.metadata import parse_comicinfo_xml
                            metadata = parse_comicinfo_xml(raw_metadata) or {}
                        except Exception as e:
                            print(f"Error parseando metadatos de {os.path.basename(cbz_path)}: {e}")
                            metadata = {}
                    elif isinstance(raw_metadata, dict):
                        metadata = raw_metadata
                
                # Si no hay metadatos válidos, crear entrada básica con el nombre del archivo
                if not metadata:
                    print(f"   ⚠️ Sin metadatos válidos, usando datos del nombre de archivo")
                    filename = os.path.splitext(os.path.basename(cbz_path))[0]
                    metadata = {'Series': filename, 'Title': filename}
                else:
                    print(f"   ✅ Metadatos procesados: {list(metadata.keys())}")
                
                # Si reset_groups está activo, intentar inferir un grupo automático o dejarlo vacío
                if reset_groups:
                    auto_group = self._detect_comic_group(metadata, cbz_path)
                    metadata['Tags'] = auto_group
                    print(f"   🏷️ Grupo detectado: {auto_group}")

                # Preparar datos para inserción
                series = metadata.get('Series', '')[:100] if metadata.get('Series') else ''  # Limitar longitud
                title = metadata.get('Title', '')[:200] if metadata.get('Title') else ''
                
                print(f"   💾 Insertando: Serie='{series}' Título='{title}'")
                
                data_tuple = (metadata.get('Tags'), series, metadata.get('Number'), title, metadata.get('Publisher'), metadata.get('Year'), metadata.get('Month'), metadata.get('Day'), metadata.get('Writer'), metadata.get('Penciller'), metadata.get('Inker'), metadata.get('Colorist'), metadata.get('Letterer'), metadata.get('CoverArtist'), metadata.get('Editor'), metadata.get('Summary'), metadata.get('StoryArc'), metadata.get('Characters'), metadata.get('Teams'), metadata.get('Web'), cbz_path)
                
                cursor.execute("INSERT INTO comics (series_group, series, number, title, publisher, year, month, day, writer, penciller, inker, colorist, letterer, coverartist, editor, summary, storyarc, characters, teams, web, path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_tuple)
                comic_id = cursor.lastrowid
                
                roles = ['Writer', 'Penciller', 'Inker', 'Colorist', 'Letterer', 'CoverArtist', 'Editor']
                for role in roles:
                    authors_str = metadata.get(role)
                    if not authors_str: continue
                    for name in [name.strip() for name in re.split('[,;]', authors_str) if name.strip()]:
                        cursor.execute("INSERT OR IGNORE INTO authors (name) VALUES (?)", (name,))
                        cursor.execute("SELECT id FROM authors WHERE name = ?", (name,))
                        author_id_res = cursor.fetchone()
                        if author_id_res:
                            cursor.execute("INSERT OR IGNORE INTO comic_authors (comic_id, author_id, role) VALUES (?, ?, ?)", (comic_id, author_id_res[0], role))
                
                commit_counter += 1
                if commit_counter >= 100:
                    conn.commit()
                    cursor.execute("BEGIN TRANSACTION")
                    commit_counter = 0
            
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"ERROR CRÍTICO en el hilo de escaneo: {e}")
            traceback.print_exc()
        finally:
            if conn:
                conn.close()

        if self.is_running:
            try:
                try:
                    self.root.after(0, self.finish_scan)
                except RuntimeError:
                    pass  # Ventana cerrada
            except RuntimeError:
                print("⚠️ No se pudo ejecutar finish_scan - ventana cerrada")
    
    def _detect_comic_group(self, metadata, cbz_path):
        """Intenta detectar automáticamente el grupo de un cómic basándose en sus metadatos"""
        publisher = (metadata.get('Publisher') or '').lower()
        language = (metadata.get('LanguageISO') or metadata.get('Language') or '').lower()
        series = (metadata.get('Series') or '').lower()
        
        # Detectar MANGA
        manga_keywords = ['manga', 'shonen', 'shojo', 'seinen', 'josei', 'kodansha', 'shueisha', 'viz']
        manga_publishers = ['norma editorial', 'planeta comic', 'ivrea', 'panini manga', 'ecc manga']
        
        if any(keyword in series for keyword in manga_keywords):
            return 'MANGA'
        if any(pub in publisher for pub in manga_publishers) and 'manga' in publisher:
            return 'MANGA'
        if language in ['ja', 'jp', 'jpn', 'japanese']:
            return 'MANGA'
        
        # Detectar MARVEL
        marvel_keywords = ['marvel', 'x-men', 'avengers', 'spider-man', 'iron man', 'captain america']
        if 'marvel' in publisher or any(keyword in series for keyword in marvel_keywords):
            return 'MARVEL'
        
        # Detectar DC COMICS
        dc_keywords = ['dc comics', 'batman', 'superman', 'wonder woman', 'justice league', 'flash', 'green lantern']
        if 'dc comics' in publisher or any(keyword in series for keyword in dc_keywords):
            return 'DC COMICS'
        
        # Detectar EUROPEO
        european_publishers = ['dargaud', 'dupuis', 'casterman', 'glenat', 'delcourt', 'lombard', 'norma', 'dibbuks']
        european_keywords = ['asterix', 'tintin', 'spirou', 'lucky luke', 'blake', 'mortimer']
        
        if any(pub in publisher for pub in european_publishers):
            # Verificar que NO sea manga
            if 'manga' not in publisher:
                return 'EUROPEO'
        
        if any(keyword in series for keyword in european_keywords):
            return 'EUROPEO'
        
        if language in ['fr', 'french', 'francés', 'frances', 'be', 'belgian']:
            return 'EUROPEO'
        
        # Detectar INDEPENDIENTES
        indie_publishers = ['image', 'dark horse', 'idw', 'boom', 'dynamite', 'valiant']
        if any(pub in publisher for pub in indie_publishers):
            return 'INDEPENDIENTES'
        
        # Si no se detecta nada, devolver None (sin grupo)
        return None
            
    def clear_thumbnail_cache(self, paths_to_clear):
        if not hasattr(self, 'thumbnail_cache'): return
        cleared_count = 0
        for path in paths_to_clear:
            if path in self.thumbnail_cache:
                del self.thumbnail_cache[path]
                cleared_count += 1
        if cleared_count > 0:
            print(f"Limpiadas {cleared_count} miniaturas del caché.")
    
    def finish_scan(self):
        self.status_var.set("Escaneo completado. Actualizando vista...")
        self.refresh_library_view()
        messagebox.showinfo("Escaneo Finalizado", "La biblioteca ha sido actualizada.", parent=self.root)
        self.status_var.set("Listo.")
        self.scan_btn_ref.config(state=tk.NORMAL)
    
    def _on_mousewheel(self, event):
        if self.library_view_mode.get() == "thumb":
            self.thumb_canvas.yview_scroll(int(-1*(event.delta/120)), "units"); self.root.after(50, self._lazy_load_thumbnails)
    
    def _toggle_library_view(self):
        if self.library_view_mode.get() == "list":
            self.library_view_mode.set("thumb")
            self.list_view_frame.grid_remove()
            self.thumb_view_frame.grid(row=2, column=0, sticky="nsew")
            self.root.update_idletasks()
            self._repopulate_thumbnail_view()
        else:
            self.library_view_mode.set("list")
            self.thumb_view_frame.grid_remove()
            self.list_view_frame.grid(row=2, column=0, sticky="nsew")
    
    def _on_stack_view_toggle(self):
        """Handler cuando se activa/desactiva vista de pilas"""
        print(f"🔄 Vista de pilas {'activada' if self.stack_view_var.get() else 'desactivada'}")
        
        # Si se desactiva vista de pilas, salir de cualquier pila abierta
        if not self.stack_view_var.get():
            self.stack_navigation.clear()
            self._hide_back_button()
        
        # Refrescar vista si estamos en modo miniaturas
        if self.library_view_mode.get() == "thumb":
            self._repopulate_thumbnail_view()

    def _repopulate_thumbnail_view(self, event=None):
        """Repuebla la vista de miniaturas al cambiar agrupación"""
        print("🔄 Repoblando vista de miniaturas...")
        
        # Limpiar widgets existentes
        for widget in self.thumb_scrollable_frame.winfo_children():
            widget.destroy()
        
        # Limpiar diccionario de widgets (pero mantener cache de imágenes)
        old_widgets = self.thumbnail_widgets.copy()
        self.thumbnail_widgets.clear()
        
        # Limpiar referencias de imágenes de widgets destruidos
        for path, widgets in old_widgets.items():
            try:
                if 'cover' in widgets and hasattr(widgets['cover'], 'image'):
                    widgets['cover'].image = None
            except:
                pass
        
        if not self.library_data:
            print("⚠️ No hay datos de biblioteca para mostrar")
            return

        # Configurar disposición
        thumb_width = 160
        container_width = self.thumb_canvas.winfo_width()
        cols = max(1, container_width // thumb_width)

        group_by_key = self.grouping_options[self.group_by_combo.get()]
        group_by_name = self.group_by_combo.get()
        
        # Detectar si estamos en modo vista de pilas y hay agrupación activa
        if self.stack_view_var.get() and group_by_key is not None:
            print(f"📚 Mostrando vista de pilas agrupada por: {group_by_name}")
            self._show_stack_view(group_by_key, group_by_name, cols)
            return
        
        print(f"📖 Mostrando vista normal de miniaturas")
        current_row = 0
        current_col = 0
        last_group_value = object()
        created_widgets = 0

        for comic in self.library_data:
            group_value = comic[group_by_key]

            # Insertar header de grupo cuando cambie
            if group_value != last_group_value:
                if current_col != 0:
                    current_row += 1
                
                group_name = str(group_value) if group_value else f"(Sin {group_by_name})"
                header_label = ttk.Label(self.thumb_scrollable_frame, text=group_name, 
                                        font="-weight bold -size 11", anchor="w")
                header_label.grid(row=current_row, column=0, columnspan=cols, 
                                sticky="ew", pady=(15, 5), padx=5)
                
                current_row += 1
                current_col = 0
                last_group_value = group_value

            # Crear widget para el cómic
            path = comic['path']
            
            frame = ttk.Frame(self.thumb_scrollable_frame, padding=5)
            frame.grid(row=current_row, column=current_col, sticky='nsew')
            
            # Label para la portada con imagen placeholder
            cover_label = ttk.Label(frame, image=self.placeholder_image, anchor=CENTER)
            cover_label.pack(fill=tk.BOTH, expand=True)
            
            # Si ya tenemos la imagen en cache, usarla inmediatamente
            if path in self.thumbnail_cache and self.thumbnail_cache[path] not in (None, "loading"):
                cached_photo = self.thumbnail_cache[path]
                cover_label.config(image=cached_photo, text="")
                cover_label.image = cached_photo
            
            # Título del cómic
            title_text = f"{comic['series'] or '?'} #{comic['number'] or '?'}"
            title_label = ttk.Label(frame, text=title_text, anchor=CENTER, 
                                   wraplength=thumb_width - 10)
            title_label.pack(fill=tk.X, pady=(5, 0))
            
            # Guardar referencias del widget
            self.thumbnail_widgets[path] = {
                'frame': frame, 
                'cover': cover_label, 
                'title': title_label
            }
            
            # Vincular eventos de click
            for widget in (frame, cover_label, title_label):
                widget.bind("<Button-1>", lambda e, p=path: self.on_thumbnail_selected(p))
            
            current_col += 1
            if current_col >= cols:
                current_col = 0
                current_row += 1
            
            created_widgets += 1
        
        print(f"✅ {created_widgets} widgets de miniaturas creados")
        
        # Iniciar carga perezosa de imágenes después de un breve retraso
        self.root.after(200, self._lazy_load_thumbnails)
    
    def _show_stack_view(self, group_by_key, group_by_name, cols):
        """Muestra la vista de pilas agrupadas"""
        print(f"🗂️ Creando vista de pilas por {group_by_name}...")
        
        # Agrupar library_data por group_by_key
        from collections import defaultdict
        groups = defaultdict(list)
        
        for comic in self.library_data:
            group_value = comic[group_by_key]
            if group_value is None:
                group_value = f"(Sin {group_by_name})"
            groups[group_value].append(comic)
        
        # Crear pilas para cada grupo
        current_row = 0
        current_col = 0
        thumb_width = 160
        
        for group_name, comics_list in sorted(groups.items()):
            # Crear frame para la pila
            stack_frame = ttk.Frame(self.thumb_scrollable_frame, padding=5)
            stack_frame.grid(row=current_row, column=current_col, sticky='nsew')
            
            # Portada representativa (del primer cómic del grupo)
            representative_path = comics_list[0]['path']
            
            # Label para la portada con placeholder
            cover_label = ttk.Label(stack_frame, image=self.placeholder_image, anchor=tk.CENTER)
            cover_label.pack(fill=tk.BOTH, expand=True)
            
            # Overlay con contador de elementos (usando tk.Label para tener más control de colores)
            overlay_frame = tk.Frame(stack_frame, bg='black')
            overlay_frame.place(relx=0.5, rely=0.05, anchor=CENTER)
            
            count_label = tk.Label(
                overlay_frame,
                text=f"📚 {len(comics_list)}",
                font="-size 14 -weight bold",
                fg="white",
                bg="black"
            )
            count_label.pack(padx=8, pady=4)
            
            # Título del grupo
            title_label = ttk.Label(
                stack_frame,
                text=str(group_name),
                anchor=tk.CENTER,
                wraplength=thumb_width - 10,
                font="-weight bold"
            )
            title_label.pack(fill=tk.X, pady=(5, 0))
            
            # Si la portada está en cache, usarla inmediatamente
            if representative_path in self.thumbnail_cache and self.thumbnail_cache[representative_path] not in (None, "loading"):
                cached_photo = self.thumbnail_cache[representative_path]
                cover_label.config(image=cached_photo, text="")
                cover_label.image = cached_photo
            else:
                # Cargar en segundo plano
                if representative_path not in self.thumbnail_cache:
                    self.thumbnail_cache[representative_path] = "loading"
                    threading.Thread(target=self._load_single_thumbnail, args=(representative_path,), daemon=True).start()
            
            # Guardar referencia del widget
            stack_key = f"stack_{group_name}"
            self.thumbnail_widgets[stack_key] = {
                'frame': stack_frame,
                'cover': cover_label,
                'title': title_label,
                'count': count_label,
                'representative_path': representative_path  # Store for thumbnail updates
            }
            
            # For lazy loading: also map representative_path to stack widget
            # This allows _update_thumbnail_widget to find the widget when thumbnail loads
            if representative_path not in self.thumbnail_widgets:
                self.thumbnail_widgets[representative_path] = self.thumbnail_widgets[stack_key]
            
            # Hacer la pila clickeable
            for widget in (stack_frame, cover_label, title_label, count_label):
                widget.bind("<Button-1>", lambda e, cl=comics_list, gn=group_name: self._enter_stack(cl, gn))
            
            current_col += 1
            if current_col >= cols:
                current_col = 0
                current_row += 1
        
        print(f"✅ {len(groups)} pilas creadas")
        
        # Iniciar carga perezosa después de un breve retraso
        self.root.after(200, self._lazy_load_thumbnails)
    
    def _enter_stack(self, comics_list, group_name):
        """Entra en una pila específica para mostrar sus cómics"""
        print(f"📂 Entrando en pila: {group_name} ({len(comics_list)} cómics)")
        
        # Guardar estado actual
        self.stack_navigation.append({
            'library_data': self.library_data.copy(),
            'group_name': group_name,
            'frame_title': self.library_frame.cget('text')  # Save original title
        })
        
        # Cambiar library_data a la lista de cómics de la pila
        self.library_data = comics_list
        
        # Actualizar título del frame
        self.library_frame.config(text=f"Biblioteca → {group_name}")
        
        # Mostrar botón "Volver"
        self._show_back_button()
        
        # Desactivar temporalmente la vista de pilas para mostrar cómics individuales
        # (guardamos el estado para restaurarlo al salir)
        saved_stack_view = self.stack_view_var.get()
        self.stack_view_var.set(False)
        
        # Repoblar vista con cómics individuales
        self._repopulate_thumbnail_view()
        
        # Restaurar estado de vista de pilas (pero no refrescar)
        self.stack_view_var.set(saved_stack_view)
        
        print(f"✅ Vista de pila mostrada: {len(comics_list)} cómics")
    
    def _exit_stack(self):
        """Sale de la pila actual y vuelve a la vista de pilas"""
        if not self.stack_navigation:
            print("⚠️ No hay pilas en el stack de navegación")
            return
        
        # Restaurar estado anterior
        previous_state = self.stack_navigation.pop()
        self.library_data = previous_state['library_data']
        group_name = previous_state['group_name']
        frame_title = previous_state.get('frame_title', 'Biblioteca')
        
        print(f"⬅️ Saliendo de pila: {group_name}")
        
        # Restaurar título del frame
        self.library_frame.config(text=frame_title)
        
        # Ocultar botón si ya no hay más pilas en el stack
        if not self.stack_navigation:
            self._hide_back_button()
        
        # Repoblar vista en modo pilas
        self._repopulate_thumbnail_view()
        
        print(f"✅ Vista de pilas restaurada")
    
    def _show_back_button(self):
        """Muestra el botón 'Volver' en la interfaz"""
        if not hasattr(self, 'back_button_frame'):
            # Crear frame para el botón si no existe
            self.back_button_frame = ttk.Frame(self.top_controls_frame)
            self.back_button = ttk.Button(
                self.back_button_frame,
                text="← Volver",
                command=self._exit_stack
            )
            self.back_button.pack()
        
        # Mostrar el frame del botón al inicio
        self.back_button_frame.pack(side=tk.LEFT, padx=(0, 10))
        print("✅ Botón 'Volver' mostrado")
    
    def _hide_back_button(self):
        """Oculta el botón 'Volver'"""
        if hasattr(self, 'back_button_frame'):
            self.back_button_frame.pack_forget()
            print("✅ Botón 'Volver' ocultado")
        
    def _lazy_load_thumbnails(self):
        """Carga perezosa de miniaturas visibles"""
        if self.library_view_mode.get() != "thumb" or not self.root.winfo_exists(): 
            return
            
        try:
            canvas_height = self.thumb_canvas.winfo_height()
            if canvas_height <= 1:
                return  # Canvas aún no inicializado
                
            # Calcular área visible
            visible_top = self.thumb_canvas.yview()[0] * self.thumb_scrollable_frame.winfo_height()
            visible_bottom = visible_top + canvas_height + 200  # Buffer extra
            
            # Lista de archivos a cargar
            to_load = []
            
            for key, widgets in self.thumbnail_widgets.items():
                if not widgets['frame'].winfo_exists(): 
                    continue
                    
                widget_y = widgets['frame'].winfo_y()
                
                # If in visible area
                if visible_top <= widget_y <= visible_bottom:
                    # For stacks, the key is 'stack_{group_name}', skip them
                    # (stacks are already processed in _show_stack_view)
                    if key.startswith('stack_'):
                        continue
                    
                    # For regular comics, the key is the path
                    path = key
                    if path not in self.thumbnail_cache and os.path.exists(path):
                        # Marcar como cargando para evitar duplicados
                        self.thumbnail_cache[path] = "loading"
                        to_load.append(path)
            
            # Cargar imágenes en lotes para mejor rendimiento
            if to_load:
                print(f"🖼️ Cargando {len(to_load)} miniaturas...")
                for path in to_load:
                    threading.Thread(target=self._load_single_thumbnail, args=(path,), daemon=True).start()
                    
        except Exception as e:
            print(f"❌ Error en _lazy_load_thumbnails: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_single_thumbnail(self, path):
        """Carga una miniatura individual en segundo plano"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(path):
                print(f"⚠️ Archivo no encontrado para miniatura: {path}")
                if path in self.thumbnail_cache:
                    del self.thumbnail_cache[path]
                return
            
            # Cargar imagen con el tamaño correcto
            pil_image = get_cover_from_cbz(path, (150, 225))
            
            if pil_image:
                try:
                    # Convertir PIL Image a PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Guardar en cache
                    self.thumbnail_cache[path] = photo
                    
                    # Actualizar widget en el hilo principal si la ventana existe
                    if self.root.winfo_exists():
                        self.root.after(0, self._update_thumbnail_widget, path, photo)
                    
                    print(f"✅ Miniatura cargada: {os.path.basename(path)}")
                    
                except Exception as photo_err:
                    print(f"❌ Error creando PhotoImage para {os.path.basename(path)}: {photo_err}")
                    # Marcar como fallida en lugar de "loading"
                    self.thumbnail_cache[path] = None
            else:
                print(f"⚠️ No se pudo extraer portada de: {os.path.basename(path)}")
                # Marcar como fallida
                self.thumbnail_cache[path] = None
                
        except Exception as e:
            print(f"❌ Error cargando miniatura {os.path.basename(path)}: {e}")
            # Limpiar estado de carga
            if path in self.thumbnail_cache:
                self.thumbnail_cache[path] = None
    
    def _update_thumbnail_widget(self, path, photo):
        """Actualiza el widget de miniatura con la imagen cargada"""
        try:
            # Verificar que el widget existe y es válido
            if (path in self.thumbnail_widgets and 
                self.thumbnail_widgets[path]['cover'].winfo_exists()):
                
                cover_widget = self.thumbnail_widgets[path]['cover']
                
                # Limpiar imagen anterior si existe
                if hasattr(cover_widget, 'image') and cover_widget.image:
                    old_image = cover_widget.image
                    cover_widget.image = None
                    del old_image
                
                # Aplicar nueva imagen
                cover_widget.config(image=photo, text="")
                
                # Mantener referencia para evitar garbage collection
                cover_widget.image = photo
                
        except Exception as e:
            print(f"❌ Error actualizando widget de miniatura para {os.path.basename(path)}: {e}")

    def clear_thumbnail_cache(self, paths_to_keep=None):
        """Limpia el cache de miniaturas manteniendo solo las especificadas"""
        if not hasattr(self, 'thumbnail_cache'):
            return
            
        try:
            if paths_to_keep is None:
                # Limpiar todo el cache
                for path, photo in self.thumbnail_cache.items():
                    if photo and photo != "loading":
                        try:
                            del photo
                        except:
                            pass
                self.thumbnail_cache.clear()
                print("🧹 Cache de miniaturas completamente limpiado")
            else:
                # Limpiar solo las que no están en paths_to_keep
                paths_to_keep_set = set(paths_to_keep)
                to_remove = []
                
                for path in self.thumbnail_cache.keys():
                    if path not in paths_to_keep_set:
                        to_remove.append(path)
                
                for path in to_remove:
                    photo = self.thumbnail_cache.get(path)
                    if photo and photo != "loading":
                        try:
                            del photo
                        except:
                            pass
                    del self.thumbnail_cache[path]
                
                print(f"🧹 {len(to_remove)} miniaturas eliminadas del cache")
                
        except Exception as e:
            print(f"❌ Error limpiando cache de miniaturas: {e}")

    def on_list_item_selected(self, event=None):
        selection = self.library_tree.selection()
        if not selection or not self.library_tree.parent(selection[0]): return
        path = self.library_tree.item(selection[0], 'values')[0]; self.on_comic_selected(path)
    
    def on_thumbnail_selected(self, path): self.on_comic_selected(path)
    
    def get_selected_comic_path(self): return self.selected_comic_path
    
    def on_comic_selected(self, path):
        if self.selected_comic_path and self.selected_comic_path in self.thumbnail_widgets and self.thumbnail_widgets[self.selected_comic_path]['frame'].winfo_exists():
            self.thumbnail_widgets[self.selected_comic_path]['frame'].config(style="TFrame")
        self.selected_comic_path = path
        if self.library_view_mode.get() == "thumb" and path in self.thumbnail_widgets and self.thumbnail_widgets[path]['frame'].winfo_exists():
            self.thumbnail_widgets[path]['frame'].config(style="Selected.TFrame")
        self.update_details_panel(path)
    
    def _on_author_link_click(self, author_name): AuthorDetailWindow(self, author_name)
    
    def _create_author_links(self, parent_frame, authors_str):
        for widget in parent_frame.winfo_children(): widget.destroy()
        if not authors_str: ttk.Label(parent_frame, text="N/A").pack(anchor="w"); return
        for i, name in enumerate([name.strip() for name in re.split('[,;]', authors_str) if name.strip()]):
            if i > 0: ttk.Label(parent_frame, text=", ").pack(side=tk.LEFT)
            link = ttk.Label(parent_frame, text=name, foreground=self.style.colors.info, cursor="hand2"); link.pack(side=tk.LEFT)
            link.bind("<Button-1>", lambda e, n=name: self._on_author_link_click(n))

    def refresh_library_view(self, event=None):
        """Refresca la vista de la biblioteca con filtros y agrupación"""
        print("🔄 Refrescando vista de biblioteca...")
        
        # Limpiar vista de lista
        for item in self.library_tree.get_children():
            self.library_tree.delete(item)

        # Obtener parámetros de filtrado y agrupación
        group_by_field = self.grouping_options[self.group_by_combo.get()]
        filter_field = self.filtering_options[self.filter_field_combo.get()]
        filter_query = self.filter_value_entry.get().strip()

        # Consultar base de datos
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        base_query = "SELECT * FROM comics"
        params = []
        
        if filter_query and filter_field != 'all':
            base_query += f" WHERE LOWER(COALESCE({filter_field}, '')) LIKE LOWER(?)"
            params.append(f"%{filter_query}%")
        elif filter_query and filter_field == 'all':
            search_fields = ["series_group", "series", "number", "title", "publisher", "year", "writer", "penciller", "inker", "colorist", "storyarc", "characters", "teams", "summary"]
            concatenated_fields = " || ' ' || ".join([f"COALESCE({field}, '')" for field in search_fields])
            base_query += f" WHERE LOWER({concatenated_fields}) LIKE LOWER(?)"
            params.append(f"%{filter_query}%")

        base_query += f" ORDER BY CASE WHEN {group_by_field} IS NULL OR {group_by_field} = '' THEN 1 ELSE 0 END, {group_by_field}, series, CAST(number AS REAL), number"

        cursor.execute(base_query, params)
        new_library_data = cursor.fetchall()
        conn.close()

        # Obtener lista de paths actuales
        current_paths = {comic['path'] for comic in new_library_data}
        
        # Limpiar cache de miniaturas que ya no están en la vista actual
        if hasattr(self, 'thumbnail_cache') and self.thumbnail_cache:
            self.clear_thumbnail_cache(paths_to_keep=current_paths)

        # Actualizar datos de biblioteca
        self.library_data = new_library_data

        # Poblar vista de lista
        group_nodes = {}
        for comic in self.library_data:
            group_value = comic[group_by_field]
            
            if not group_value:
                group_name = f"(Sin {self.group_by_combo.get()})"
            else:
                group_name = str(group_value)
            
            if group_name not in group_nodes:
                group_nodes[group_name] = self.library_tree.insert("", tk.END, text=group_name, open=True)
            
            parent_node = group_nodes[group_name]
            comic_values = (comic['path'], comic['series'], comic['number'], comic['title'], comic['year'])
            self.library_tree.insert(parent_node, tk.END, values=comic_values)

        # Actualizar vista de miniaturas si está activa
        if self.library_view_mode.get() == "thumb":
            self._repopulate_thumbnail_view()
        
        self.status_var.set(f"Biblioteca cargada. {len(self.library_data)} cómics encontrados.")
        print(f"✅ Vista refrescada: {len(self.library_data)} cómics mostrados")

    def clear_filter(self):
        self.filter_value_entry.delete(0, tk.END)
        self.filter_field_combo.set(list(self.filtering_options.keys())[0])
        self.refresh_library_view()
    
    def on_library_double_click(self, event=None):
        """Abre el lector al hacer doble clic en un cómic"""
        selection = self.lib_list.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.lib_list.item(item, 'values')
        
        if not values or len(values) == 0:
            return
        
        # Obtener la ruta del cómic
        comic_path = None
        for path, data in self.path_to_comic_map.items():
            if data['series'] == values[0] and str(data['number']) == str(values[1]):
                comic_path = path
                break
        
        if comic_path and os.path.exists(comic_path):
            # Cambiar a la pestaña del lector
            self.notebook.select(self.reader_tab)
            # Abrir el cómic
            self.reader_open_specific_comic(comic_path)
        else:
            messagebox.showwarning("Error", "No se pudo encontrar el archivo del cómic")
    
    def reader_open_specific_comic(self, path):
        """Abre un cómic específico en el lector (usado internamente)"""
        if not os.path.exists(path):
            messagebox.showerror("Error", "El archivo no existe")
            return
        
        try:
            # Limpiar cache anterior
            self.reader_images_cache = {}
            
            # Abrir el archivo
            if path.lower().endswith('.cbz'):
                zf = zipfile.ZipFile(path, 'r')
            else:  # CBR
                messagebox.showwarning("Formato no soportado", 
                                     "El lector actualmente solo soporta archivos CBZ.")
                return
            
            # Obtener lista de imágenes
            self.reader_pages = sorted([f for f in zf.namelist() 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
                                      key=natural_sort_key)
            
            if not self.reader_pages:
                messagebox.showwarning("Sin páginas", "No se encontraron imágenes en el archivo.")
                return
            
            self.reader_current_comic = path
            self.reader_zipfile = zf
            self.reader_current_page = 0
            
            # Actualizar interfaz
            comic_name = os.path.basename(path)
            self.reader_label.config(text=f"📖 {comic_name}")
            self.reader_page_label.config(text=f"Página: 1 / {len(self.reader_pages)}")
            
            # Mostrar primera página
            self.reader_show_page(0)
            
            # Bind de teclas para navegación
            self.reader_canvas.focus_set()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el cómic:\n{e}")
            traceback.print_exc()
    
    def update_details_panel(self, comic_path):
        try:
            print(f"📊 DEBUG: update_details_panel llamado para: {os.path.basename(comic_path)}")
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM comics WHERE path = ?", (comic_path,))
            comic_data = cursor.fetchone()
            conn.close()
            
            if not comic_data:
                print("⚠️ No se encontraron datos para este cómic")
                return
            
            print(f"✅ Datos encontrados: {len(comic_data.keys())} campos")
            
            get_text = lambda key: comic_data[key] if comic_data[key] is not None else ""
            
            # Envolver toda la actualización de UI en after() para asegurar hilo principal
            def update_ui():
                try:
                    self.detail_title_label.config(text=get_text('title') or "Sin Título")
                    self.detail_series_label.config(text=f"{get_text('series')} #{get_text('number')}")
                    
                    for widget in self.detail_publisher_frame.winfo_children():
                        widget.destroy()
                    publisher_name = get_text('publisher')
                    logo_photo = load_publisher_logo(publisher_name, height=40)
                    if logo_photo:
                        logo_label = ttk.Label(self.detail_publisher_frame, image=logo_photo)
                        logo_label.image = logo_photo
                        logo_label.pack(anchor="w")
                    else:
                        ttk.Label(self.detail_publisher_frame, text=publisher_name or "N/A").pack(anchor="w")
                        
                    for widget in self.detail_date_label_frame.winfo_children():
                        widget.destroy()
                    year, month, day = comic_data['year'], comic_data['month'], comic_data['day']
                    date_parts = []
                    if day: date_parts.append(str(day))
                    if month: date_parts.append(str(month))
                    if year: date_parts.append(str(year))
                    date_str = "/".join(date_parts) if date_parts else "Fecha desconocida"
                    ttk.Label(self.detail_date_label_frame, text=date_str).pack(anchor="w")

                    self._create_author_links(self.writer_frame, get_text('writer'))
                    self._create_author_links(self.penciller_frame, get_text('penciller'))
                    self._create_author_links(self.inker_frame, get_text('inker'))
                    self._create_author_links(self.colorist_frame, get_text('colorist'))
                    self._create_author_links(self.coverartist_frame, get_text('coverartist'))

                    self.detail_storyarc_label.config(text=get_text('storyarc') or "N/A")
                    self.detail_characters_label.config(text=get_text('characters') or "N/A")
                    self.detail_teams_label.config(text=get_text('teams') or "N/A")
                    self.detail_web_label.config(text=get_text('web') or "N/A")

                    self.detail_summary_text.config(state="normal")
                    self.detail_summary_text.delete(1.0, tk.END)
                    self.detail_summary_text.insert(tk.END, get_text('summary') or "No hay resumen disponible.")
                    self.detail_summary_text.config(state="disabled")
                    
                    self.detail_read_btn.config(state="normal")
                    self.detail_edit_btn.config(state="normal")
                    self.detail_open_folder_btn.config(state="normal")
                    
                    # Forzar actualización visual
                    self.root.update_idletasks()
                    print("✅ Panel de detalles actualizado")
                except Exception as e:
                    print(f"❌ Error actualizando UI: {e}")
                    traceback.print_exc()
            
            if self.root.winfo_exists():
                try:
                    self.root.after(0, update_ui)
                except RuntimeError:
                    pass  # Ventana cerrada
            
            threading.Thread(target=self._load_and_display_cover, args=(comic_path,), daemon=True).start()
        except Exception as e:
            print(f"❌ Error en update_details_panel: {e}")
            traceback.print_exc()

    def _load_and_display_cover(self, comic_path):
        """Carga y muestra la portada del cómic"""
        try:
            print(f"📷 Cargando portada de: {os.path.basename(comic_path)}")
            # get_cover_from_cbz ahora retorna PIL Image (no PhotoImage)
            pil_image = get_cover_from_cbz(comic_path, (350, 525))
            
            def update_cover():
                try:
                    if self.root.winfo_exists() and hasattr(self, 'detail_cover_label') and self.detail_cover_label.winfo_exists():
                        if pil_image:
                            # Convertir a PhotoImage en el hilo principal
                            cover_photo = ImageTk.PhotoImage(pil_image)
                            self.detail_cover_label.config(image=cover_photo, text="", bg="#1a1a1a")
                            self.detail_cover_label.image = cover_photo  # Mantener referencia
                            print("✅ Portada mostrada correctamente")
                        else:
                            self.detail_cover_label.config(image='', text="Portada no disponible", fg="white", bg="#1a1a1a")
                            self.detail_cover_label.image = None
                            print("⚠️ No hay portada disponible")
                        # Forzar actualización visual
                        self.detail_cover_label.update_idletasks()
                except Exception as e:
                    print(f"Error actualizando widget de portada: {e}")
            
            if self.root.winfo_exists():
                self.root.after(100, update_cover)
        except Exception as e:
            print(f"❌ Error cargando portada: {e}")
            traceback.print_exc()
            def show_error():
                try:
                    if self.root.winfo_exists() and hasattr(self, 'detail_cover_label') and self.detail_cover_label.winfo_exists():
                        self.detail_cover_label.config(image='', text=f"Error cargando portada", fg="red", bg="#1a1a1a")
                        self.detail_cover_label.image = None
                except:
                    pass
            if self.root.winfo_exists():
                try:
                    self.root.after(0, show_error)
                except RuntimeError:
                    pass  # Ventana cerrada

    def check_opencv(self):
        if self.remove_page_numbers_var.get() and not OPENCV_AVAILABLE:
            messagebox.showerror("Dependencia Faltante", "Esta función requiere OpenCV.\n\nEjecuta:\npip install opencv-python numpy\n\nY reinicia la aplicación."); self.remove_page_numbers_var.set(False)
    
    def open_batch_translator(self): BatchTranslatorWindow(self.root)
    def open_batch_renamer(self): BatchRenamerWindow(self.root)
    
    def load_settings(self):
        config = configparser.ConfigParser()
        if not os.path.exists('config.ini'): return
        config.read('config.ini')
        if 'Settings' in config:
            settings = config['Settings']
            self.profile_combo.set(settings.get('resolution_profile', 'Tablet 2K (2560p)')); self.update_resolution_fields()
            if self.profile_combo.get() == "Personalizado": self.entry_max_width.delete(0, tk.END); self.entry_max_width.insert(0, settings.get('custom_width', '1600')); self.entry_max_height.delete(0, tk.END); self.entry_max_height.insert(0, settings.get('custom_height', '2560'))
            self.quality_slider.set(settings.getfloat('webp_quality', 90.0))
            self.protect_double_pages_var.set(settings.getboolean('protect_double_pages', True)); self.remove_page_numbers_var.set(settings.getboolean('remove_page_numbers', False))
    
    def save_settings(self):
        config = configparser.ConfigParser(); config['Settings'] = {'resolution_profile': self.profile_combo.get(), 'custom_width': self.entry_max_width.get(), 'custom_height': self.entry_max_height.get(), 'webp_quality': self.quality_slider.get(), 'protect_double_pages': self.protect_double_pages_var.get(), 'remove_page_numbers': self.remove_page_numbers_var.get()}
        with open('config.ini', 'w') as configfile: config.write(configfile)

    def load_folder(self, folder_path):
        self.lbl_folder_path.config(text=folder_path, bootstyle=DEFAULT); self.status_var.set("Cargando..."); self.root.update_idletasks(); self.image_files.clear(); self.listbox.delete(0, tk.END)
        supported = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        try:
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(supported)], key=natural_sort_key)
            self.image_files = [os.path.join(folder_path, f) for f in files]; [self.listbox.insert(tk.END, f) for f in files]
            self.status_var.set(f"{len(self.image_files)} imágenes encontradas.");
            if self.image_files: self.listbox.selection_set(0, tk.END)
        except Exception as e: messagebox.showerror("Error", f"No se pudieron leer las imágenes.\n{e}"); self.status_var.set("Error al cargar.")
    
    def select_folder(self):
        folder_path = filedialog.askdirectory();
        if folder_path: self.load_folder(folder_path)
    
    def show_preview(self, event):
        if not self.listbox.curselection(): 
            return
        image_path = self.image_files[self.listbox.curselection()[0]]
        try:
            with Image.open(image_path) as img:
                # Obtener tamaño del widget de previsualización
                w, h = self.lbl_preview.winfo_width(), self.lbl_preview.winfo_height()
                
                # Tamaño objetivo balanceado para ver imagen y opciones
                if w > 1 and h > 1:
                    # Usar el 95% del espacio disponible del widget
                    target_w = int(w * 0.95)
                    target_h = int(h * 0.95)
                else:
                    # Valores por defecto balanceados
                    target_w = 380
                    target_h = 280
                
                # Asegurar un tamaño mínimo razonable
                target_w = max(target_w, 300)
                target_h = max(target_h, 250)
                
                print(f"DEBUG: Widget size: {w}x{h}, Target: {target_w}x{target_h}")
                
                # Calcular proporción para mantener aspect ratio
                img_w, img_h = img.size
                ratio = min(target_w / img_w, target_h / img_h)
                
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)
                
                # Redimensionar con alta calidad
                resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Crear y mostrar imagen
                photo = ImageTk.PhotoImage(resized_img)
                self.lbl_preview.config(image=photo, text="")
                self.lbl_preview.image = photo  # Mantener referencia
                
                print(f"DEBUG: Final image size: {new_w}x{new_h}")
                
        except Exception as e: 
            print(f"DEBUG: Error in preview: {e}")
            self.lbl_preview.config(image='', text=f"No se puede previsualizar\n\n{e}")
    
    def _on_preview_resize(self, event):
        """Redimensionar previsualización cuando cambia el tamaño del widget"""
        if hasattr(self, 'listbox') and self.listbox.curselection():
            # Pequeño delay para evitar múltiples redimensionados
            self.root.after(100, lambda: self.show_preview(None))
    
    def update_resolution_fields(self, event=None):
        profile = self.profile_combo.get(); dims = self.resolution_profiles.get(profile); is_custom = profile == "Personalizado"; state = tk.NORMAL if is_custom else tk.DISABLED
        for e in [self.entry_max_width, self.entry_max_height]: e.config(state='normal'); e.delete(0, tk.END)
        if not is_custom and dims: self.entry_max_width.insert(0, str(dims[0])); self.entry_max_height.insert(0, str(dims[1]))
        for e in [self.entry_max_width, self.entry_max_height]: e.config(state=state)
    
    def start_cbz_creation_thread(self):
        if not self.listbox.curselection(): return messagebox.showwarning("Selección Vacía", "Selecciona al menos una imagen.")
        selected_files = [self.image_files[i] for i in self.listbox.curselection()]
        output_path = filedialog.asksaveasfilename(defaultextension=".cbz", filetypes=[("Comic Book Archive", "*.cbz")], title="Guardar CBZ como...")
        if not output_path: return self.status_var.set("Creación cancelada.")
        threading.Thread(target=self.create_single_cbz_from_list, args=(selected_files, output_path, True), daemon=True).start()
    
    def decompress_cbz(self):
        cbz_path = filedialog.askopenfilename(title="Seleccionar CBZ", filetypes=[("Comic Book Archive", "*.cbz")])
        if not cbz_path: return
        folder_path = filedialog.askdirectory(title="Seleccionar Carpeta de Destino")
        if not folder_path: return
        try:
            self.status_var.set(f"Descomprimiendo {os.path.basename(cbz_path)}...")
            with zipfile.ZipFile(cbz_path, 'r') as zf: zf.extractall(folder_path)
            self.status_var.set("Descompresión completada."); messagebox.showinfo("Éxito", f"'{os.path.basename(cbz_path)}' descomprimido en:\n{folder_path}")
        except Exception as e:
            self.status_var.set("Error de descompresión."); messagebox.showerror("Error", f"Ocurrió un error al descomprimir:\n{e}")
    
    def reoptimize_cbz(self):
        input_cbz = filedialog.askopenfilename(title="Seleccionar CBZ para Re-optimizar", filetypes=[("Comic Book Archive", "*.cbz")])
        if not input_cbz: return
        output_cbz = filedialog.asksaveasfilename(defaultextension=".cbz", filetypes=[("Comic Book Archive", "*.cbz")], title="Guardar CBZ optimizado como...", initialfile=os.path.basename(input_cbz))
        if not output_cbz: return
        threading.Thread(target=self._reoptimize_thread, args=(input_cbz, output_cbz), daemon=True).start()
    
    def _reoptimize_thread(self, input_cbz, output_cbz):
        self.status_var.set(f"Re-optimizando {os.path.basename(input_cbz)}...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.status_var.set("Extrayendo imágenes..."); image_list, xml_data = [], None
                with zipfile.ZipFile(input_cbz, 'r') as zf:
                    file_list = sorted([f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif'))], key=natural_sort_key)
                    for filename in file_list: image_list.append(zf.extract(filename, temp_dir))
                    if 'comicinfo.xml' in [f.lower() for f in zf.namelist()]:
                        xml_data = zf.read('ComicInfo.xml').decode('utf-8-sig')
                if not self.create_single_cbz_from_list(image_list, output_cbz, ask_metadata=False): raise Exception("Fallo al crear el CBZ optimizado.")
                if xml_data:
                    self.status_var.set("Inyectando metadatos originales...")
                    if not inject_xml_into_cbz(output_cbz, xml_data): print(f"AVISO: No se pudo re-inyectar el XML.")
            self.status_var.set("Re-optimización completada."); self.root.after(0, lambda: messagebox.showinfo("Éxito", "CBZ re-optimizado correctamente."))
        except Exception as e: self.status_var.set(f"Error en re-optimización: {e}"); self.root.after(0, lambda: messagebox.showerror("Error", f"Ocurrió un error:\n{e}"))
    
    def start_volume_splitting(self):
        folder_path = filedialog.askdirectory(title="Selecciona la carpeta con TODAS las páginas del tomo")
        if not folder_path: return
        supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
        all_files_absolute = [os.path.join(folder_path, f) for f in sorted([f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)], key=natural_sort_key)]
        if not all_files_absolute: messagebox.showerror("Error", "No se encontraron imágenes.", parent=self.root); return
        splitter_dialog = ManualSplitterWindow(self.root, all_files_absolute, folder_path)
        cover_indices = splitter_dialog.final_indices
        if not cover_indices: self.status_var.set("División cancelada."); return
        config_dialog = BatchConfigDialog(self.root, initialvalue={'series_name': os.path.basename(folder_path), 'start_number': 1})
        config = config_dialog.result
        if not config: self.status_var.set("División cancelada."); return
        marathon_dialog = MarathonDialog(self.root, all_files_absolute, cover_indices, config)
        comic_configs = marathon_dialog.results
        if not comic_configs: self.status_var.set("División cancelada."); return
        output_folder = filedialog.askdirectory(title="Selecciona la carpeta para GUARDAR los CBZ creados")
        if not output_folder: self.status_var.set("División cancelada."); return
        self.lbl_folder_path.config(text=f"Dividiendo desde: {os.path.basename(folder_path)}"); self.status_var.set("Iniciando procesado en lote...")
        threading.Thread(target=self.split_volume_thread, args=(all_files_absolute, output_folder, comic_configs), daemon=True).start()
    
    def split_volume_thread(self, all_files_absolute, output_folder, comic_configs):
        try:
            for i, config in enumerate(comic_configs):
                self.status_var.set(f"Procesando en lote ({i+1}/{len(comic_configs)}): {config['filename']}")
                issue_files = all_files_absolute[config['start_index']:config['end_index']]
                output_path = os.path.join(output_folder, config['filename'])
                if not self.create_single_cbz_from_list(issue_files, output_path, ask_metadata=True, search_query=config['query']): 
                    raise Exception(f"Fallo al crear {config['filename']}")
            try:
                self.root.after(0, lambda: messagebox.showinfo("Proceso Completo", f"¡Se han procesado {len(comic_configs)} cómics con éxito!", parent=self.root))
            except RuntimeError:
                pass  # Ventana cerrada
            self.status_var.set("División y etiquetado finalizados.")
        except Exception as e:
            self.status_var.set("¡Error durante el procesado en lote!"); self.root.after(0, lambda: messagebox.showerror("Error", f"Ocurrió un error:\n{e}", parent=self.root)); traceback.print_exc()
    
    def create_single_cbz_from_list(self, image_list, output_path, ask_metadata=False, search_query=None):
        try:
            target_w, target_h = int(self.entry_max_width.get()), int(self.entry_max_height.get()); webp_quality = self.quality_slider.get(); protect_doubles = self.protect_double_pages_var.get(); remove_numbers = self.remove_page_numbers_var.get()
        except (ValueError, tk.TclError): target_w, target_h, webp_quality, protect_doubles, remove_numbers = 1600, 2560, 90, True, False
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for i, file_path in enumerate(image_list):
                    status_msg = f"Procesando {os.path.basename(output_path)} ({i+1}/{len(image_list)})"
                    if remove_numbers: status_msg += " - Limpiando..."
                    if self.is_running: self.status_var.set(status_msg)
                    with Image.open(file_path) as img:
                        if img.mode in ("RGBA", "P", "L"): img = img.convert("RGB")
                        if remove_numbers: img = remove_page_number(img)
                        o_w, o_h = img.size; is_double_page = o_w > o_h * 1.4
                        ratio = target_h / o_h if protect_doubles and is_double_page else min(target_w / o_w, target_h / o_h)
                        if ratio < 1: n_w, n_h = int(o_w * ratio), int(o_h * ratio); img = img.resize((n_w, n_h), Image.Resampling.LANCZOS)
                        img.save(os.path.join(temp_dir, f"{i:03d}.webp"), 'webp', quality=int(webp_quality))
                if self.is_running: self.status_var.set(f"Empaquetando '{os.path.basename(output_path)}'...")
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for webp_file in sorted(os.listdir(temp_dir)): zf.write(os.path.join(temp_dir, webp_file), arcname=webp_file)
            if ask_metadata and self.is_running: self.root.after(100, lambda: self.open_metadata_editor(output_path, search_query=search_query))
            return True
        except Exception as e: print(f"Error creando CBZ: {e}"); self.status_var.set(f"Error: {e}"); return False
    
    def open_existing_cbz_for_metadata(self):
        """Abrir ventana para editar metadatos de un CBZ externo"""
        cbz_path = filedialog.askopenfilename(
            title="Selecciona un archivo CBZ/CBR",
            filetypes=[("Archivos de cómic", "*.cbz *.cbr"), ("Todos", "*.*")],
            parent=self.root
        )
        
        if not cbz_path:
            return
        
        print(f"🔍 Abriendo CBZ para edición: {cbz_path}")
        
        # Si es CBR, convertir a CBZ temporal
        working_path = cbz_path
        temp_file = None
        
        if cbz_path.lower().endswith('.cbr'):
            print("📦 Convirtiendo CBR a CBZ...")
            try:
                temp_file = self._convert_cbr_to_cbz(cbz_path)
                if temp_file:
                    working_path = temp_file
                    print("✅ Conversión exitosa")
                else:
                    messagebox.showerror("Error", "No se pudo convertir el archivo CBR", parent=self.root)
                    return
            except Exception as e:
                messagebox.showerror("Error", f"Error convirtiendo CBR:\\n{e}", parent=self.root)
                return
        
        # Abrir ventana de edición
        self.open_metadata_editor(working_path)
    
    
    def open_metadata_editor(self, cbz_path, file_list_context=None, current_index_context=None, search_query=None):
        try:
            editor = MetadataEditorWindow(
                self.root, 
                cbz_path, 
                self.status_var, 
                app_instance=self,
                file_list_context=file_list_context,
                current_index_context=current_index_context
            )
            if search_query:
                editor.cv_search_entry.delete(0, tk.END)
                editor.cv_search_entry.insert(0, search_query)
                editor.after(200, editor.start_precise_search)
            self.root.wait_window(editor)
            self.status_var.set("Listo.")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error Crítico", f"No se pudo abrir la ventana de edición.\n\nError: {e}")
        
# REEMPLAZA ESTE BLOQUE COMPLETO EN TU metaB.py


    def setup_reader_tab(self, tab):
        """Configura la pestaña del lector de cómics"""
        main_frame = ttk.Frame(tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame superior con botones
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(top_frame, text="📂 Abrir Cómic", 
                  command=self.reader_open_comic, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        
        self.reader_label = ttk.Label(top_frame, text="Selecciona un cómic para leer", 
                                     font=('Segoe UI', 10))
        self.reader_label.pack(side=tk.LEFT, padx=20)
        
        # Frame para el visor
        viewer_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        viewer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas para mostrar las páginas
        self.reader_canvas = tk.Canvas(viewer_frame, bg='#2b2b2b')
        self.reader_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame de controles de navegación
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)
        
        ttk.Button(controls_frame, text="⏮️ Primera", 
                  command=lambda: self.reader_go_to_page(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="◀️ Anterior", 
                  command=self.reader_prev_page).pack(side=tk.LEFT, padx=2)
        
        self.reader_page_label = ttk.Label(controls_frame, text="Página: - / -", 
                                          font=('Segoe UI', 10, 'bold'))
        self.reader_page_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(controls_frame, text="▶️ Siguiente", 
                  command=self.reader_next_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="⏭️ Última", 
                  command=lambda: self.reader_go_to_page(-1)).pack(side=tk.LEFT, padx=2)
        
        # Variables del lector
        self.reader_current_comic = None
        self.reader_pages = []
        self.reader_current_page = 0
        self.reader_images_cache = {}
    
    def reader_open_comic(self):
        """Abre un cómic para leer"""
        path = filedialog.askopenfilename(
            title="Seleccionar cómic",
            filetypes=[("Archivos de cómic", "*.cbz *.cbr"), ("Todos los archivos", "*.*")]
        )
        if not path:
            return
        
        try:
            # Limpiar cache anterior
            self.reader_images_cache = {}
            
            # Abrir el archivo
            if path.lower().endswith('.cbz'):
                zf = zipfile.ZipFile(path, 'r')
            else:  # CBR
                # Para CBR necesitaríamos rarfile, por ahora solo CBZ
                messagebox.showwarning("Formato no soportado", 
                                     "El lector actualmente solo soporta archivos CBZ.\n"
                                     "Puedes convertir CBR a CBZ desde Herramientas.")
                return
            
            # Obtener lista de imágenes
            self.reader_pages = sorted([f for f in zf.namelist() 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
                                      key=natural_sort_key)
            
            if not self.reader_pages:
                messagebox.showwarning("Sin páginas", "No se encontraron imágenes en el archivo.")
                return
            
            self.reader_current_comic = path
            self.reader_zipfile = zf
            self.reader_current_page = 0
            
            # Actualizar interfaz
            comic_name = os.path.basename(path)
            self.reader_label.config(text=f"📖 {comic_name}")
            self.reader_page_label.config(text=f"Página: 1 / {len(self.reader_pages)}")
            
            # Mostrar primera página
            self.reader_show_page(0)
            
            # Bind de teclas para navegación
            self.reader_canvas.focus_set()
            self.reader_canvas.bind('<Left>', lambda e: self.reader_prev_page())
            self.reader_canvas.bind('<Right>', lambda e: self.reader_next_page())
            self.reader_canvas.bind('<Home>', lambda e: self.reader_go_to_page(0))
            self.reader_canvas.bind('<End>', lambda e: self.reader_go_to_page(-1))
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el cómic:\n{e}")
            traceback.print_exc()
    
    def reader_show_page(self, page_index):
        """Muestra una página específica"""
        if not self.reader_current_comic or not self.reader_pages:
            return
        
        try:
            # Cargar imagen
            if page_index in self.reader_images_cache:
                img = self.reader_images_cache[page_index]
            else:
                with self.reader_zipfile.open(self.reader_pages[page_index]) as img_file:
                    img = Image.open(BytesIO(img_file.read()))
                    img.load()  # Cargar completamente
                    self.reader_images_cache[page_index] = img
            
            # Obtener dimensiones del canvas
            canvas_width = self.reader_canvas.winfo_width()
            canvas_height = self.reader_canvas.winfo_height()
            
            if canvas_width <= 1:  # Canvas aún no renderizado
                canvas_width = 800
                canvas_height = 1000
            
            # Escalar imagen manteniendo aspecto
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convertir a PhotoImage
            self.reader_photo = ImageTk.PhotoImage(img_resized)
            
            # Mostrar en canvas centrado
            self.reader_canvas.delete('all')
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.reader_canvas.create_image(max(x, 0), max(y, 0), 
                                          anchor=tk.NW, image=self.reader_photo)
            
            self.reader_current_page = page_index
            self.reader_page_label.config(
                text=f"Página: {page_index + 1} / {len(self.reader_pages)}"
            )
            
        except Exception as e:
            print(f"Error mostrando página: {e}")
            traceback.print_exc()
    
    def reader_next_page(self):
        """Ir a la siguiente página"""
        if self.reader_current_page < len(self.reader_pages) - 1:
            self.reader_show_page(self.reader_current_page + 1)
    
    def reader_prev_page(self):
        """Ir a la página anterior"""
        if self.reader_current_page > 0:
            self.reader_show_page(self.reader_current_page - 1)
    
    def reader_go_to_page(self, page_index):
        """Ir a una página específica"""
        if page_index == -1:
            page_index = len(self.reader_pages) - 1
        if 0 <= page_index < len(self.reader_pages):
            self.reader_show_page(page_index)


# ==============================================================================
# 5. SERVIDOR API (FLASK)
# ==============================================================================

def create_flask_app():
    """Crea y configura la aplicación Flask para el servidor API"""
    app = Flask(__name__)
    
    def get_image_from_cbz(comic_path, page_index):
        try:
            with zipfile.ZipFile(comic_path, 'r') as zf:
                image_list = sorted([f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))], key=natural_sort_key)
                if 0 <= page_index < len(image_list):
                    with zf.open(image_list[page_index]) as image_file:
                        return BytesIO(image_file.read())
        except Exception as e:
            print(f"Error extrayendo imagen: {e}")
        return None

    @app.route('/api/comic/<int:comic_id>/page/<int:page_num>', methods=['GET'])
    def get_comic_page(comic_id, page_num):
        conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
        cursor.execute("SELECT path FROM comics WHERE id = ?", (comic_id,)); result = cursor.fetchone(); conn.close()
        if not result: abort(404, description="Comic not found")
        image_io = get_image_from_cbz(result[0], page_num)
        if image_io: return send_file(image_io, mimetype='image/jpeg')
        else: abort(404, description="Page not found")

    @app.route('/api/comic/<int:comic_id>/cover', methods=['GET'])
    def get_comic_cover(comic_id):
        return get_comic_page(comic_id, 0)

    @app.route('/api/comic/<int:comic_id>/details', methods=['GET'])
    def get_comic_details(comic_id):
        conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("SELECT * FROM comics WHERE id = ?", (comic_id,)); comic_data = cursor.fetchone(); conn.close()
        if comic_data: return jsonify(dict(comic_data))
        else: abort(404, description="Comic details not found")

    @app.route('/api/reading_orders', methods=['GET'])
    def get_reading_orders():
        conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
        cursor.execute("SELECT id, name, description FROM reading_orders ORDER BY name")
        orders = [{'id': row[0], 'name': row[1], 'description': row[2]} for row in cursor.fetchall()]; conn.close()
        return jsonify(orders)

    @app.route('/api/reading_orders/<int:order_id>', methods=['GET'])
    def get_reading_order_comics(order_id):
        conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.series, c.number, c.title, c.path
            FROM reading_order_items roi JOIN comics c ON roi.comic_id = c.id
            WHERE roi.order_id = ? ORDER BY roi.sequence_number
        """, (order_id,))
        comics_data = [dict(row) for row in cursor.fetchall()]; conn.close()
        for comic in comics_data:
            try:
                with zipfile.ZipFile(comic['path'], 'r') as zf:
                    comic['page_count'] = len([f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            except Exception:
                comic['page_count'] = 0
        return jsonify(comics_data)
    
    return app

class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__()
        self.daemon = True
        self.server = make_server(host, port, app, threaded=True)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print("Iniciando servidor Flask...")
        self.server.serve_forever()

    def shutdown(self):
        print("Deteniendo servidor Flask...")
        self.server.shutdown()

# ==============================================================================
# 6. PUNTO DE ENTRADA DE LA APLICACIÓN
# ==============================================================================
if __name__ == "__main__":
    try:
        try:
            import ttkbootstrap as ttkb
            # Temas disponibles modernos y atractivos:
            # 'cosmo', 'flatly', 'litera', 'minty', 'lumen', 'sandstone', 'yeti', 'pulse', 'united', 'morph', 
            # 'journal', 'darkly', 'superhero', 'solar', 'cyborg', 'vapor', 'simplex', 'cerculean'
            # Usando 'darkly' para un aspecto moderno y oscuro elegante
            root = ttkb.Window(themename="darkly")
            print("✨ Tema moderno 'darkly' aplicado exitosamente")
        except ImportError:
            import tkinter as tk
            import tkinter.ttk as ttk
            root = tk.Tk()
            print("⚠️ Usando tema clásico (instala ttkbootstrap para interfaz moderna)")
        
        app = GestorApp(root)
        app.run()
    except Exception as e:
        print("\n--- ERROR FATAL DURANTE LA INICIALIZACIÓN ---")
        traceback.print_exc()
        input("\nPresiona Enter para salir.")