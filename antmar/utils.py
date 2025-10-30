import re
import socket

def get_local_ip() -> str:
    """Devuelve la IP local sin depender de UI ni librerías externas."""
    ip = "127.0.0.1"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        pass
    finally:
        s.close()
    return ip


def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

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
    
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
        
import os

def scan_comics_folder(folder_path: str, exts=(".cbz", ".cbr", ".cb7", ".zip")) -> list[str]:
    """
    Escanea recursivamente una carpeta y devuelve una lista de archivos de cómic.
    Acepta CBZ, CBR, CB7, ZIP.
    """
    comics = []
    if not folder_path or not os.path.exists(folder_path):
        return comics

    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(exts):
                comics.append(os.path.join(root, f))
    return comics
