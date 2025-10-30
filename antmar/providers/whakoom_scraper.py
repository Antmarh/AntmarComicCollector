# whakoom_scraper.py (Versión actualizada y fusionada)
import requests
from bs4 import BeautifulSoup
import re
import html

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}

def parse_spanish_date(date_str):
    """Convierte fechas como 'Marzo de 2021' o '15/03/2021' a (año, mes, día)."""
    if not date_str: return None, None, None
    
    # Intentar formato DD/MM/YYYY
    match_numeric = re.search(r'(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{4})', date_str)
    if match_numeric:
        return int(match_numeric.group(3)), int(match_numeric.group(2)), int(match_numeric.group(1))

    # Intentar formato "Mes de Año"
    parts = date_str.lower().split(' de ')
    if len(parts) == 2:
        month_str, year_str = parts[0], parts[1]
        months = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        year = int(re.search(r'\d{4}', year_str).group())
        month = months.get(month_str.strip())
        return year, month, None # Día es desconocido en este formato
        
    return None, None, None

def get_whakoom_details(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        metadata = {'Web': url, 'ScanInformation': 'Scraped from Whakoom'}

        # --- Título, Serie y Número ---
        title_el = soup.select_one('h1[itemprop="name"]')
        if not title_el:
            title_el = soup.select_one('h1')
        
        if title_el:
            # Buscar el span (serie) y strong (número)
            series_span = title_el.find('span')
            number_strong = title_el.find('strong')
            
            if series_span:
                metadata['Series'] = series_span.get_text(strip=True)
            
            if number_strong:
                num_text = number_strong.get_text(strip=True)
                # Quitar el # si existe
                metadata['Number'] = num_text.replace('#', '').strip()
            
            # Si no encontramos la estructura, intentar parsear todo el texto
            if 'Series' not in metadata:
                full_title = title_el.get_text(strip=True)
                match = re.match(r'^(.*?)(\s+#?(\d+(\.\d+)?))$', full_title.strip())
                if match:
                    metadata['Series'] = match.group(1).strip()
                    metadata['Number'] = match.group(3).strip()
                else:
                    metadata['Series'] = full_title
            
            # El título específico del número (buscar subtítulo si existe)
            subtitle_el = soup.select_one('h2')
            if subtitle_el:
                subtitle_text = subtitle_el.get_text(strip=True)
                if subtitle_text and subtitle_text != metadata.get('Series', '') and 'Información' not in subtitle_text:
                    metadata['Title'] = subtitle_text
            
            # Si no hay título específico, usar Serie #Número
            if 'Title' not in metadata and 'Series' in metadata and 'Number' in metadata:
                metadata['Title'] = f"{metadata['Series']} #{metadata['Number']}"
            elif 'Title' not in metadata and 'Series' in metadata:
                metadata['Title'] = metadata['Series']

        # --- Editorial ---
        publisher_el = soup.select_one('[itemprop="publisher"] [itemprop="name"]')
        if publisher_el:
            metadata['Publisher'] = publisher_el.get_text(strip=True)

        # --- Fecha de publicación ---
        date_el = soup.select_one('[itemprop="datePublished"]')
        if date_el and date_el.get('content'):
            date_str = date_el.get('content')
            # Formato ISO: 2025-05-08
            match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
            if match:
                metadata['Year'] = match.group(1)
                metadata['Month'] = str(int(match.group(2)))  # Quitar cero inicial
                metadata['Day'] = str(int(match.group(3)))

        # --- Idioma ---
        lang_el = soup.select_one('[itemprop="inLanguage"]')
        if lang_el:
            metadata['Language'] = lang_el.get_text(strip=True)

        # --- Autores (buscar en info-item) ---
        info_items = soup.select('div.info-item')
        roles = {'Writer': [], 'Penciller': [], 'Inker': [], 'Colorist': [], 'Letterer': [], 'CoverArtist': [], 'Editor': []}
        
        for item in info_items:
            h3 = item.find('h3')
            if not h3:
                continue
            
            role_text = h3.get_text(strip=True).lower()
            authors = [a.get_text(strip=True) for a in item.select('p a')]
            
            if 'guion' in role_text or 'script' in role_text:
                roles['Writer'].extend(authors)
            elif 'dibujo' in role_text or 'pencil' in role_text or 'art' in role_text:
                roles['Penciller'].extend(authors)
            elif 'tinta' in role_text or 'ink' in role_text:
                roles['Inker'].extend(authors)
            elif 'color' in role_text:
                roles['Colorist'].extend(authors)
            elif 'rotulación' in role_text or 'letter' in role_text:
                roles['Letterer'].extend(authors)
            elif 'portada' in role_text or 'cover' in role_text:
                roles['CoverArtist'].extend(authors)
            elif 'editor' in role_text:
                roles['Editor'].extend(authors)
        
        # Agregar roles con autores
        for role, names in roles.items():
            if names:
                metadata[role] = ", ".join(names)

        # --- Resumen/Argumento ---
        # Buscar en div.wiki-text
        wiki_text = soup.select_one('div.wiki-text')
        if wiki_text:
            # Buscar el h3 "Argumento" y obtener los párrafos siguientes
            arg_h3 = wiki_text.find('h3', string=lambda s: s and 'Argumento' in s)
            if arg_h3:
                summary_parts = []
                for sibling in arg_h3.find_next_siblings('p'):
                    text = sibling.get_text(strip=True)
                    if text:
                        summary_parts.append(text)
                
                if summary_parts:
                    metadata['Summary'] = '\n\n'.join(summary_parts)

        # --- Notas adicionales (ISBN, páginas, etc.) ---
        notes_parts = []
        for item in info_items:
            h3 = item.find('h3')
            if not h3:
                continue
            
            label = h3.get_text(strip=True).lower()
            p = item.find('p')
            if not p:
                continue
            
            value = p.get_text(strip=True)
            
            if 'isbn' in label:
                notes_parts.append(f"ISBN: {value}")
            elif 'páginas' in label:
                notes_parts.append(f"Páginas: {value}")
            elif 'formato' in label:
                metadata['Format'] = value
        
        if notes_parts:
            metadata['Notes'] = " | ".join(notes_parts)

        # Devolvemos solo los campos que tengan valor
        return {k: v for k, v in metadata.items() if v}

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"No se pudo conectar a Whakoom: {e}")
    except Exception as e:
        # Importante para depuración: mostramos el traceback del error original
        import traceback
        traceback.print_exc()
        raise ValueError(f"Error al parsear la página de Whakoom: {e}")