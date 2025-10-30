\# 🕹️ Antmar Comic Collector



\*\*Antmar Comic Collector\*\* es un gestor avanzado de cómics digitales creado por \*\*Antonio María (Antmar)\*\*, escrito en \*\*Python + Tkinter\*\*, diseñado para organizar, completar y editar metadatos de cómics \*\*CBZ/CBR/CB7\*\*.



Permite trabajar con tu colección sin conexión, automatizar metadatos desde \*\*ComicVine\*\* y \*\*Whakoom\*\*, y generar archivos \*\*ComicInfo.xml\*\* compatibles con los principales lectores.



---



\## 🚀 Características principales



\### 📦 Gestión de archivos

\- Abre y analiza cómics `.cbz`, `.cbr` y `.cb7`.

\- Extrae automáticamente la \*\*portada\*\*.

\- Lee y escribe archivos \*\*ComicInfo.xml\*\* dentro del cómic.

\- Genera o reescribe el ComicInfo conservando los datos existentes.



\### 🧩 Metadatos automáticos

\- \*\*ComicVine API\*\* → completa título, número, fecha, autores, etc.

\- \*\*Whakoom Scraper\*\* → al pegar la URL de un cómic, extrae los metadatos directamente.

\- \*\*DeepL API (opcional)\*\* → traduce automáticamente descripciones y campos al español.



\### 🧰 Edición avanzada

\- \*\*Editor de metadatos por lotes\*\* (ventana dedicada).

\- Normalización de campos (Series, Título, Número…).

\- Reordenación natural por nombre o número de cómic.

\- Filtrado y búsqueda rápida de archivos.



\### 🧠 Automatización

\- Carga los metadatos, genera el XML y lo inyecta en el CBZ con un clic.

\- Guarda las claves API una sola vez (en `config.ini`) y las reutiliza automáticamente.

\- Crea logs diarios en `%APPDATA%\\AntmarComicCollector\\logs`.



\### 🎨 Interfaz moderna

\- Basada en \*\*Tkinter + ttkbootstrap\*\* (modo claro/oscuro).

\- Ventanas flotantes con iconos personalizados.

\- Soporte para imágenes \*\*Pillow (WebP, PNG, JPG)\*\*.



\### 🌐 Funciones en línea

\- Integración directa con \*\*ComicVine\*\* (búsqueda por nombre o ID).

\- Scraper de \*\*Whakoom\*\* que convierte cualquier ficha pública en ComicInfo.xml.

\- Traductor opcional con \*\*DeepL\*\* (si introduces tu API key).



\### 🔧 Utilidades internas

\- Conversión de imágenes.

\- Generador de nombres naturales (orden correcto: 1, 2, 10…).

\- Obtención automática de IP local (para servidor Flask opcional).

\- Sistema de actualización: avisa si hay una versión nueva en GitHub.



---



\## 🧭 Cómo usarlo



1\. \*\*Abre el programa\*\* (`AntmarComicCollector.exe` o `python run.py`).

2\. \*\*Carga un cómic (.cbz)\*\* → se mostrará su portada y metadatos.

3\. Si el cómic no tiene ComicInfo.xml:

&nbsp;  - Pulsa “\*\*ComicVine\*\*” o “\*\*Whakoom\*\*” para completarlo.

&nbsp;  - La primera vez se te pedirá tu API key → se guarda para siempre.

4\. Pulsa \*\*“Guardar ComicInfo”\*\* → el XML se genera e inserta dentro del CBZ.

5\. Usa el \*\*editor por lotes\*\* para actualizar varios cómics a la vez.



---



\## 🧠 Requisitos (modo código)



```bash

pip install -r requirements.txt



