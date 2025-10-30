# antmar/ui/batch_editor.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
from PIL import Image, ImageTk

# Imports del proyecto (ajusta si los usas dentro de la ventana)
from antmar.utils import natural_sort_key, get_local_ip
from antmar.metadata import generate_comicinfo_xml
from antmar.cbz import read_comicinfo_from_cbz, inject_xml_into_cbz, get_cover_from_cbz
from antmar.providers.comicvine import get_comicvine_details
from antmar.providers.translate import get_translator

class BatchMetadataEditorWindow(tk.Toplevel):
    def __init__(self, parent, app_instance):
        super().__init__(parent)
        self.app = app_instance
        self.db_file = self.app.db_file
        
        self.title("Editor de Metadatos por Lote")
        self.geometry("900x700")
        self.transient(parent)
        self.grab_set()

        # --- CAMPOS EDITABLES ---
        # Lista completa de campos comunes de ComicInfo.xml
        self.ALL_METADATA_FIELDS = [
            "Series", "Number", "Count", "Volume", "AlternateSeries", "AlternateNumber", "AlternateCount",
            "Title", "Publisher", "Imprint", "Day", "Month", "Year", "LanguageISO",
            "Writer", "Penciller", "Inker", "Colorist", "Letterer", "CoverArtist", "Editor",
            "Summary", "Notes", "Web", "PageCount", "Genre", "Tags", "StoryArc",
            "Characters", "Teams", "Locations", "ScanInformation"
        ]

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=3)

        # --- Panel de selección de cómics (Izquierda) ---
        list_frame = ttk.LabelFrame(main_frame, text="1. Selecciona los cómics a modificar")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)

        search_frame = ttk.Frame(list_frame, padding=5)
        search_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(search_frame, text="Buscar:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_entry.bind("<KeyRelease>", self.filter_comic_list)
        
        self.comic_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.comic_listbox.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.comic_listbox.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.comic_listbox.config(yscrollcommand=scrollbar.set)
        
        self.load_comics_from_library()

        # --- Panel de acciones (Derecha) ---
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(1, weight=1)

        actions_frame = ttk.LabelFrame(right_panel, text="2. Define los cambios")
        actions_frame.pack(fill=tk.X, expand=False)
        
        self.changes = [] # Lista para guardar los cambios a aplicar
        
        add_change_frame = ttk.Frame(actions_frame, padding=5)
        add_change_frame.pack(fill=tk.X)
        add_change_frame.columnconfigure(1, weight=1)
        
        # ComboBox con todos los campos
        self.field_combo = ttk.Combobox(add_change_frame, values=sorted(self.ALL_METADATA_FIELDS), state="readonly")
        self.field_combo.grid(row=0, column=0, padx=(0, 5))
        self.field_combo.set("Tags")
        
        self.new_value_entry = ttk.Entry(add_change_frame)
        self.new_value_entry.grid(row=0, column=1, sticky="ew", padx=5)
        
        add_btn = ttk.Button(add_change_frame, text="Añadir Cambio ↓", command=self.add_change_to_list)
        if MODERN_UI:
            add_btn.config(bootstyle=INFO)
        add_btn.grid(row=0, column=2, padx=5)

        # Treeview para mostrar la lista de cambios
        changes_list_frame = ttk.LabelFrame(right_panel, text="Cambios a aplicar")
        changes_list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        cols = ("Campo", "Nuevo Valor")
        self.changes_tree = ttk.Treeview(changes_list_frame, columns=cols, show='headings', height=5)
        self.changes_tree.heading("Campo", text="Campo")
        self.changes_tree.heading("Nuevo Valor", text="Nuevo Valor")
        self.changes_tree.column("Campo", width=150)
        self.changes_tree.pack(fill=tk.BOTH, expand=True)
        self.changes_tree.bind("<Delete>", self.remove_selected_change) # Permitir borrar con la tecla Supr

        # --- Panel de ejecución (Abajo) ---
        execute_frame = ttk.Frame(self)
        execute_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_bar = ttk.Progressbar(execute_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.apply_btn = ttk.Button(execute_frame, text="Aplicar Cambios a Cómics Seleccionados", command=self.start_batch_edit)
        if MODERN_UI:
            self.apply_btn.config(bootstyle=SUCCESS)
        self.apply_btn.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(execute_frame, textvariable=self.status_var).pack(anchor="w", pady=5)
        
    def add_change_to_list(self):
        field = self.field_combo.get()
        value = self.new_value_entry.get() # No usamos strip() para permitir espacios si el usuario quiere
        if not field:
            return
        
        # Evitar duplicados
        for item in self.changes_tree.get_children():
            if self.changes_tree.item(item, "values")[0] == field:
                messagebox.showwarning("Campo duplicado", f"Ya has añadido un cambio para el campo '{field}'.\nElimina el anterior si quieres cambiarlo.", parent=self)
                return
        
        self.changes_tree.insert("", tk.END, values=(field, value))
        self.new_value_entry.delete(0, tk.END)

    def remove_selected_change(self, event=None):
        selected_items = self.changes_tree.selection()
        for item in selected_items:
            self.changes_tree.delete(item)

    def load_comics_from_library(self):
        self.comic_map = {f"{comic['series'] or 'Sin Serie'} #{comic['number'] or 'S/N'} ({os.path.basename(comic['path'])})": comic['path'] for comic in self.app.library_data}
        self.all_comics_display = sorted(self.comic_map.keys(), key=natural_sort_key)
        self.filter_comic_list()

    def filter_comic_list(self, event=None):
        query = self.search_entry.get().lower()
        self.comic_listbox.delete(0, tk.END)
        for name in self.all_comics_display:
            if query in name.lower():
                self.comic_listbox.insert(tk.END, name)

    def start_batch_edit(self):
        selected_indices = self.comic_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Sin selección", "Selecciona al menos un cómic de la lista.", parent=self)
            return

        changes_to_apply = [self.changes_tree.item(item, "values") for item in self.changes_tree.get_children()]
        if not changes_to_apply:
            messagebox.showwarning("Sin cambios", "Añade al menos un cambio a la lista.", parent=self)
            return

        comics_to_process = [self.comic_listbox.get(i) for i in selected_indices]
        
        changes_str = "\n".join([f"- Cambiar '{field}' a '{value}'" for field, value in changes_to_apply])
        if not messagebox.askyesno("Confirmar", 
                                   f"Vas a aplicar los siguientes cambios a {len(comics_to_process)} cómics:\n\n"
                                   f"{changes_str}\n\n¿Estás seguro?", parent=self):
            return

        self.apply_btn.config(state=tk.DISABLED)
        self.progress_bar['maximum'] = len(comics_to_process)
        
        threading.Thread(target=self._batch_edit_thread, 
                         args=(comics_to_process, changes_to_apply), daemon=True).start()

    def _batch_edit_thread(self, display_names, changes):
        total = len(display_names)
        for i, name in enumerate(display_names):
            path = self.comic_map.get(name)
            if not path: continue
            
            self.after(0, self.status_var.set, f"Procesando ({i+1}/{total}): {os.path.basename(path)}")
            self.after(0, self.progress_bar.config, {'value': i + 1})

            try:
                # 1. Leer metadatos existentes (o crear un diccionario vacío)
                metadata = read_comicinfo_from_cbz(path) or {}
                
                # 2. Aplicar todos los cambios definidos
                for field, new_value in changes:
                    metadata[field] = new_value
                
                # 3. Guardar el XML en el CBZ
                xml_string = generate_comicinfo_xml(metadata)
                inject_xml_into_cbz(path, xml_string)
                
                # 4. Actualizar la base de datos
                # Reutilizamos la función ya optimizada, pero no necesita mostrar popups
                self.app.add_or_update_comic_in_db_batch(path, metadata)
                
            except Exception as e:
                print(f"Error al procesar {path}: {e}")
                traceback.print_exc()

        self.after(0, self.finish_batch_edit)

    def finish_batch_edit(self):
        messagebox.showinfo("Proceso completado", "La edición por lote ha finalizado.", parent=self)
        self.status_var.set("Listo.")
        self.apply_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.app.refresh_library_view()
        self.destroy()
        
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Editor de metadatos (lote)")
        self.master = master
        self._build_ui()

    def _build_ui(self):
        # Crea aquí el layout básico; luego pega tu implementación real
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="(UI cargada desde antmar/ui/batch_editor.py)").pack(anchor="w")

