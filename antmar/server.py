def create_flask_app():
    """
    Función que crea y configura la aplicación Flask.
    Esto es crucial para que PyInstaller funcione correctamente.
    """
    app = Flask(__name__)

    @app.route('/api/library', methods=['GET'])
    def get_library():
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, series, number, title, path FROM comics ORDER BY series, CAST(number AS REAL)")
        comics_data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        for comic in comics_data:
            try:
                with zipfile.ZipFile(comic['path'], 'r') as zf:
                    comic['page_count'] = len([f for f in zf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            except Exception:
                comic['page_count'] = 0
        return jsonify(comics_data)
    
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
