# run.py — Lanza legacy/metaB.py tal como si lo ejecutaras directamente.
import runpy, sys

def start():
    try:
        # Ejecuta el script original respetando su comportamiento
        runpy.run_module("legacy.metaB", run_name="__main__")
    except SystemExit:
        # Permite que el programa se cierre con normalidad
        raise
    except Exception as e:
        import traceback
        print("Error al iniciar la app:", e)
        traceback.print_exc()
        input("Pulsa Enter para salir...")

if __name__ == "__main__":
    start()
