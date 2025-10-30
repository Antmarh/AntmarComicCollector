import os, logging
from datetime import datetime

def setup_logging(app_name="AntmarComicCollector"):
    base = os.path.join(os.getenv("APPDATA", "."), app_name, "logs")
    os.makedirs(base, exist_ok=True)
    log_file = os.path.join(base, f"{datetime.now():%Y-%m-%d}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    )
    logging.getLogger(__name__).info("Logs en %s", base)
