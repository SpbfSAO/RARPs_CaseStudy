import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
MAIN_CFG_PATH = os.path.join(CONFIG_DIR, "main_config.yaml")
AE_CFG_PATH = os.path.join(CONFIG_DIR, "ae_config.yaml")
MODELLING_CFG_PATH = os.path.join(CONFIG_DIR, "modelling_config.yaml")

BASE_DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DATA_DIR, "interim")
EXTERNAL_DATA_DIR = os.path.join(BASE_DATA_DIR, "external")
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, "processed")

REPORTS_DIR = os.path.join(ROOT_DIR, "output", "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "figures")

AE_OUTPUTS = os.path.join(ROOT_DIR, "output", "ae")

def create_dirs():
    """Creates all necessary directories if they do not exist."""
    dirs = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        EXTERNAL_DATA_DIR,
        PROCESSED_DATA_DIR,
        AE_OUTPUTS,
        REPORTS_DIR,
        PLOTS_DIR,
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
