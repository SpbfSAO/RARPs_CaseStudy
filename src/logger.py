import logging
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_logger(source_file: str) -> logging.Logger:

    filename = os.path.basename(source_file)
    name, _ = os.path.splitext(filename)

    log_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

