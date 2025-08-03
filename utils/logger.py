
import os
from datetime import datetime
import logging


def init_logging(name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, name)

    # --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
                logging.FileHandler(LOG_PATH, mode='a'),
                logging.StreamHandler()
            ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized. Log file: {LOG_PATH}")

    return logger

def setup_logger(name, log_file="evaluation.log"):
    """Configure file and console logging"""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(f"./logs/{log_file}")
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger