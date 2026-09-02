"""
Step 0: Shared logger used by every other step. 

Every module in this app asks this file for a logger instead of 
setting up its own logger. That way all logs (from documents loading to the final output) 
are in one place and can be configured in one place.

"""

import logging
import sys
import os
from datetime import datetime

LOGS_DIR = "logs"

# Keep each run inside a directory for the day it started.
_run_started_at = datetime.now()
_run_date = _run_started_at.strftime("%Y-%m-%d")
_run_timestamp = _run_started_at.strftime("%Y-%m-%d_%H-%M-%S-%f")
daily_logs_dir = os.path.join(LOGS_DIR, _run_date)
os.makedirs(daily_logs_dir, exist_ok=True)
log_file_path = os.path.join(daily_logs_dir, f"run_{_run_timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the given name.
    """
    return logging.getLogger(name)