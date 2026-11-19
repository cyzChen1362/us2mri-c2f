"""Data locations.

All paths are read from environment variables so that no machine-specific
location is stored in the code. ReMIND must be downloaded from TCIA and
pre-processed (trajectory slicing, see ``c2f/trajectory``) before use.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get('DATA_ROOT', './data'))          # processed ReMIND root
RAW = DATA_ROOT / 'raw'                                          # original NIfTI volumes
CACHE = Path(os.environ.get('C2F_CACHE', DATA_ROOT / 'cache'))   # prepared model inputs
OUTPUT = Path(os.environ.get('C2F_OUTPUT', './output'))          # experiment outputs
WEIGHTS = Path(os.environ.get('C2F_WEIGHTS', './weights'))       # pretrained backbones


def case_cache_path(modality, case):
    """Folder holding the prepared inputs of one patient for one MRI branch."""
    return CACHE / modality / case
