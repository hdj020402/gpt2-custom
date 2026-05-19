import importlib.util
import os
import logging

logger = logging.getLogger(__name__)

def load_custom_module(path: str | None):
    if not path:
        return None

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Custom module not found: {path}")

    spec = importlib.util.spec_from_file_location("custom_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info(f"Loaded custom module from {path}")
    return module
