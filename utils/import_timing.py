"""Timed import helper for diagnosing cold RAG initialization."""

import importlib
import logging
import time
from types import ModuleType

logger = logging.getLogger("rag_import_timing")


def timed_import(module_name: str) -> ModuleType:
    started = time.perf_counter()
    logger.info("[RAG-IMPORT] %s START", module_name)
    try:
        return importlib.import_module(module_name)
    finally:
        logger.info("[RAG-IMPORT] %s END: %.3f sec", module_name, time.perf_counter() - started)