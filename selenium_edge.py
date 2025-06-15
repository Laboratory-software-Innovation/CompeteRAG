# selenium_helper.py

import os
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# ──────────────────────────────────────────────────────────────────────────────
# If you have an Edge binary in a nondefault location, set that path here.
# On Windows, Edge is usually in:
#    C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
# On macOS, Edge (if installed) lives in:
#    /Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge
#
# If you leave this as None, webdriver-manager will try to find Edge on PATH.
# ──────────────────────────────────────────────────────────────────────────────
EDGE_BINARY_PATH = None
# Example for Windows:
# EDGE_BINARY_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# Example for macOS:
# EDGE_BINARY_PATH = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"


def init_selenium_driver() -> webdriver.Edge:
    """
    Return a headless Edge WebDriver instance.
    If EDGE_BINARY_PATH is set, point EdgeOptions.binary_location there.
    Otherwise, rely on webdriver-manager to locate a matching msedgedriver.
    """
    edge_options = EdgeOptions()
    edge_options.use_chromium = True
    edge_options.add_argument("--headless")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--window-size=1920,1080")

    if EDGE_BINARY_PATH:
        edge_options.binary_location = EDGE_BINARY_PATH

    # webdriver-manager will download the correct msedgedriver for you
    service = EdgeService(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)
    return driver
