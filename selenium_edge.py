import os
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.microsoft import EdgeChromiumDriverManager

EDGE_BINARY_PATH = None

def init_selenium_driver() -> webdriver.Edge:

    edge_options = EdgeOptions()
    edge_options.use_chromium = True
    edge_options.add_argument("--headless")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--window-size=1920,1080")

    if EDGE_BINARY_PATH:
        edge_options.binary_location = EDGE_BINARY_PATH

    service = EdgeService(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)
    return driver
