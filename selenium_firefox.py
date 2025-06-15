import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager

# If Firefox is in a nonstandard location, set that here.
# Otherwise leave as None and let webdriver-manager find it on PATH.
FIREFOX_BINARY_PATH = None
# Example (Ubuntu DEB): "/usr/bin/firefox"
# Example (custom build): "/opt/firefox/firefox"

def init_selenium_driver() -> webdriver.Firefox:
    """
    Return a headless Firefox WebDriver instance.
    If FIREFOX_BINARY_PATH is set, point FirefoxOptions.binary_location there.
    Otherwise rely on webdriver-manager to install geckodriver.
    """
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")
    firefox_options.add_argument("--window-size=1920,1080")

    if FIREFOX_BINARY_PATH:
        firefox_options.binary_location = FIREFOX_BINARY_PATH

    service = FirefoxService(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=firefox_options)
    return driver

