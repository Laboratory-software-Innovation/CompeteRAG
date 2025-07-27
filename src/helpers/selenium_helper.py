from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

CHROME_BINARY_PATH = None

def init_selenium_driver() -> webdriver.Chrome:
    """
    Return a headless Chrome WebDriver instance.
    If CHROME_BINARY_PATH is set, points ChromeOptions.binary_location there.
    Otherwise relies on webdriver-manager to download chromedriver.
    """
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    if CHROME_BINARY_PATH:
        chrome_options.binary_location = CHROME_BINARY_PATH

    service = ChromeService(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

