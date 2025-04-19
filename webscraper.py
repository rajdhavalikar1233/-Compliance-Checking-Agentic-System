import os
import time
import random
import re
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def sanitize_domain(url):
    """Sanitize domain name to be filesystem-safe and consistent."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.replace("www.", "").strip()
        domain = re.sub(r'[<>:"/\\|?*]', '', domain)
        if not domain:
            domain = "unknown_domain"
        return domain
    except Exception:
        return "unknown_domain"


def get_policy_links(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 122)}.0.0.0 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.find_all("a", href=True)

        policy_links = {"Terms of Use": None, "Privacy Policy": None, "Cookie Policy": None}

        for link in links:
            href = link.get("href")
            text = link.get_text().strip().lower()

            if "terms" in text or "terms" in href:
                policy_links["Terms of Use"] = urljoin(url, href)
            elif "privacy" in text or "privacy" in href:
                policy_links["Privacy Policy"] = urljoin(url, href)
            elif "cookie" in text or "cookie" in href:
                policy_links["Cookie Policy"] = urljoin(url, href)

        return policy_links
    except Exception as e:
        print(f"Error fetching policy links from {url}: {e}")
        return None
    finally:
        driver.quit()

def scrape_policy(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 122)}.0.0.0 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
    finally:
        driver.quit()

def main(website_url):
    if not website_url.startswith("http"):
        website_url = "https://" + website_url

    domain = sanitize_domain(website_url)

    output_dir = os.path.join("data", "scraped_policies", domain)
    os.makedirs(output_dir, exist_ok=True)

    policies = get_policy_links(website_url)

    output_file = os.path.join(output_dir, f"{domain}_Policies.txt")

    with open(output_file, "w", encoding="utf-8") as file:
        for policy, link in policies.items():
            if link:
                print(f"Scraping {policy} from: {link}")
                policy_text = scrape_policy(link)
                if policy_text:
                    file.write(f"===== {policy} =====\n{policy_text}\n\n")
                else:
                    file.write(f"===== {policy} =====\nCould not fetch content.\n\n")

    print(f"Scraping complete! Data saved in '{output_file}'.")
    return output_file

if __name__ == "__main__":
    url = input("Enter the website URL (e.g., https://www.amazon.in): ").strip()
    main(url)
