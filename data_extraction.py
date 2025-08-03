import requests
from bs4 import BeautifulSoup
import pandas as pd

from fpdf import FPDF

def get_page_content(url):
    """
    Fetches the content of a webpage.

    Args:
    url (str): The URL of the webpage.

    Returns:
    str: The raw HTML content of the webpage.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to retrieve content from {url}")
        return None

def parse_content(html_content):
    """
    Parses the HTML content to extract required information.

    Args:
    html_content (str): The raw HTML content of the webpage.

    Returns:
    dict: Extracted information from the webpage.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {}

    sections = {
        'Overview': 'Overview',
        'Symptoms': 'Symptoms',
        'Causes': 'Causes',
        'Risk factors': 'Risk factors',
        'Complications': 'Complications',
        'Prevention': 'Prevention',
        'Diagnosis': 'Diagnosis',
        'Treatment': 'Treatment'
    }

    for key, value in sections.items():
        section = soup.find('h2', text=value)
        if section:
            if key in ['Symptoms', 'Risk factors', 'Complications', 'Prevention', 'Diagnosis', 'Treatment']:
                list_items = section.find_next('ul').find_all('li')
                data[key] = [li.get_text() for li in list_items]
            else:
                data[key] = section.find_next('p').get_text()

    return data

def navigate_and_scrape(url):
    """
    Navigates to a relative URL and scrapes the content.

    Args:
    base_url (str): The base URL of the website.
    relative_url (str): The relative URL to navigate to.

    Returns:
    dict: Extracted information from the navigated webpage.
    """
    
    html_content = get_page_content(url)
    if html_content:
        page_data = parse_content(html_content)
        return page_data
    return None

def scrape_multiple_pages(urls):
    """
    Scrapes multiple pages and extracts information.

    Args:
    urls (list): List of URLs to scrape.

    Returns:
    list: List of dictionaries containing extracted information from each webpage.
    """
    all_data = []
    for url in urls:
        html_content = get_page_content(url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            heading = {'Type': soup.find('a', href=url[url.find('/diseases'):]).get_text()}
            tab_navigation = soup.find('div', id = 'access-nav')
            if tab_navigation:
                links = tab_navigation.find_all('a')
                if len(links) > 1:
                    second_link = links[1]['href']
                    base_url = 'https://www.mayoclinic.org'  # Adjust the base URL as needed
                    full_url = base_url + second_link
                    #page_data = navigate_and_scrape(base_url, full_url)
                    first_page_data = navigate_and_scrape(url)
                    second_page_data = navigate_and_scrape(full_url)
                    # Merge the dictionaries
                    merged_dict = {**heading, **first_page_data, **second_page_data}
                    if merged_dict:
                        all_data.append(merged_dict)
    return all_data

def save_to_pdf(data, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for entry in data:
        for key, value in entry.items():
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(200, 10, txt=key.encode('latin1', 'replace').decode('latin1'), ln=True, align='L')
            pdf.set_font("Arial", size=12)
            if isinstance(value, list):
                for item in value:
                    pdf.multi_cell(0, 10, txt=item.encode('latin1', 'replace').decode('latin1'))
            else:
                pdf.multi_cell(0, 10, txt=value.encode('latin1', 'replace').decode('latin1'))
            pdf.ln(5)
        pdf.add_page()

    pdf.output(filename)

urls = [
    'https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444',
    'https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193',
    'https://www.mayoclinic.org/diseases-conditions/gestational-diabetes/symptoms-causes/syc-20355339',
    'https://www.mayoclinic.org/diseases-conditions/type-1-diabetes-in-children/symptoms-causes/syc-20355306',
    'https://www.mayoclinic.org/diseases-conditions/type-1-diabetes/symptoms-causes/syc-20353011',
    'https://www.mayoclinic.org/diseases-conditions/type-2-diabetes-in-children/symptoms-causes/syc-20355318',
    'https://www.mayoclinic.org/diseases-conditions/hyperglycemia/symptoms-causes/syc-20373631',
    'https://www.mayoclinic.org/diseases-conditions/diabetes-insipidus/symptoms-causes/syc-20351269',
    'https://www.mayoclinic.org/diseases-conditions/diabetic-nephropathy/symptoms-causes/syc-20354556',
    'https://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611',
    
]

data = scrape_multiple_pages(urls)

save_to_pdf(data, 'diabetes_faq.pdf')