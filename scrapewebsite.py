import asyncio
import json
import re
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from openai import OpenAI
import logging
from rapidfuzz import process
from sqlalchemy import create_engine, literal,Integer , Numeric, Column, String,and_, or_,func,Boolean, DateTime, Date,ForeignKey, VARCHAR, Float,desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os
import random
from random import choice
import aiohttp
from decimal import Decimal
#from playwright_stealth.stealth_async import stealth_async


openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

DB_EALA_USER = os.getenv('DB_EALA_USER')
DB_EALA_PASSWORD = os.getenv('DB_EALA_PASSWORD')
#database_url = "postgresql://"+DB_RIIYO_USER+":"+DB_RIIYO_PASSWORD+"@ep-round-thunder-a4bd4scl.us-east-1.aws.neon.tech/neondb?sslmode=require&options=endpoint%3Dep-round-thunder-a4bd4scl"
database_url = 'postgresql://'+DB_EALA_USER+':' +DB_EALA_PASSWORD +'@ep-round-thunder-a4bd4scl.us-east-1.aws.neon.tech/neondb?sslmode=require&options=endpoint%3Dep-round-thunder-a4bd4scl'#os.getenv('DATABASE_URL')

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_size=10,       # Keep 10 connections open
    max_overflow=20,    # Allow 20 extra connections if needed
    pool_timeout=30,    # Wait 30s before failing
    pool_recycle=300,  # Refresh connections every 30 minutes
    echo=False
)
Base = declarative_base()

SessionFactory  = sessionmaker(bind=engine)
session = scoped_session( SessionFactory )

class Retailers(Base):
    __tablename__ = "retailers"
    id = Column(Integer, primary_key=True)
    company_name = Column(String)
    company_information = Column(String)
    phone_number = Column(String)
    email_address = Column(String)
    address =  Column(String)
    website =  Column(String)
    supplier_link =  Column(String)
    organization_number =  Column(String)
    updated=Column(Boolean)
    logo=Column(String)
    revenue=Column(String)
    revenue_currency= Column(String)
    assortment_link=Column(String)
    instagram=Column(String)
    sales_channel = Column(String)
    nr_of_stores = Column(String)
    nr_of_brands = Column(String)
    revenue_size = Column(String)
    keywords = Column(String)
    total_scraping_cost =  Column(Numeric(10, 6))
    owned_by_userid = Column(Integer)
    created_by_vendorid = Column(Integer)

class Countries(Base):
    __tablename__ = "countries"
    id = Column(Integer, primary_key=True)
    country_name_english = Column(String)
    country_name_swedish = Column(String)
    country_code = Column(String)
    continent = Column(String)



TARGET_FIELDS = ["company_name","phone_number", "email_address", "types", "categories", "countries"]

def is_done(data):
    return all(data.get(field) for field in TARGET_FIELDS)

def merge_data(existing, new):
    for field in TARGET_FIELDS:
        if field not in existing or not existing[field]:
            existing[field] = new.get(field, [])
        elif isinstance(existing[field], list) and isinstance(new.get(field), list):
            existing[field] = list(set(existing[field] + new[field]))
    return existing


def extract_links(html, base_url, mode):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    priority_links = []

    for a in soup.find_all("a", href=True):
        raw_href = a["href"].strip()
        href = urljoin(base_url, raw_href)
        href_lower = href.lower()

        # Skip non-http links
        if any(href_lower.startswith(p) for p in ["mailto:", "tel:", "javascript:", "#"]):
            continue

        # Priority logic depending on scraping purpose
        if mode == "contact":
            if any(k in href_lower for k in ["contact", "kontakt", "about", "om-oss", "kontakta"]):
                priority_links.append(href)
            else:
                links.add(href)

        elif mode == "content":
            if any(k in href_lower for k in ["about", "om-oss", "foretag", "company", "info", "who-we-are"]):
                priority_links.append(href)
            else:
                links.add(href)

        else:
            # fallback: add all valid HTTP links
            links.add(href)

    # Ensure priority links are not duplicated
    return priority_links + list(links - set(priority_links))


def clean_openai_json(text):
    if text.startswith("```json") or text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text

def extract_structured_fields(text,country_name_map):
    type_ids = {
        "Pharmacy": 1, "Salon": 2, "Clinic": 3, "Retailer": 4, "Distributor": 5,
        "Wholesaler": 6, "Spa": 7, "Grocery/Supermarket": 8, "Marketplace": 9
    }
    category_ids = {
        "Cosmetics": 1, "Skin care": 2, "Hair care": 3, "Nail care": 4,
        "Perfume": 5, "Foot care": 6, "Body care": 7, "Make-up": 8, "Supplements": 10
    }

    type_map = ", ".join(f"{k}={v}" for k, v in type_ids.items())
    category_map = ", ".join(f"{k}={v}" for k, v in category_ids.items())

    prompt = f"""
You are a data extraction AI. From the following text, extract:
- Company name
- One email
- One phone number
- Country or countries where the company sells
- Business types (choose from: {type_map})
- Product categories (choose from: {category_map})

Return as raw JSON with keys:"company_name", "email_address", "phone_number", "countries", "types", "categories".
If you can't find a field, return an empty string or array.
TEXT:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return {}


    reply = clean_openai_json(response.choices[0].message.content.strip())
    usage=response.usage

    prompt_tokens  =usage.prompt_tokens
    completion_tokens =usage.completion_tokens
    total_tokens =usage.total_tokens
    input_cost = (usage.prompt_tokens / 1000) * 0.005
    output_cost = (usage.completion_tokens / 1000) * 0.015
    total_cost = input_cost + output_cost

    logging.info(f"Total tokens prompt1:{total_tokens},total cost:{total_cost},prompt_tokens:{prompt_tokens}, input cost: {input_cost},completion_tokens:{completion_tokens} ,output cost:{output_cost} ")

    try:
        structured = json.loads(reply)
        scraped_countries = structured.get("countries", [])
        country_ids = []
        for name in scraped_countries:
            best_match, score, _ = process.extractOne(name, country_name_map.keys())
            if score > 80:
                country_ids.append(country_name_map[best_match])
        structured["countries"] = country_ids  # Replace names with IDs

        return structured, total_cost

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in OpenAI response: {reply}\nError: {str(e)}")
        return {}


'''
async def fetch_rendered_html(url, page, retries=2):
    for attempt in range(retries + 1):
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            return {"html": await page.evaluate("() => document.documentElement.outerHTML"), "error": None}

        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt == retries:
                return {"html": None, "error": str(e)}
            await asyncio.sleep(2)
'''
async def fetch_rendered_html(url, page, retries=3):
    for attempt in range(retries):
        try:
            # Try to load but don’t wait forever

            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
                #await asyncio.sleep(3)
            except PlaywrightTimeoutError:
                # fallback to domcontentloaded for resiliency
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)

            try:
                await page.locator("button:has-text('Accept'), button:has-text('Acceptera'), button:has-text('Godkänn'), button:has-text('OK')").click(timeout=3000)
            except:
                pass

            # Give the browser a bit of time to render dynamic content
            await asyncio.sleep(random.uniform(1.0, 3.0))

            html = await page.content()
            if html and html.strip():
                return {"html": html, "error": None}
            else:
                raise ValueError("Empty HTML content")


        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt == retries-1:
                return {"html": None, "error": str(e)}
            # Reload a fresh page if stuck
            try:
                await page.goto("about:blank", timeout=5000)
            except:
                pass
            await asyncio.sleep(2)
    return {"html": None, "error": "Unknown error"}


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
]

async def scrape_focus_fields(start_url,country_name_map, max_pages=5):
    visited = set()
    to_visit = [start_url]
    parsed_domain = urlparse(start_url).netloc.replace("www.", "")
    merged_data = {}
    found_on = None
    errors = []
    total_cost=0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
            args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",  # Prevents crashes in low-memory environments
            "--disable-gpu",
            "--single-process",
            "--no-zygote",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            ]
        )
        context = await browser.new_context(
            user_agent=choice(USER_AGENTS),
            viewport={"width": 1366, "height": 768},
            locale="en-US",
        )
        page = await context.new_page()
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            window.chrome = { runtime: {} };
            """
        )
        while to_visit and len(visited) < max_pages and not is_done(merged_data):
            url = to_visit.pop(0)
            if url in visited or parsed_domain not in url:
                continue

            logging.info(f"Scraping {url} (to_visit size {len(to_visit)})")
            result = await fetch_rendered_html(url, page)
            html, error = result["html"], result["error"]
            if error:
                errors.append({"url": url, "error": error})
                logging.warning(f"Error fetching {url}: {error}")
                if any(x in error.lower() for x in ["403", "blocked", "denied", "captcha"]):
                    logging.warning(f"Access blocked for {parsed_domain}; stopping.")
                    break
                continue

            if not html or not isinstance(html, str):
                logging.error(f"Skipping URL {url}: html is None or not a string")
                continue

            text = BeautifulSoup(html, "html.parser").get_text()
            data, cost = extract_structured_fields(text, country_name_map)
            total_cost += cost

            if any(data.values()) and not found_on:
                found_on = url

            merged_data = merge_data(merged_data, data)
            visited.add(url)

            links = extract_links(html, url, mode="contact")
            # If we haven't found data yet, sort links to prioritize keywords
            if not any(merged_data.values()):
                links = sorted(links, key=lambda link: sum(kw in link.lower() for kw in ["contact", "about", "om", "kontakta", "kontakt"]), reverse=True)

            for link in links:
                if parsed_domain in link and link not in visited and link not in to_visit:
                    to_visit.append(link)

            await asyncio.sleep(random.uniform(0.8, 1.6))

        try:
            await browser.close()
        except:
            pass

    return {
        "source_url": found_on or start_url,
        "structured_data": merged_data,
        "errors": errors,
        "total_cost":round(total_cost, 5)
    }


#########scraping 2

TARGET_FIELDS2 = ["description","keywords", "sales_channel", "number_of_stores", "number_of_brands", "adress", "instagram"]

def merge_data2(existing, new):
    for field in TARGET_FIELDS2:
        if field not in existing or not existing[field]:
            existing[field] = new.get(field, [])
        elif isinstance(existing[field], list) and isinstance(new.get(field), list):
            if field == "keywords":
                # Merge and deduplicate (case-insensitive), then limit to 8
                combined = existing[field] + new[field]
                seen = set()
                deduped = []
                for kw in combined:
                    normalized = kw.strip().lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        deduped.append(kw.strip())
                existing[field] = deduped[:8]  # Limit to 8 keywords
            else:
                # Normal deduplication for other list fields
                existing[field] = list(set(existing[field] + new[field]))
    return existing


async def fetch_rendered_html2(url, page):
    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
        return await page.content()
    except Exception as e:
        logging.warning(f"⚠️ Failed to fetch {url}: {e}")
        return None


def extract_brand_names_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    for tag in soup.find_all(["li", "div", "a", "span"]):
        text = tag.get_text(strip=True)
        if 2 < len(text) < 40:
            candidates.append(text)

    seen = set()
    brands = []
    for name in candidates:
        lower = name.lower()
        if lower not in seen:
            seen.add(lower)
            brands.append(name)
    return brands

async def extract_number_of_brands(base_url, original_html, browser):
    brand_keywords = ["brands", "varumärken", "assortment", "sortiment", "our-brands"]
    links = extract_links(original_html, base_url,mode="all")
    brand_links = [link for link in links if any(k in link.lower() for k in brand_keywords)]

    brand_names = set()
    page = await browser.new_page()

    for link in brand_links[:3]:  # Limit to 3 pages
        try:
            await page.goto(link, timeout=10000)
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
            html = await page.content()
            names = extract_brand_names_from_html(html)
            for n in names:
                brand_names.add(n.strip())
        except Exception as e:
            logging.warning(f"Error scraping {link}: {e}")

    await page.close()
    return len(brand_names), list(brand_names)[:10]


def extract_prio2_fields(text, brand_count, sample_brands):
    example_brands_text = f" End with: 'Example brands: {', '.join(sample_brands)}.'" if sample_brands else ""
    prompt = f"""
You are a AI extraction assistant. From the text below, extract the following:
Respond as JSON with:
- description: (A company description in easy-to-read English (6-7 lines), meant for brands considering partnering with them. Focus on values, products, positioning. {example_brands_text}
- keywords: maximum 6 relevant keywords (e.g., premium, sustainable, K-beauty, anti-aging, low-price).
- sales_channel: Sales channel: 'Online', 'In-store', or 'Online and In-store' based on whether they have a functioning webshop with products.
- number_of_stores: Number of stores they have.
- number_of_brands: {brand_count}
- adress: "Address (The address of the company)"
- instagram: "Instagram (if multiple links take the one that best matches the company name), must begin with 'https://www.instagram.com/' ",
No guessing, if no result found keep empty
{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return {}

    usage=response.usage
    prompt_tokens  =usage.prompt_tokens
    completion_tokens =usage.completion_tokens
    total_tokens =usage.total_tokens
    input_cost = (usage.prompt_tokens / 1000) * 0.005
    output_cost = (usage.completion_tokens / 1000) * 0.015
    total_cost = input_cost + output_cost

    logging.info(f"Total tokens prompt1:{total_tokens},total cost:{total_cost},prompt_tokens:{prompt_tokens}, input cost: {input_cost},completion_tokens:{completion_tokens} ,output cost:{output_cost} ")

    reply = clean_openai_json(response.choices[0].message.content.strip())
    try:
        data = json.loads(reply)
        return data, total_cost, total_tokens
    except json.JSONDecodeError:
        logging.error("Invalid JSON in Prio 2 OpenAI response")
        return {}



async def scrape_prio2_fields(start_url,max_pages=7):
    visited = set()
    to_visit = [start_url]
    parsed_domain = urlparse(start_url).netloc.replace("www.", "")
    merged_data = {}
    found_on = None
    errors = []
    total_cost = 0
    total_tokens = 0
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
    ]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(user_agent=choice(USER_AGENTS))
        page = await context.new_page()
        while to_visit and len(visited) < max_pages and not is_done(merged_data):
            url = to_visit.pop(0)
            if url in visited or parsed_domain not in url:
                continue


            result = await fetch_rendered_html(url, page)
            html, error = result["html"], result["error"]

            if error:
                errors.append({"url": url, "error": error})
                if "403" in error or "blocked" in error.lower() or "denied" in error.lower():
                    logging.warning(f"❌ Access denied or blocked on domain: {parsed_domain}. Stopping further scraping.")
                    break  # stop scraping early
                continue

            if not html or not isinstance(html, str):
                logging.error(f"Skipping URL {url}: html is None or not a string")
                continue

            text = BeautifulSoup(html, "html.parser").get_text()

            brand_count, sample_brands = await extract_number_of_brands(start_url, html, browser)

            data, cost, tokens = extract_prio2_fields(text, brand_count, sample_brands)


            total_cost += cost
            total_tokens += tokens

            if any(data.values()) and not found_on:
                found_on = url

            merged_data = merge_data2(merged_data, data)
            visited.add(url)

            links = extract_links(html, url,mode="content")
            for link in links:
                if parsed_domain in link and link not in visited and link not in to_visit:
                    to_visit.append(link)

            await asyncio.sleep(0.8)

        await browser.close()

    return {
        "source_url": found_on or start_url,
        "structured_data": merged_data,
        "total_cost": round(total_cost, 5),
        "total_tokens": total_tokens,
        "errors":errors
    }



def upload_logo(file_path, retailer_id):
    api_url = "https://backend-riiyo-jonathan165.replit.app/uploadretailer"

    if not file_path or not os.path.exists(file_path):
        logging.info("File does not exist.")
        return

    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'retailer_id': retailer_id}

        try:
            response = requests.post(api_url, files=files, data=data)
            logging.info(f"[⬆️] Upload response for {retailer_id}:, {response.json()}")
        except Exception as e:
            logging.info(f"Upload failed:{e}")
        finally:
            os.remove(file_path)

# Create download folder
TEMP_FOLDER = "temp_logos"
os.makedirs(TEMP_FOLDER, exist_ok=True)


async def download_file(file_url, filename):
    save_path = os.path.join(TEMP_FOLDER, filename)
    try:
        async with aiohttp.ClientSession() as dsession:
            async with dsession.get(file_url) as resp:
                if resp.status == 200:
                    with open(save_path, 'wb') as f:
                        f.write(await resp.read())
                    return save_path
    except Exception as e:
        print(f"⚠️ Failed to download {file_url}: {e}")
    return None


async def scrape_logo(playwright, site_url, retailer_id):
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()

    try:

        logging.info(f"\n🔍 Visiting: {site_url}")
        await page.goto(site_url, timeout=20000)

        # --- 1. Try image logos ---
        img_elements = await page.query_selector_all("img")
        for img in img_elements:
            alt = (await img.get_attribute("alt")) or ""
            cls = (await img.get_attribute("class")) or ""
            src = (await img.get_attribute("src")) or ""

            if src and ('logo' in alt.lower() or 'logo' in cls.lower() or 'logo' in src.lower()):
                logo_url = urljoin(site_url, src)
                #ext = os.path.splitext(logo_url)[-1].split("?")[0] or ".png"
                _, ext = os.path.splitext(urlparse(logo_url).path)
                ext = ext or ".png"
                filename = f"{retailer_id}_logo{ext}"
                file_path = await download_file(logo_url, filename)
                if file_path:
                    logging.info(f"[✔] Found logo for {filename}")
                    upload_logo(file_path, retailer_id)
                    return

        # --- 2. Try inline SVG logos ---
        svg_elements = await page.query_selector_all("svg")
        for i, svg in enumerate(svg_elements):
            outer_html = await svg.evaluate('(el) => el.outerHTML')
            if 'logo' in outer_html.lower():
                filename = f"{retailer_id}_logo.svg"
                file_path = os.path.join(TEMP_FOLDER, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(outer_html)

                logging.info(f"[✔] Found SVG for {filename}")

                upload_logo(file_path, retailer_id)
                return

        logging.info(f"[✖] No logo or favicon found for {site_url}")

    except Exception as e:
        logging.info(f"[⚠] Error scraping {site_url}: {e}")
    finally:
        await browser.close()


async def handle_logo_scraping(website, retailer_id):
    """
    Isolated function to handle everything related to logo scraping with Playwright.
    """
    async with async_playwright() as playwright:
        await scrape_logo(playwright, website, retailer_id)


async def update_database(website, retailer_id):

    results = await scrape_prio2_fields(website,max_pages=7)
    try:
        await handle_logo_scraping(website, retailer_id)
    except Exception as e:
        logging.error(f"⚠️ Error scraping logo for {website}: {e}")

    description_data = results["structured_data"]
    logging.info(f"description_data : {description_data}")


    brand_count_get = description_data.get("number_of_brands", "")
    try:
        brand_count_int = int(brand_count_get)
        brand_count = "" if brand_count_int == 0 else str(brand_count_int)
    except (ValueError, TypeError):
        brand_count = ""


    logging.info(f"brand_count : {brand_count}")
    try:
        session.query(Retailers).filter(Retailers.id ==retailer_id
                    ).update(
                        {Retailers.company_information:description_data.get("description", "") ,
                         Retailers.keywords:", ".join(description_data.get("keywords", [])) ,
                         Retailers.sales_channel:description_data.get("sales_channel", "") ,
                         Retailers.nr_of_stores:description_data.get("number_of_stores", "") ,
                         Retailers.nr_of_brands : brand_count,
                         Retailers.instagram:description_data.get("instagram", "") ,
                         Retailers.address:description_data.get("adress", "") ,
                         Retailers.total_scraping_cost: Retailers.total_scraping_cost + literal(Decimal(str(results["total_cost"])))
                        }
                    )
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Commit failed: {e}")
    return results


#        retailer.number_of_brands = 0#description_data.get("number_of_brands", brand_count)

# async def main():
#     website = 'https://www.hudterapeuten.com/'
#     results = await scrape_focus_fields(website, max_pages=1)
#     print("results", results)


# # --- Run this script ---
# if __name__ == "__main__":
#     try:
#         website='https://hellosister.se/'
#         asyncio.run(scrape_focus_fields(website, max_pages=5))
#         #asyncio.run(update_database(website, 306))
#     except Exception as e:
#         print(f"🔥 Critical error in asyncio run: {e}")
