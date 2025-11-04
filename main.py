from flask import Flask, request, jsonify
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from scrapewebsite import scrape_focus_fields,update_database, Countries, session as Riiyosession
from functools import wraps,lru_cache
from threading import Thread

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Flask + Playwright is running!"

@lru_cache(maxsize=1)
def get_country_name_map():
    all_countries = Riiyosession.query(Countries).all()
    return {c.country_name_english: c.id for c in all_countries}

@app.route("/scrapingriiyo")
def scrapingriiyo_main():
    website = request.args.get("website")
    if not website or not website.startswith("http"):
        return jsonify({"error": "Invalid or missing website URL"}), 400
    country_name_map = get_country_name_map()
    results = asyncio.run(scrape_focus_fields(website, country_name_map, max_pages=7))
    if not results["structured_data"]:
        return jsonify({
            "error": "Failed to extract structured data",
            "details": results.get("errors", []),
            "source_url": results.get("source_url")
        }), 500

    return jsonify(results)

@app.route('/scrapingriiyo2')
def scrapingriiyo2():
    website = request.args.get("website")
    retailer_id = request.args.get("retailer_id")

    def run_scrape():
        asyncio.run(update_database(website, retailer_id))

    Thread(target=run_scrape).start()
    return jsonify({"status": "started"})  # returns fast



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
