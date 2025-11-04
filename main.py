from flask import Flask, request, jsonify
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Flask + Playwright is running!"

@app.route("/scrape", methods=["GET"])
def scrape_route():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Missing ?url parameter"}), 400

    result = asyncio.run(scrape_page(url))
    return jsonify(result)

async def scrape_page(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = await browser.new_page()
        await page.goto(url, timeout=20000)
        html = await page.content()
        await browser.close()

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    return {"url": url, "title": title}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
