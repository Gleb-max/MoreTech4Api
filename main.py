import aiohttp
import async_timeout
import feedparser
from fastapi import FastAPI

app = FastAPI()
RSS_URL = "https://www.vedomosti.ru/rss/rubric/"


async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()


@app.get("/news/{theme}")
async def theme_news(theme: str):
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, f"{RSS_URL}{theme}")
        rss = feedparser.parse(html)
        print(rss)
        return rss.entries
