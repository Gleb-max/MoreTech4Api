import aiohttp
import feedparser
from fastapi import FastAPI

import config
import utils

app = FastAPI()


@app.get("/news/{theme}")
async def theme_news(theme: str):
    async with aiohttp.ClientSession() as session:
        html = await utils.fetch(session, f"{config.RSS_URL}{theme}")
        rss = feedparser.parse(html)
        return rss.entries
