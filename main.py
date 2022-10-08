import asyncio

import aiohttp
import feedparser
from fastapi import FastAPI

import utils

tags_metadata = [
    {
        "name": "news",
        "description": "Get news by theme (role)",
    },
]

app = FastAPI(title="MoreTech4 API", openapi_tags=tags_metadata)


@app.get("/news/{theme}", tags=["news"])
async def theme_news(theme: str):
    async with aiohttp.ClientSession() as session:
        tasks = (
            utils.fetch(session, source) for source in utils.get_sources_by_theme(theme)
        )
        htmls = await asyncio.gather(*tasks)
        return {
            "news": [
                {
                    "title": news.title,
                    "description": news.title
                } for entries in map(lambda h: feedparser.parse(h).entries, htmls) for news in entries
            ]
        }
