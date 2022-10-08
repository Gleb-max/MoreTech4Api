import async_timeout


async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()


def get_sources_by_theme(theme: str):
    return [
        f"https://www.vedomosti.ru/rss/rubric/{theme}",
        f"https://news.rambler.ru/rss/{theme}",
    ]
