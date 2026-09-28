"""Capture the Signal Observatory v2 preview images with Playwright (Chromium).

    pip install playwright && playwright install chromium
    python scripts/capture_observatory_v2.py

Outputs (docs/figures/v2/):
    observatory_hero_{light,dark}.png     README hero, 1600 x 1000, no browser chrome or scrollbar
    observatory_compare_{light,dark}.png  the side-by-side comparison section, 1600 x 760
    og_observatory.png                  1200 x 630 social card referenced by the page's og:image

The page is paused at the received stage so the capture is deterministic.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / 'docs' / 'observatory_v2.html').resolve().as_uri()
OUT = ROOT / 'docs' / 'figures' / 'v2'

PREPARE = """() => {
  const b = document.getElementById('play'); if (/Pause/.test(b.textContent)) b.click();
  document.querySelector('[data-stage="3"]').click();
  document.documentElement.style.scrollBehavior = 'auto';
}"""


async def shoot(browser, path, width, height, scheme, scroll_to, scale=1):
    ctx = await browser.new_context(viewport={'width': width, 'height': height}, color_scheme=scheme,
                                    device_scale_factor=scale)
    page = await ctx.new_page()
    await page.goto(PAGE)
    await page.wait_for_timeout(700)
    await page.evaluate(PREPARE)
    await page.evaluate(f"() => window.scrollTo(0, document.querySelector('{scroll_to}').offsetTop - 12)")
    await page.wait_for_timeout(400)
    await page.screenshot(path=str(path))
    await ctx.close()
    print('wrote', path)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for scheme in ('light', 'dark'):
            await shoot(browser, OUT / f'observatory_hero_{scheme}.png', 1600, 1000, scheme, '.picker')
            await shoot(browser, OUT / f'observatory_compare_{scheme}.png', 1600, 760, scheme, '.section')
        await shoot(browser, OUT / 'og_observatory.png', 1200, 630, 'light', '.workspace')
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
