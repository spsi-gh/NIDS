"""Capture README images from a locally running Streamlit dashboard.

Optional: install playwright, then run the dashboard at http://127.0.0.1:8501.
Uses an existing Chrome installation; no browser download is needed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
CHROME = os.environ.get("NIDS_CHROME") or shutil.which("chrome") or shutil.which("google-chrome") or r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(CHROME), headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_role("heading", name="Network Intrusion Detection", exact=True).wait_for(timeout=60_000)
        page.get_by_text("Analyzed flows", exact=True).wait_for(timeout=60_000)
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "overview.png"), full_page=True)
        page.get_by_text("Batch Detection", exact=True).first.click()
        page.get_by_role("heading", name="Batch Detection", exact=True).wait_for(timeout=60_000)
        page.locator('[data-testid="stFileUploader"]').wait_for(timeout=60_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "batch_detection.png"), full_page=True)
        page.get_by_text("Alerts", exact=True).first.click()
        page.get_by_role("heading", name="Alerts", exact=True).wait_for(timeout=60_000)
        page.get_by_text("Attack type", exact=True).wait_for(timeout=60_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "alerts.png"), full_page=True)
        page.get_by_text("Model Performance", exact=True).first.click()
        page.get_by_role("heading", name="Model Performance", exact=True).wait_for(timeout=60_000)
        page.get_by_text("Validation comparison", exact=True).wait_for(timeout=60_000)
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "performance.png"), full_page=True)
        print("Saved dashboard screenshots")
        if errors:
            print("Browser errors:", errors)
        browser.close()


if __name__ == "__main__":
    main()
