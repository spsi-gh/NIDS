"""Browser smoke check for upload, prediction, and alert navigation."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CHROME = os.environ.get("NIDS_CHROME") or shutil.which("chrome") or shutil.which("google-chrome") or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
TEST_DATA = ROOT / "data" / "processed" / "UNSW_NB15_testing-set.csv"


def main() -> None:
    if not TEST_DATA.exists():
        raise FileNotFoundError("Run preprocessing first to create the test CSV")
    with tempfile.TemporaryDirectory() as directory:
        sample = Path(directory) / "smoke_upload.csv"
        pd.read_csv(TEST_DATA).sample(12, random_state=7).drop(columns=["label", "attack_cat", "id"]).to_csv(sample, index=False)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(CHROME), headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
            page.get_by_text("Batch Detection", exact=True).first.click()
            page.get_by_role("heading", name="Batch Detection", exact=True).wait_for(timeout=60_000)
            page.locator('input[type="file"]').set_input_files(str(sample))
            page.get_by_role("button", name="Analyze flows").wait_for(timeout=60_000)
            page.get_by_role("button", name="Analyze flows").click()
            page.get_by_text("Suspicious flows", exact=True).wait_for(timeout=60_000)
            if page.get_by_text("Could not analyze CSV").count():
                raise AssertionError("Batch prediction displayed an error")
            page.get_by_text("Alerts", exact=True).first.click()
            page.get_by_role("heading", name="Alerts", exact=True).wait_for(timeout=60_000)
            print("Dashboard upload, prediction, and alerts passed")
            browser.close()


if __name__ == "__main__":
    main()
