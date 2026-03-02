from pathlib import Path
import re
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SPACE_URL = "https://huggingface.co/spaces/shinzobolte/ai-resume-screener-job-matcher"
OUT_DIR = Path("assets/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def click_if(locator, timeout=4000):
    try:
        locator.first.click(timeout=timeout)
        return True
    except Exception:
        return False


def wait_text(frame, text, timeout=90000):
    frame.get_by_text(text).first.wait_for(timeout=timeout)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    page.goto(SPACE_URL, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(5000)

    click_if(page.get_by_role("link", name="App"), timeout=3000)
    page.wait_for_timeout(2500)

    app_frame = None
    for _ in range(30):
        for fr in page.frames:
            if "hf.space" in fr.url:
                app_frame = fr
                break
        if app_frame:
            break
        page.wait_for_timeout(1000)

    frame = app_frame or page.main_frame

    # Wait for app to be visible
    try:
        wait_text(frame, "ResumePilot AI", timeout=90000)
    except Exception:
        # fallback to previous app title if needed
        wait_text(frame, "AI Resume Screener", timeout=90000)

    # Home screenshot
    frame.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT_DIR / "01-home-mode-selection.png"), full_page=False)

    # JD Match report screenshot
    click_if(frame.get_by_text("JD Match", exact=True), timeout=3000)
    click_if(frame.get_by_text("Use Built-in Sample", exact=True), timeout=3000)
    page.wait_for_timeout(1000)

    # Analyze button variants
    analyzed = (
        click_if(frame.get_by_role("button", name=re.compile("Analyze Resume", re.I)), timeout=7000)
        or click_if(frame.get_by_role("button", name=re.compile("Run JD Match Analysis", re.I)), timeout=7000)
        or click_if(frame.get_by_role("button", name=re.compile("Analyze", re.I)), timeout=7000)
    )
    if analyzed:
        try:
            wait_text(frame, "Match Overview", timeout=120000)
        except Exception:
            page.wait_for_timeout(8000)

    frame.evaluate("window.scrollTo(0, 450)")
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT_DIR / "02-jd-match-report.png"), full_page=False)

    # Resume Health report screenshot
    click_if(frame.get_by_text("Resume Health", exact=True), timeout=5000)
    page.wait_for_timeout(1000)
    analyzed = (
        click_if(frame.get_by_role("button", name=re.compile("Run Resume Health Analysis", re.I)), timeout=7000)
        or click_if(frame.get_by_role("button", name=re.compile("Analyze Resume", re.I)), timeout=7000)
        or click_if(frame.get_by_role("button", name=re.compile("Analyze", re.I)), timeout=7000)
    )
    if analyzed:
        try:
            wait_text(frame, "Resume Health Report", timeout=120000)
        except Exception:
            page.wait_for_timeout(8000)

    frame.evaluate("window.scrollTo(0, 450)")
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT_DIR / "03-resume-health-report.png"), full_page=False)

    browser.close()

print("Captured screenshots:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(p)
