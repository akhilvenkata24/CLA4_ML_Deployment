"""
CLA 4 - Capture Execution Screenshots
Generates high-resolution screenshots for all terminal executions and FastAPI endpoints.
"""

import os
import subprocess
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

terminal_tasks = [
    ("01_model_training.png", "Model Training Pipeline", ["python", "train_model.py"]),
    ("02_online_inference.png", "Online Single Record Inference", ["python", "online_inference.py"]),
    ("03_batch_inference.png", "Batch Inference (20 Records)", ["python", "batch_inference.py"]),
    ("04_streaming_inference.png", "Streaming Inference (10 Events)", ["python", "streaming_inference.py"]),
    ("08_fastapi_client_test.png", "FastAPI Client Verification Suite", ["python", "client.py"])
]


def render_terminal_screenshots(playwright_instance):
    browser = playwright_instance.chromium.launch()
    page = browser.new_page(viewport={"width": 1100, "height": 900})

    for filename, title, cmd in terminal_tasks:
        cmd_str = " ".join(cmd)
        print(f"Executing and capturing: {cmd_str}")
        res = subprocess.run(cmd, cwd=os.path.dirname(__file__), capture_output=True, text=True)
        raw_output = res.stdout

        escaped_output = (
            raw_output.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            background: #0d1117;
            margin: 20px;
            font-family: 'Consolas', 'Courier New', monospace;
            display: flex;
            justify-content: center;
        }}
        .window {{
            width: 100%;
            max-width: 1000px;
            background: #161b22;
            border-radius: 10px;
            border: 1px solid #30363d;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            overflow: hidden;
        }}
        .header {{
            background: #21262d;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #30363d;
        }}
        .dots {{
            display: flex;
            gap: 8px;
            margin-right: 16px;
        }}
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .red {{ background: #ff5f56; }}
        .yellow {{ background: #ffbd2e; }}
        .green {{ background: #27c93f; }}
        .title {{
            color: #8b949e;
            font-size: 13px;
            font-weight: 600;
            flex-grow: 1;
            text-align: center;
        }}
        .content {{
            padding: 18px 22px;
            color: #e6edf3;
            font-size: 13.5px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .prompt {{
            color: #58a6ff;
            font-weight: bold;
        }}
        .cmd {{
            color: #7ee787;
        }}
    </style>
</head>
<body>
    <div class="window">
        <div class="header">
            <div class="dots">
                <div class="dot red"></div>
                <div class="dot yellow"></div>
                <div class="dot green"></div>
            </div>
            <div class="title">Terminal — {title}</div>
        </div>
        <div class="content"><span class="prompt">PS C:\\Users\\akhil\\CLA4_ML_Deployment&gt; </span><span class="cmd">{cmd_str}</span>\n\n{escaped_output}</div>
    </div>
</body>
</html>"""

        page.set_content(html)
        page.wait_for_timeout(400)
        out_path = os.path.join(SCREENSHOTS_DIR, filename)
        page.locator(".window").screenshot(path=out_path)
        print(f"--> Saved: {out_path}")

    browser.close()


def capture_fastapi_screenshots(playwright_instance):
    browser = playwright_instance.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 800})

    # 1. Swagger UI Overview
    page.goto("http://127.0.0.1:8000/docs")
    page.wait_for_timeout(1500)
    docs_path = os.path.join(SCREENSHOTS_DIR, "05_fastapi_swagger_docs.png")
    page.screenshot(path=docs_path, full_page=True)
    print(f"--> Saved: {docs_path}")

    # 2. Expand all endpoints in Swagger
    blocks = page.locator(".opblock-summary")
    for i in range(blocks.count()):
        blocks.nth(i).click()
        page.wait_for_timeout(400)
    exp_path = os.path.join(SCREENSHOTS_DIR, "06_fastapi_endpoints_expanded.png")
    page.screenshot(path=exp_path, full_page=True)
    print(f"--> Saved: {exp_path}")

    # 3. Health Endpoint
    page.goto("http://127.0.0.1:8000/health")
    page.wait_for_timeout(1000)
    health_path = os.path.join(SCREENSHOTS_DIR, "07_fastapi_health_endpoint.png")
    page.screenshot(path=health_path)
    print(f"--> Saved: {health_path}")

    browser.close()


def main():
    print("Capturing all project screenshots...")
    with sync_playwright() as p:
        render_terminal_screenshots(p)
        capture_fastapi_screenshots(p)
    print("All screenshots generated successfully in:", SCREENSHOTS_DIR)


if __name__ == "__main__":
    main()
