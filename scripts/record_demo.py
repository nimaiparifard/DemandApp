"""Record the platform demo video (one continuous browser session -> docs/media/demo.mp4).

Prerequisites: the compose stack is up and filled (see docs/media/README.md or `make demo-video`),
Google Chrome installed (Playwright drives the installed Chrome, no browser download needed), and
`pip install playwright imageio-ffmpeg pillow`.

    python scripts/record_demo.py --terminal scratch/terminal.txt --out docs/media/demo.mp4
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

W, H = 1600, 900
API, UI, MLFLOW, GRAFANA = "http://127.0.0.1:8001", "http://127.0.0.1:8601", "http://127.0.0.1:5000", "http://127.0.0.1:3001"
PROM, AIRFLOW, S3MASTER, FILER, PGW, MAIL = (
    "http://127.0.0.1:9090",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:9333",
    "http://127.0.0.1:8888",
    "http://127.0.0.1:9091",
    "http://127.0.0.1:8025",
)

CSS = """
body{margin:0;background:#0f172a;color:#e2e8f0;font-family:Segoe UI,Arial,sans-serif;}
.wrap{padding:70px 90px;} h1{font-size:54px;margin:0 0 10px;color:#fff} h2{font-size:30px;color:#38bdf8;margin:0 0 30px;font-weight:500}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:20px}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px 18px;font-size:19px}
.card b{display:block;color:#fff;font-size:21px;margin-bottom:4px}.ok{color:#4ade80}.no{color:#fbbf24}
pre{background:#020617;border:1px solid #334155;border-radius:10px;padding:24px;font-size:20px;line-height:1.35;color:#a7f3d0;white-space:pre-wrap}
table{border-collapse:collapse;font-size:24px;margin-top:14px}td,th{border:1px solid #334155;padding:10px 20px;text-align:left}th{background:#1e293b}
.small{font-size:20px;color:#94a3b8;margin-top:22px}
"""


def card_page(page: Page, body: str) -> None:
    page.set_content(f"<html><head><style>{CSS}</style></head><body><div class='wrap'>{body}</div></body></html>")


def caption(page: Page, text: str) -> None:
    page.evaluate(
        """(t) => {
          let d = document.getElementById('__cap');
          if (!d) { d = document.createElement('div'); d.id='__cap';
            d.style.cssText='position:fixed;left:0;right:0;bottom:0;z-index:2147483647;background:rgba(2,6,23,.92);'+
            'color:#fff;font:600 24px Segoe UI,Arial,sans-serif;padding:14px 28px;border-top:3px solid #38bdf8';
            document.body.appendChild(d); }
          d.textContent = t; }""",
        text,
    )


def scene(page: Page, url: str, text: str, secs: float = 5, wait: float = 2.5) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(int(wait * 1000))
    caption(page, text)
    page.wait_for_timeout(int(secs * 1000))


def safe(label: str, fn, failures: list[str]) -> None:
    try:
        fn()
    except Exception as exc:  # keep recording; report at the end
        failures.append(f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
        print("SCENE FAILED", failures[-1], file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminal", type=Path, default=Path("scratch/terminal.txt"))
    ap.add_argument("--out", type=Path, default=Path("docs/media/demo.mp4"))
    ap.add_argument("--shots", type=Path, default=Path("docs/media/screens"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.shots.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    fails: list[str] = []
    terminal = a.terminal.read_text(encoding="utf-8") if a.terminal.exists() else "(no terminal log)"

    def shot(page: Page, name: str) -> None:
        page.screenshot(path=str(a.shots / f"{name}.png"))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--force-device-scale-factor=1"])
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
        page = ctx.new_page()

        # ---- 1. title
        card_page(page, """
          <h1>DemandOps</h1><h2>Retail demand forecasting platform &middot; MLOps-first &middot; 30,490 Walmart series (M5)</h2>
          <div class='small'>One global LightGBM model. The product is the platform: data simulator &rarr; validation &rarr; feature store &rarr;
          training with promotion gates &rarr; canary-ready serving &rarr; monitoring &rarr; drift-triggered retraining.</div>
          <div class='grid'>
            <div class='card'><b>Data &amp; ML</b>pandas, pandera, LightGBM, statsforecast, DVC</div>
            <div class='card'><b>Lifecycle</b>MLflow tracking + registry (champion / challenger)</div>
            <div class='card'><b>Orchestration</b>Airflow 3: ingest, batch forecast, retrain, monitoring</div>
            <div class='card'><b>Serving</b>FastAPI (KServe V2 route), Streamlit what-if UI</div>
            <div class='card'><b>Storage</b>SeaweedFS (S3 API), Postgres, Redis, Feast</div>
            <div class='card'><b>Observability</b>Prometheus, Pushgateway, Grafana, alert rules</div>
            <div class='card'><b>Delivery</b>Docker, Helm, Argo CD + Rollouts, GitHub Actions</div>
            <div class='card'><b>Infra as code</b>OpenTofu, Ansible, k3s / k3d, Harbor mirrors</div>
          </div>""")
        caption(page, "What you are about to see: every service of the stack, running live in Docker Compose")
        page.wait_for_timeout(7000)

        # ---- 2. terminal
        card_page(page, f"<h2>The pipeline, run from the command line (<code>make demo</code>)</h2><pre>{html.escape(terminal)}</pre>")
        caption(page, "Simulate new days of sales, train + backtest, promotion gates, publish features, batch-forecast 28 days")
        page.wait_for_timeout(9000)

        # ---- 3. API docs
        def api_docs() -> None:
            page.goto(f"{API}/docs", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector(".opblock-summary-post", timeout=30_000)
            caption(page, "Inference API (FastAPI, KServe V2 route) - interactive OpenAPI docs")
            page.wait_for_timeout(2500)
            page.locator(".opblock-summary-post").first.click()
            page.get_by_role("button", name="Try it out").click()
            ta = page.locator(".body-param__text")
            ta.fill('{"item_id": "FOODS_1_005", "store_id": "TX_3", "horizon": 7}')
            page.get_by_role("button", name="Execute").click()
            page.wait_for_selector(".live-responses-table", timeout=20_000)
            page.locator(".live-responses-table").scroll_into_view_if_needed()
            page.evaluate("window.scrollBy(0, 260)")
            caption(page, "POST /v2/models/demand-forecaster/infer -> 7-day forecast with prediction interval, served by the champion model")
            page.wait_for_timeout(6000)
            shot(page, "01-api-docs")

        safe("api", api_docs, fails)

        # ---- 4. Streamlit
        def streamlit() -> None:
            page.goto(UI, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector("text=28-day demand forecast", timeout=30_000)
            page.wait_for_timeout(2500)
            page.get_by_role("combobox").first.click()
            page.get_by_role("option", name="TX_3").click()
            page.wait_for_timeout(4000)
            caption(page, "Streamlit: batch forecasts per store (data from the Postgres forecast table)")
            page.wait_for_timeout(3500)
            shot(page, "02-streamlit-forecast")
            page.get_by_role("tab", name="What-if price").click()
            page.wait_for_timeout(3000)
            box = page.get_by_label("New price")
            box.fill("2.50")
            box.press("Enter")
            page.wait_for_timeout(4000)
            caption(page, "What-if price: the API recomputes price features and returns a new forecast (learned association, not causal)")
            page.wait_for_timeout(5000)
            shot(page, "03-streamlit-whatif")

        safe("streamlit", streamlit, fails)

        # ---- 5. MLflow
        def mlflow() -> None:
            page.goto(f"{MLFLOW}/#/experiments/1", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(4000)
            for label in ("Got it", "Skip", "Close"):
                btn = page.get_by_role("button", name=label)
                if btn.count():
                    btn.first.click()
                    break
            page.get_by_text("Model training", exact=True).first.click()
            page.wait_for_timeout(3500)
            caption(page, "MLflow: every training run logs params, backtest metrics (WRMSSE, MAE, bias, WAPE), git SHA, data fingerprint")
            page.wait_for_timeout(6000)
            shot(page, "04-mlflow-runs")
            scene(page, f"{MLFLOW}/#/models/demand-forecaster", "MLflow Model Registry: versions with champion / challenger aliases drive promotion", 6, 4)
            shot(page, "05-mlflow-registry")

        safe("mlflow", mlflow, fails)

        # ---- 6. Airflow
        def airflow() -> None:
            scene(page, f"{AIRFLOW}/dags", "Airflow: the 4 DAGs - daily_ingest, daily_batch_forecast, retrain, monitoring", 6, 5)
            shot(page, "06-airflow-dags")
            page.goto(f"{AIRFLOW}/dags/retrain/runs", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(4500)
            caption(page, "retrain DAG run: train + backtest, gates ('kept champion' under a demand shock), GitOps PR only when gates pass")
            page.wait_for_timeout(6000)
            shot(page, "07-airflow-retrain")
            page.goto(f"{AIRFLOW}/dags/monitoring/runs", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(4500)
            caption(page, "monitoring DAG: PSI drift + realised forecast error pushed to Prometheus")
            page.wait_for_timeout(5000)
            page.goto(f"{AIRFLOW}/dags/daily_ingest/runs", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(4500)
            caption(page, "daily_ingest: validation + feature publish run; the Feast materialize step needs the cluster's Feast registry (not in Compose)")
            page.wait_for_timeout(6000)
            shot(page, "07a-airflow-ingest")

        safe("airflow", airflow, fails)

        # ---- 6b. e-mail sent by the retrain DAG
        def mail() -> None:
            page.goto(MAIL, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector("text=retrain", timeout=20_000)
            page.get_by_text("[DemandOps] retrain").first.click()
            page.wait_for_timeout(3500)
            caption(page, "Mailpit: the decision e-mail the retrain DAG sends (SMTP, no Slack needed)")
            page.wait_for_timeout(6000)
            shot(page, "07b-mail")

        safe("mail", mail, fails)

        # ---- 7. Grafana
        def grafana() -> None:
            scene(page, f"{GRAFANA}/d/demandops-main/demandops?orgId=1&from=now-30m&to=now&kiosk", "Grafana (provisioned from git): RPS, p50/p95/p99 latency, errors, model version, drift, WAPE", 5, 7)
            page.mouse.wheel(0, 700)
            page.wait_for_timeout(5000)
            shot(page, "08-grafana")

        safe("grafana", grafana, fails)

        # ---- 8. Prometheus
        def prometheus() -> None:
            scene(page, f"{PROM}/alerts", "Prometheus alerts: DemandForecastBias and DemandTargetDrift FIRING (label action=retrain) after a 4x demand shock", 7, 3)
            shot(page, "09-prometheus-alerts")
            scene(page, f"{PROM}/targets", "Scrape targets: inference API and Pushgateway", 4, 2)
            scene(
                page,
                f"{PROM}/graph?g0.expr=demandops_forecast_bias&g0.tab=0&g0.range_input=30m",
                "Realised forecast bias reported by the monitoring DAG (backtest bias is within 5 %)",
                5,
                3,
            )

        safe("prometheus", prometheus, fails)

        # ---- 9. Pushgateway + S3 store
        def storage() -> None:
            scene(page, PGW, "Pushgateway: where the short-lived monitoring DAG leaves its drift gauges", 4, 2)
            scene(page, f"{S3MASTER}/", "SeaweedFS master: the S3-compatible object store (MinIO images are no longer published)", 4, 2)
            scene(page, f"{FILER}/buckets/", "SeaweedFS filer: buckets for landing data, features, MLflow artifacts, backups", 4, 2)
            scene(page, f"{FILER}/buckets/demandops-features/", "Feature store offline table: horizon_features.parquet (28 days x every series)", 4, 2)
            shot(page, "10-s3-features")

        safe("storage", storage, fails)

        # ---- 10. results + honest scope
        card_page(page, """
          <h2>Measured results (real M5 data)</h2>
          <table><tr><th>Model (3,000 series, 3 folds)</th><th>WRMSSE</th><th>MAE</th><th>Bias</th></tr>
          <tr><td>Seasonal naive</td><td>0.909</td><td>1.206</td><td>-4.5 %</td></tr>
          <tr><td>AutoETS (store x dept)</td><td>0.752</td><td>1.014</td><td>-3.7 %</td></tr>
          <tr><td><b>LightGBM (global)</b></td><td><b>0.656</b></td><td>1.038</td><td>-1.6 %</td></tr></table>
          <div class='grid'>
            <div class='card'><b>Serving</b>p95 6-8 ms at 50-120 req/s, 0 errors</div>
            <div class='card'><b>Scale</b>30,490 series: features 49 s, 853,720 forecasts in 6 s</div>
            <div class='card'><b>Quality</b>98 tests, 93 % coverage, ruff + mypy clean</div>
            <div class='card'><b>Drift loop</b>4x shock &rarr; Prometheus alerts fired</div>
          </div>""")
        caption(page, "WRMSSE is computed on a sample: comparable across rows, not with the M5 leaderboard")
        page.wait_for_timeout(8000)

        card_page(page, """
          <h2>Defined, validated - but not deployed in this demo</h2>
          <div class='grid'>
            <div class='card'><b class='ok'>Helm chart</b>Rollout canary 10/50/100 + auto-rollback analysis, KServe mode - lint + schema validated</div>
            <div class='card'><b class='ok'>Argo CD</b>app-of-apps for the whole platform - schema validated</div>
            <div class='card'><b class='ok'>OpenTofu + Ansible</b>k3d / VPS provisioning - validate + ansible-lint pass</div>
            <div class='card'><b class='ok'>GitHub Actions</b>CI, image build/push, tofu plan</div>
            <div class='card'><b class='no'>Needs a cluster</b>Feast materialize, Alertmanager &rarr; Airflow trigger, Rollouts rollback, HPA</div>
            <div class='card'><b class='no'>Not measured</b>pod-kill chaos test, backup restore, VPS cost</div>
          </div>""")
        caption(page, "Honest scope: the ML pipeline and monitoring loop ran for real; cluster-only parts are validated, not run")
        page.wait_for_timeout(8000)

        ctx.close()
        browser.close()

    webm = next(tmp.glob("*.webm"))
    ff = subprocess.check_output([sys.executable, "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"], text=True).strip()
    subprocess.run([ff, "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow", "-r", "25", "-movflags", "+faststart", str(a.out)], check=True, capture_output=True)
    # GIF preview for the README (GitHub does not inline-play repo MP4s)
    gif = a.out.with_suffix(".gif")
    subprocess.run([ff, "-y", "-i", str(a.out), "-vf", "fps=4,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer", "-loop", "0", str(gif)], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out} ({a.out.stat().st_size/1e6:.1f} MB) and {gif.name} ({gif.stat().st_size/1e6:.1f} MB)")
    if fails:
        print("FAILED SCENES:\n  " + "\n  ".join(fails))


if __name__ == "__main__":
    main()
