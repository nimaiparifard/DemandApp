# Demo media

* `demo.mp4` - 3.5 min screen recording of every UI of the stack (H.264, no audio, captions burned in).
* `demo.gif` - low-fps preview used in the main README (GitHub does not inline-play repository MP4s).
* `screens/` - full-resolution screenshots taken during the recording.

## How it was produced (reproducible)

```bash
export STREAMLIT_PORT=8601
DC="docker compose -f infra/docker-compose.yml --profile airflow --profile jobs"
$DC up -d --build postgres redis s3 s3-init mlflow pushgateway prometheus grafana mailpit
$DC run --rm jobs simulate --days 3
$DC run --rm jobs retrain          # first model becomes champion
$DC run --rm jobs publish-features
$DC run --rm jobs batch-forecast
$DC up -d serving streamlit airflow
# traffic for the Grafana panels (tests/load/infer.js), then the drift scenario:
$DC run --rm jobs simulate --days 19 --shift 4.0
docker exec demandops-airflow-1 airflow dags trigger monitoring
docker exec demandops-airflow-1 airflow dags trigger retrain
# put the terminal transcript in scratch/terminal.txt, then:
python scripts/record_demo.py --terminal scratch/terminal.txt --out docs/media/demo.mp4
```
The script drives the locally installed Google Chrome through Playwright (no browser download).
