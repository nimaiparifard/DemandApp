"""`demandops <command>` - local entry points (the Makefile and the DAGs call the same code)."""

from __future__ import annotations

import json
import logging

import typer

from demandops.config import Settings

app = typer.Typer(no_args_is_help=True, help="DemandOps command line")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _settings(n_series: int | None) -> Settings:
    from demandops.config import get_settings

    s = get_settings().model_copy(deep=True)
    if n_series is not None:
        s.demandops_n_series = n_series or None  # 0 = all series
    return s


@app.command()
def train(
    n_series: int | None = typer.Option(None, help="Sample size; 0 = all 30,490 series"),
    rounds: int | None = typer.Option(None, help="LightGBM boosting rounds"),
    baselines: bool = typer.Option(True, help="Also backtest seasonal-naive and AutoETS"),
) -> None:
    """Backtest all models, log to MLflow, register LightGBM as 'challenger'."""
    from demandops.training.train import numeric_summary, train_all

    s = _settings(n_series)
    if rounds:
        s.train.num_boost_round = rounds
    out = train_all(s, baselines=baselines)
    typer.echo(numeric_summary(out.results).round(4).to_string())
    typer.echo(f"registered {s.model_name} v{out.model_version} as 'challenger' (run {out.run_id})")


@app.command()
def evaluate(
    n_series: int | None = typer.Option(None, help="Sample size; 0 = all series"),
    out: str | None = typer.Option(None, help="Write the markdown results table to this file"),
) -> None:
    """Rolling-origin backtest of every model; prints (and optionally saves) the results table.
    Nothing is logged to MLflow or registered."""
    from demandops.data.loader import load_panel
    from demandops.models.lgbm import LGBMForecaster
    from demandops.training.backtest import backtest
    from demandops.training.train import available_baselines

    s = _settings(n_series)
    panel = load_panel(
        s.demandops_data_dir, s.sales_file, s.demandops_n_series, s.train.seed, s.history_days
    )
    rows = []
    for model in [*available_baselines(), LGBMForecaster(s.features, s.train)]:
        r = backtest(model, panel, s.train.n_folds, s.features.horizon).summary()
        rows.append(
            f"| {model.name} | {r['wrmsse']:.4f} | {r['mae']:.4f} | {r['bias']:+.2%} | {r['train_seconds']:.1f} s |"
        )
    header = (
        f"Backtest: {panel['id'].nunique()} series, {s.train.n_folds} rolling-origin folds x {s.features.horizon} days\n\n"
        "| Model | WRMSSE | MAE | Bias | Fit + predict time |\n|---|---|---|---|---|\n"
    )
    table = header + "\n".join(rows)
    typer.echo(table)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(table + "\n")


@app.command()
def retrain(n_series: int | None = None) -> None:
    """Train -> gates -> promote (what DAG 3 does)."""
    from pipelines.retrain_flow import as_dict, run_retrain

    typer.echo(json.dumps(as_dict(run_retrain(_settings(n_series))), indent=2, default=str))


@app.command()
def simulate(days: int = 1, shift: float = 1.0, n_series: int | None = None) -> None:
    """Release `days` new days of sales into the landing bucket (shift != 1 injects demand drift)."""
    from demandops.data.live import landing_store
    from demandops.data.simulator import Simulator

    s = _settings(n_series)
    sim = Simulator(
        landing_store(s),
        s.demandops_data_dir,
        s.sales_file,
        start_day=s.simulator_start_day,
        n_series=s.demandops_n_series,
        seed=s.train.seed,
    )
    released = sim.catch_up(days, demand_shift=shift)
    typer.echo(
        f"released {len(released)} day(s): {released[-1]}" if released else "no day released"
    )


@app.command("publish-features")
def publish_features(n_series: int | None = None) -> None:
    """Build the 28-day feature table and write it to the feature store location."""
    from demandops.data.live import build_horizon_table
    from demandops.storage import ObjectStore

    s = _settings(n_series)
    table = build_horizon_table(s)
    ObjectStore(s.features_uri, s.s3_endpoint_url).write_parquet(table, "horizon_features.parquet")
    typer.echo(f"published {len(table)} feature rows (d_{table['d'].min()}..d_{table['d'].max()})")


@app.command("batch-forecast")
def batch_forecast() -> None:
    """Champion model -> 28-day forecast for all series -> forecasts table."""
    from demandops.batch import predict_all
    from demandops.config import get_settings
    from demandops.storage import ObjectStore
    from demandops.training.train import load_model

    s = get_settings()
    model, version = load_model(s, alias=s.model_alias)
    table = ObjectStore(s.features_uri, s.s3_endpoint_url).read_parquet("horizon_features.parquet")
    typer.echo(f"wrote {predict_all.run(s, model, version, table)} rows (model v{version})")


@app.command()
def drift(n_series: int | None = None) -> None:
    """Print the monitoring report (and push it if PUSHGATEWAY_URL is set)."""
    import os

    from demandops.data.live import load_live_panel
    from demandops.monitoring import exporters
    from demandops.monitoring.drift import should_retrain
    from demandops.monitoring.run import champion_info, compute_drift

    s = _settings(n_series)
    version, baseline = champion_info(s)
    rep = compute_drift(
        s, load_live_panel(s, history_days=s.ingest_history_days), baseline_wape=baseline
    )
    typer.echo(
        json.dumps(
            {
                "psi": rep.feature_psi,
                "data_drift_score": rep.data_drift_score,
                "wape": rep.forecast_wape,
                "should_retrain": should_retrain(rep),
            },
            indent=2,
        )
    )
    if gw := os.environ.get("PUSHGATEWAY_URL"):
        exporters.push(rep, version, gw)


@app.command("reorder-sim")
def reorder_sim(n_series: int | None = None, lead_time: int = 3) -> None:
    """Business metric: stock-out rate and overstock cost of the forecast-driven reorder policy vs. a
    naive trailing-mean policy, replayed OUT OF SAMPLE on the last 28 observed days."""
    from demandops.batch.reorder_sim import compare_holdout
    from demandops.data.live import load_live_panel

    s = _settings(n_series)
    panel = load_live_panel(s, history_days=s.history_days)
    res, n = compare_holdout(panel, s.features, s.train, lead_time=lead_time)
    typer.echo(f"{n} series, lead time {lead_time} d, model fitted before the replayed 28 days")
    typer.echo("| policy | stock-out rate | avg inventory | overstock cost |")
    typer.echo("|---|---|---|---|")
    for name, r in res.items():
        typer.echo(
            f"| {name} | {r.stockout_rate:.2%} | {r.avg_inventory:.2f} | {r.overstock_cost:,.1f} |"
        )


@app.command()
def serve(port: int = 8001, host: str = "0.0.0.0") -> None:  # noqa: S104
    import uvicorn

    uvicorn.run("demandops.serving.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
