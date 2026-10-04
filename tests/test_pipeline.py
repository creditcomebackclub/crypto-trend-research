from pathlib import Path

from scripts.run_all import load_fixture
from ctr.config import load_config
from ctr.report import run_research


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_pipeline_runs_end_to_end():
    inputs = load_fixture(ROOT / "tests" / "fixtures" / "synthetic_daily.csv")
    config = load_config()
    config["validation"]["bootstrap_samples"] = 40
    report, runs = run_research(*inputs, source_label="synthetic test fixture", config=config)
    assert "not trading advice" in report.lower()
    assert "BTC buy-and-hold" in runs
    assert "BTC trend" in runs
    assert all(len(run.portfolio.returns) for run in runs.values())
