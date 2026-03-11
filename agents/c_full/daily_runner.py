from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from data_fetcher import DataFetcher, WATCHLIST
from ml_scorer import CrossSectionalMLScorer
from report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


def run_daily(use_mock: bool = False) -> dict:
    today = date.today().isoformat()
    logger.info("Running agent C daily prediction for %s", today)

    fetcher = DataFetcher(use_mock=use_mock)
    scorer = CrossSectionalMLScorer()
    reporter = ReportGenerator()

    all_data = fetcher.fetch_all(list(WATCHLIST.keys()), days=320)
    predictions, diagnostics = scorer.rank(all_data, WATCHLIST)
    top5 = predictions[:5]

    record = {
        "date": today,
        "agent": "C",
        "method": "Cross-sectional ML ranking (ridge regression on price/volume features)",
        "model_diagnostics": diagnostics,
        "predictions": [
            {
                "rank": rank,
                "ticker": item.code,
                "name": item.name,
                "price": item.latest_close,
                "predicted_return_pct": item.predicted_return_pct,
                "up_probability": item.up_probability,
                "score": item.score,
                "reason": item.reason,
                "features": item.feature_snapshot,
            }
            for rank, item in enumerate(top5, start=1)
        ],
    }

    json_path, md_path = reporter.write_prediction_files(OUTPUT_DIR, record)
    logger.info("Wrote outputs to %s and %s", json_path, md_path)

    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


if __name__ == "__main__":
    import sys

    run_daily(use_mock="--mock" in sys.argv)
