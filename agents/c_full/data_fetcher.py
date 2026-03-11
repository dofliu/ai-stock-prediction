from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WATCHLIST = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2382": "廣達",
    "2881": "富邦金",
    "2882": "國泰金",
    "2303": "聯電",
    "2412": "中華電",
    "2002": "中鋼",
    "1301": "台塑",
    "1303": "南亞",
    "2886": "兆豐金",
    "2891": "中信金",
    "3711": "日月光投控",
    "2379": "瑞昱",
    "2395": "研華",
    "4938": "和碩",
    "3034": "聯詠",
    "2357": "華碩",
    "2327": "國巨",
    "6505": "台塑化",
    "5880": "合庫金",
    "2884": "玉山金",
    "2892": "第一金",
}


class DataFetcher:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def fetch(self, code: str, days: int = 320) -> Optional[pd.DataFrame]:
        if self.use_mock:
            return self._mock_data(code, days)

        try:
            import yfinance as yf

            end = datetime.today()
            start = end - timedelta(days=days + 60)
            df = yf.Ticker(f"{code}.TW").history(start=start, end=end, auto_adjust=False)
            if df.empty:
                logger.warning("No data for %s, falling back to mock data.", code)
                return self._mock_data(code, days)

            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.tail(days)
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", code, exc)
            return self._mock_data(code, days)

    def fetch_all(self, codes: list[str], days: int = 320) -> dict[str, Optional[pd.DataFrame]]:
        return {code: self.fetch(code, days=days) for code in codes}

    def _mock_data(self, code: str, days: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed=int(code))
        dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
        close = rng.uniform(40, 180)
        rows = []

        for _ in dates:
            daily_alpha = rng.normal(0.0005, 0.018)
            open_ = close * (1 + rng.normal(0, 0.004))
            close = close * (1 + daily_alpha)
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.006)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.006)))
            volume = int(rng.uniform(3_000_000, 40_000_000))
            rows.append(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
            )

        return pd.DataFrame(rows, index=dates)
