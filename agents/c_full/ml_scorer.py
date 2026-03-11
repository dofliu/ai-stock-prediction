from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "volatility_5",
    "volatility_20",
    "range_pos_20",
    "ma_gap_5_20",
    "ma_gap_10_30",
    "breakout_20",
    "volume_ratio_5",
    "volume_ratio_20",
]


@dataclass
class StockPrediction:
    code: str
    name: str
    latest_close: float
    predicted_return_pct: float
    up_probability: float
    score: float
    reason: str
    feature_snapshot: dict[str, float] = field(default_factory=dict)


class CrossSectionalMLScorer:
    def __init__(self, min_history: int = 140):
        self.min_history = min_history

    def rank(self, all_data: dict[str, Optional[pd.DataFrame]], names: dict[str, str]) -> tuple[list[StockPrediction], dict]:
        dataset = self._build_dataset(all_data)
        if dataset.empty:
            raise ValueError("No usable training data was built from the available histories.")

        unique_dates = sorted(dataset["date"].unique())
        if len(unique_dates) < 60:
            raise ValueError("Not enough time periods to train the model.")

        prediction_date = unique_dates[-1]
        train_df = dataset[dataset["date"] < prediction_date].copy()
        inference_df = dataset[dataset["date"] == prediction_date].copy()

        if train_df.empty or inference_df.empty:
            raise ValueError("Training or inference set is empty.")

        validation_span = min(30, max(10, len(unique_dates) // 6))
        validation_dates = unique_dates[-(validation_span + 1):-1]
        model, diagnostics = self._fit_model(train_df, validation_dates)

        inference_df["predicted_return"] = model.predict(inference_df[FEATURE_COLUMNS])
        residual_scale = max(diagnostics["rmse"], 0.003)
        inference_df["up_probability"] = self._sigmoid(inference_df["predicted_return"] / residual_scale)

        ranked = inference_df.sort_values(
            by=["predicted_return", "up_probability"], ascending=False
        ).reset_index(drop=True)

        predictions = []
        for _, row in ranked.iterrows():
            feature_snapshot = {
                "ret_20_pct": round(row["ret_20"] * 100, 2),
                "ret_60_pct": round(row["ret_60"] * 100, 2),
                "vol_ratio": round(row["volume_ratio_20"], 2),
                "range_pos_20": round(row["range_pos_20"], 2),
                "ma_gap_10_30_pct": round(row["ma_gap_10_30"] * 100, 2),
            }
            predictions.append(
                StockPrediction(
                    code=row["code"],
                    name=names[row["code"]],
                    latest_close=round(row["close"], 2),
                    predicted_return_pct=round(row["predicted_return"] * 100, 2),
                    up_probability=round(row["up_probability"] * 100, 1),
                    score=round(self._to_score(row["predicted_return"], row["up_probability"]), 1),
                    reason=self._build_reason(row),
                    feature_snapshot=feature_snapshot,
                )
            )

        diagnostics["prediction_date"] = str(pd.Timestamp(prediction_date).date())
        diagnostics["training_rows"] = int(len(train_df))
        diagnostics["inference_rows"] = int(len(inference_df))
        return predictions, diagnostics

    def _build_dataset(self, all_data: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
        frames = []
        for code, df in all_data.items():
            if df is None or len(df) < self.min_history:
                continue

            frame = df.copy().sort_index()
            frame["code"] = code
            frame["date"] = frame.index
            frame["ret_1"] = frame["close"].pct_change(1)
            frame["ret_3"] = frame["close"].pct_change(3)
            frame["ret_5"] = frame["close"].pct_change(5)
            frame["ret_10"] = frame["close"].pct_change(10)
            frame["ret_20"] = frame["close"].pct_change(20)
            frame["ret_60"] = frame["close"].pct_change(60)
            frame["volatility_5"] = frame["ret_1"].rolling(5).std()
            frame["volatility_20"] = frame["ret_1"].rolling(20).std()
            frame["range_low_20"] = frame["low"].rolling(20).min()
            frame["range_high_20"] = frame["high"].rolling(20).max()
            frame["range_pos_20"] = (
                (frame["close"] - frame["range_low_20"])
                / (frame["range_high_20"] - frame["range_low_20"] + 1e-9)
            )
            frame["ma_5"] = frame["close"].rolling(5).mean()
            frame["ma_10"] = frame["close"].rolling(10).mean()
            frame["ma_20"] = frame["close"].rolling(20).mean()
            frame["ma_30"] = frame["close"].rolling(30).mean()
            frame["ma_gap_5_20"] = frame["ma_5"] / frame["ma_20"] - 1
            frame["ma_gap_10_30"] = frame["ma_10"] / frame["ma_30"] - 1
            frame["breakout_20"] = frame["close"] / frame["range_high_20"].shift(1) - 1
            frame["volume_ratio_5"] = frame["volume"] / (frame["volume"].rolling(5).mean() + 1e-9)
            frame["volume_ratio_20"] = frame["volume"] / (frame["volume"].rolling(20).mean() + 1e-9)
            frame["target"] = frame["close"].shift(-1) / frame["close"] - 1
            frame = frame.dropna(subset=FEATURE_COLUMNS + ["target"])
            frames.append(frame)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fit_model(self, train_df: pd.DataFrame, validation_dates: list[pd.Timestamp]) -> tuple[Pipeline, dict]:
        train_mask = ~train_df["date"].isin(validation_dates)
        fit_df = train_df[train_mask]
        valid_df = train_df[~train_mask]

        if fit_df.empty:
            fit_df = train_df
            valid_df = train_df.tail(min(100, len(train_df)))

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=2.0)),
            ]
        )
        model.fit(fit_df[FEATURE_COLUMNS], fit_df["target"])

        valid_pred = model.predict(valid_df[FEATURE_COLUMNS])
        valid_actual = valid_df["target"].to_numpy()
        rmse = float(np.sqrt(np.mean((valid_pred - valid_actual) ** 2)))
        direction_acc = float(np.mean((valid_pred > 0) == (valid_actual > 0)))
        corr = float(np.corrcoef(valid_pred, valid_actual)[0, 1]) if len(valid_df) > 1 else 0.0

        final_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=2.0)),
            ]
        )
        final_model.fit(train_df[FEATURE_COLUMNS], train_df["target"])

        diagnostics = {
            "rmse": rmse,
            "directional_accuracy": round(direction_acc * 100, 1),
            "prediction_correlation": round(corr, 3) if np.isfinite(corr) else 0.0,
        }
        return final_model, diagnostics

    def _build_reason(self, row: pd.Series) -> str:
        reasons = []

        if row["ret_20"] > 0.08 and row["ma_gap_10_30"] > 0:
            reasons.append("中期趨勢偏強")
        elif row["ret_20"] < -0.06 and row["range_pos_20"] < 0.35:
            reasons.append("跌深後有反彈條件")

        if row["volume_ratio_20"] > 1.2:
            reasons.append("量能高於 20 日均量")
        if row["breakout_20"] > 0:
            reasons.append("接近 20 日突破")
        if row["volatility_20"] < 0.025:
            reasons.append("波動相對收斂")

        return "、".join(reasons[:3]) if reasons else "模型偏好其短中期報酬結構"

    def _to_score(self, predicted_return: float, up_probability: float) -> float:
        score = 50 + predicted_return * 4000 + (up_probability - 0.5) * 40
        return float(np.clip(score, 0, 100))

    def _sigmoid(self, values: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-values))
