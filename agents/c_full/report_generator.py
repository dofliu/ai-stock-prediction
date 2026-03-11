from __future__ import annotations

import json
from pathlib import Path


class ReportGenerator:
    def write_prediction_files(self, output_dir: Path, record: dict) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "daily_prediction.json"
        md_path = output_dir / "daily_prediction.md"

        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._to_markdown(record), encoding="utf-8")
        return json_path, md_path

    def _to_markdown(self, record: dict) -> str:
        lines = [
            f"# 參賽者 C 每日推薦 ({record['date']})",
            "",
            f"- 方法: {record['method']}",
            f"- 訓練樣本: {record['model_diagnostics']['training_rows']}",
            f"- 驗證方向準確率: {record['model_diagnostics']['directional_accuracy']}%",
            f"- 驗證相關係數: {record['model_diagnostics']['prediction_correlation']}",
            "",
            "| 排名 | 代碼 | 名稱 | 收盤價 | 預測隔日報酬 | 上漲機率 | 分數 | 理由 |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]

        for item in record["predictions"]:
            lines.append(
                "| {rank} | {ticker} | {name} | {price:.2f} | {pred:+.2f}% | {prob:.1f}% | {score:.1f} | {reason} |".format(
                    rank=item["rank"],
                    ticker=item["ticker"],
                    name=item["name"],
                    price=item["price"],
                    pred=item["predicted_return_pct"],
                    prob=item["up_probability"],
                    score=item["score"],
                    reason=item["reason"],
                )
            )

        return "\n".join(lines) + "\n"
