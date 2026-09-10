"""Kaggle 대회 데이터 → data/raw/

실행: uv run --env-file .env python src/download_data.py
"""
from collections import Counter
from pathlib import Path

import kagglehub

COMPETITION = "ai14-level-project"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

RAW.mkdir(parents=True, exist_ok=True)
path = Path(kagglehub.competition_download(COMPETITION, output_dir=str(RAW)))
print("경로:", path)

files = [p for p in path.rglob("*") if p.is_file()]
print(f"총 {len(files)}개 / {Counter(p.suffix for p in files).most_common()}")
for d, n in Counter(p.parent.name for p in files).most_common(5):
    print(f"  {d}  {n}")