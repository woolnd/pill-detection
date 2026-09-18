from pathlib import Path
import json
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/sprint_ai_project1_data"


def main():
    class_combos = defaultdict(set)

    for file in RAW.glob("train_annotations/**/*.json"):
        data = json.loads(
            file.read_text(encoding="utf-8")
        )

        image = data["images"][0]
        category = data["categories"][0]

        combo = image["file_name"].split("_")[0]
        class_combos[category["id"]].add(combo)

    print()
    print("===== 클래스별 Combo 분포 =====")

    for class_id in sorted(class_combos):
        combos = sorted(class_combos[class_id])

        print(
            f"{class_id}: "
            f"{len(combos)}개 Combo"
        )

        if len(combos) <= 3:
            for combo in combos:
                print(f"  - {combo}")


if __name__ == "__main__":
    main()