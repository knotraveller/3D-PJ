
import random
import json
from pathlib import Path
from multiprocessing import freeze_support
import objaverse



DOWNLOAD_DIR = Path("./objaverse")

CATEGORY_NUMS = {
    "chair": 20,
    "table": 20,
    "mug": 20,
    "airplane": 20,
    "apple": 20,
    "bottle": 20,
    "banana": 20,
}

SEED = 42
DOWNLOAD_PROCESSES = 2

def main():

    random.seed(SEED)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    lvis = objaverse.load_lvis_annotations()
    categories = sorted(lvis.keys())

    with open(DOWNLOAD_DIR / "all_categories.txt", "w", encoding="utf-8") as f:
        for cat in categories:
            f.write(f"{cat}: {len(lvis[cat])}\n")

    lower_to_real = {cat.lower(): cat for cat in categories}

    selected = {}

    for cat, num in CATEGORY_NUMS.items():
        cat_lower = cat.lower()

        if cat_lower not in lower_to_real:
            print(f"[跳过] 没找到类别: {cat}")
            continue

        real_cat = lower_to_real[cat_lower]
        uids = lvis[real_cat]

        sample_num = min(num, len(uids))
        selected[real_cat] = random.sample(uids, sample_num)

        print(f"[选择] {real_cat}: {sample_num} / {len(uids)}")

    all_uids = []
    for uids in selected.values():
        all_uids.extend(uids)

    all_uids = list(set(all_uids))

    print(f"总共准备下载: {len(all_uids)} 个 objects")

    objects = objaverse.load_objects(
        uids=all_uids,
        download_processes=DOWNLOAD_PROCESSES,
    )

    with open(DOWNLOAD_DIR / "selected_uids.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    with open(DOWNLOAD_DIR / "downloaded_objects.json", "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2, ensure_ascii=False)

    print("下载完成。")
    print(f"下载清单保存到: {DOWNLOAD_DIR / 'downloaded_objects.json'}")

if __name__ == "__main__":
    freeze_support()
    main()