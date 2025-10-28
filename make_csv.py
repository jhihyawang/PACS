#!/usr/bin/env python3
import os, re, argparse, csv, sys
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 6→3 類的映射：0→0；1/2/3→1；4/5→2
LABEL_MAP = {0:0, 1:1, 2:1, 3:1, 4:2, 5:2}

VIEW_PATTERNS = {
    "L-CC":  [r"\bL[-_ ]?CC\b", r"\bLEFT[-_ ]?CC\b", r"\bLCC\b"],
    "R-CC":  [r"\bR[-_ ]?CC\b", r"\bRIGHT[-_ ]?CC\b", r"\bRCC\b"],
    "L-MLO": [r"\bL[-_ ]?MLO\b", r"\bLEFT[-_ ]?MLO\b", r"\bLMLO\b"],
    "R-MLO": [r"\bR[-_ ]?MLO\b", r"\bRIGHT[-_ ]?MLO\b", r"\bRMLO\b"],
}

def match_view(name: str):
    base = name.upper()
    base = base.replace("（","(").replace("）",")").replace("、"," ").replace("　"," ")
    for view, pats in VIEW_PATTERNS.items():
        for p in pats:
            if re.search(p, base):
                return view
    return None

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def scan_dataset(root: Path):
    """掃描 Category 0~5，每個病人資料夾下找四視角影像；回傳 rows 與缺視角清單。"""
    rows, missing = [], []
    cat_dirs = []
    for c in range(0, 6):  # 排除 Category 6
        cand = root / f"Category {c}"
        if cand.is_dir():
            cat_dirs.append((c, cand))

    if not cat_dirs:
        print(f"[Error] 找不到 Category 0~5 於 {root}", file=sys.stderr)
        sys.exit(1)

    for orig_label, cdir in cat_dirs:
        for patient_dir in sorted([d for d in cdir.iterdir() if d.is_dir()]):
            pid = patient_dir.name
            view_map = {"L-CC": None, "R-CC": None, "L-MLO": None, "R-MLO": None}

            # 掃描病人資料夾（含子層）
            for p in patient_dir.rglob("*"):
                if not p.is_file() or not is_image_file(p):
                    continue
                v = match_view(p.name)
                if v and (view_map[v] is None):  # 第一個匹配者優先
                    view_map[v] = str(p.resolve())

            if all(view_map.values()):
                rows.append({
                    "id": pid,
                    "L-CC": view_map["L-CC"],
                    "L-MLO": view_map["L-MLO"],
                    "R-CC": view_map["R-CC"],
                    "R-MLO": view_map["R-MLO"],
                    "orig_label": orig_label,                 # 原始 0~5
                    "label": LABEL_MAP[int(orig_label)],      # 映射後 0/1/2
                })
            else:
                missing.append((pid, orig_label, view_map))

    return rows, missing

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","L-CC","L-MLO","R-CC","R-MLO","label","orig_label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="原始資料根目錄，例如 /home/.../PACS/dataset")
    ap.add_argument("--outdir", default="datasets", help="輸出 CSV 的資料夾（預設：專案下的 datasets）")
    ap.add_argument("--val", type=float, default=0.1, help="validation 比例（預設 0.1）")
    ap.add_argument("--test", type=float, default=0.2, help="test 比例（預設 0.2）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()

    print(f"[Info] 掃描資料：{root}")
    rows, missing = scan_dataset(root)
    print(f"[Info] 完整四視角樣本：{len(rows)} 筆")
    if missing:
        print(f"[Warn] 缺視角樣本：{len(missing)} 筆（將不納入 CSV）")

    # 用「映射後的新 label (0/1/2)」做 stratified split
    labels = [r["label"] for r in rows]

    train_val_rows, test_rows = train_test_split(
        rows, test_size=args.test, random_state=args.seed, stratify=labels
    )

    train_labels = [r["label"] for r in train_val_rows]
    train_rows, val_rows = train_test_split(
        train_val_rows, test_size=args.val/(1.0-args.test),
        random_state=args.seed, stratify=train_labels
    )

    write_csv(outdir / "train_label.csv", train_rows)
    write_csv(outdir / "val_label.csv",   val_rows)
    write_csv(outdir / "test_label.csv",  test_rows)

    def stat(rows):
        return dict(Counter([r["label"] for r in rows]))
    print(f"[OK] 輸出：{outdir}/train_label.csv, val_label.csv, test_label.csv")
    print(f"[Stat] train 分佈（新 0/1/2）：{stat(train_rows)}")
    print(f"[Stat] val   分佈（新 0/1/2）：{stat(val_rows)}")
    print(f"[Stat] test  分佈（新 0/1/2）：{stat(test_rows)}")

    if missing:
        print("[Hint] 以下病人缺視角（只顯示前 10 筆）：")
        for i, (pid, lab, vm) in enumerate(missing[:10]):
            miss = [k for k,v in vm.items() if v is None]
            print(f"  - Category {lab} / {pid} 缺：{','.join(miss)}")

if __name__ == "__main__":
    main()
