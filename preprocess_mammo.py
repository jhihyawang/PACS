"""
Batch preprocess & autocrop for mammography dataset.

- 只處理 category 0~5（支援頂層資料夾名為 "Category 0" ~ "Category 5" 或 "0" ~ "5"）
- 維持檔名與副檔名不變
- 保留原始相對路徑層級到 pre_datasets/
"""

import os
import cv2
import argparse
import numpy as np
from typing import Tuple

# -----------------------------
# 影像前處理 + 自動裁切
# -----------------------------
def preprocess_and_autocrop(
    image_path: str,
    threshold_offset: int = -50,
    padding_percent: float = 0.08,
    min_area_ratio: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    預處理並自動裁切乳房X光影像

    參數:
        image_path: 輸入影像路徑
        threshold_offset: 閾值偏移量（負值=更寬鬆，預設 -50）
        padding_percent: 裁切邊距百分比（預設 0.08 = 8%）
        min_area_ratio: 最小輪廓面積比例（預設 0.01 = 1%）

    返回:
        original: 原始灰階影像
        enhanced: CLAHE 增強後的影像
        mask: 偵測到的乳房遮罩
        cropped: 自動裁切後的影像（基於 enhanced）
    """
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"無法讀取影像: {image_path}")

    # 1) CLAHE 對比度增強
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(original)

    # 2) 平滑
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 3) Otsu + 偏移，放寬暗區
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_thresh = max(5, int(otsu_thresh) + int(threshold_offset))
    _, binary = cv2.threshold(blurred, lower_thresh, 255, cv2.THRESH_BINARY)

    # 4) 形態學
    kernel_large = np.ones((11, 11), np.uint8)
    kernel_small = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 5) 找外部輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6) 過濾輪廓
    min_area = original.shape[0] * original.shape[1] * float(min_area_ratio)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not valid_contours:
        raise ValueError("未找到有效輪廓")

    largest_contour = max(valid_contours, key=cv2.contourArea)

    # 7) 凸包
    hull = cv2.convexHull(largest_contour)

    # 8) 遮罩（以凸包填滿）
    mask = np.zeros_like(original)
    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    # 9) 填孔（若有層級資訊）
    contours_all, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(contours_all)):
            # 有父輪廓 => 內部孔洞
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(mask, contours_all, i, 255, thickness=cv2.FILLED)

    # 10) 輕微擴張 + 邊緣羽化（羽化結果此處未直接用於裁切，僅保留如後續需要）
    kernel_dilate = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=3)
    _ = cv2.GaussianBlur(mask, (21, 21), 0)

    # 11) 以凸包外接矩形裁切 + 邊距
    x, y, w, h = cv2.boundingRect(hull)
    pad_x = int(original.shape[1] * float(padding_percent))
    pad_y = int(original.shape[0] * float(padding_percent))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(original.shape[1] - x, w + 2 * pad_x)
    h = min(original.shape[0] - y, h + 2 * pad_y)

    cropped = enhanced[y:y + h, x:x + w]
    return original, enhanced, mask, cropped


# -----------------------------
# 檔案/資料夾處理工具
# -----------------------------
ALLOWED_TOPS = {f"Category {i}" for i in range(6)} | {str(i) for i in range(6)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def should_process(rel_path: str) -> bool:
    """
    僅處理頂層 category 為 0~5 的資料夾。
    允許 "Category 0" ~ "Category 5" 或 "0" ~ "5"。
    """
    parts = rel_path.split(os.sep)
    if not parts:
        return False
    top = parts[0]
    return top in ALLOWED_TOPS


def save_cropped(cropped: np.ndarray, image_path: str, input_root: str, output_root: str) -> str:
    """
    以 input_root 作為相對根目錄，將輸出寫到 output_root，保留相對路徑與檔名/副檔名。
    """
    rel = os.path.relpath(image_path, start=input_root)
    out_dir = os.path.join(output_root, os.path.dirname(rel))
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(image_path)
    out_path = os.path.join(out_dir, base)

    # 若輸出目標與來源相同副檔名，cv2.imwrite 會依副檔名決定編碼器
    ok = cv2.imwrite(out_path, cropped)
    if not ok:
        raise RuntimeError(f"cv2.imwrite 失敗: {out_path}")
    return out_path


# -----------------------------
# 主流程
# -----------------------------
def process_dataset(
    input_root: str,
    output_root: str,
    threshold_offset: int = -50,
    padding_percent: float = 0.08,
    min_area_ratio: float = 0.01,
) -> None:
    total, done, skipped, errors = 0, 0, 0, 0

    for root, _, files in os.walk(input_root):
        for fname in files:
            if not is_image(fname):
                continue

            image_path = os.path.join(root, fname)
            rel = os.path.relpath(image_path, start=input_root)
            total += 1

            # 僅處理 category 0~5
            if not should_process(rel):
                skipped += 1
                continue

            try:
                _, _, _, cropped = preprocess_and_autocrop(
                    image_path,
                    threshold_offset=threshold_offset,
                    padding_percent=padding_percent,
                    min_area_ratio=min_area_ratio,
                )
                save_cropped(cropped, image_path, input_root, output_root)
                done += 1
            except Exception as e:
                errors += 1
                print(f"[錯誤] {image_path}: {e}")

    print("\n=== 統計 ===")
    print(f"總檔數: {total}")
    print(f"處理完成: {done}")
    print(f"跳過(非 0-5): {skipped}")
    print(f"失敗: {errors}")

def build_argparser():
    p = argparse.ArgumentParser(description="Mammography autocrop for categories 0-5")
    p.add_argument("--input-root", type=str, default="check",
                   help="原始資料集根目錄（含 category 資料夾）")
    p.add_argument("--output-root", type=str, default="pre_datasets",
                   help="輸出根目錄（將建立對應 0-5 結構）")
    p.add_argument("--threshold-offset", type=int, default=-50,
                   help="二值化閾值偏移（負值更寬鬆）")
    p.add_argument("--padding-percent", type=float, default=0.08,
                   help="裁切外擴邊距比例（0~1）")
    p.add_argument("--min-area-ratio", type=float, default=0.01,
                   help="最小輪廓面積比例（0~1）")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    process_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        threshold_offset=args.threshold_offset,
        padding_percent=args.padding_percent,
        min_area_ratio=args.min_area_ratio,
    )
