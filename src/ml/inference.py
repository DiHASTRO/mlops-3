# predict_veins.py
"""
Инференс модели сегментации сосудов глазного дна.

На вход — «сырое» изображение (путь/PIL/np.ndarray).
Внутри: FOV-crop -> квадрат -> resize -> нарезка на патчи -> модель -> сборка.

Как модуль:
    from predict_veins import predict
    img = predict("raw_image.png")                # PIL.Image, цветная маска
    img = predict("raw_image.png", return_type="overlay")
    img = predict("raw_image.png", return_type="mask_idx")

Как CLI:
    python predict_veins.py raw_image.png --out out_dir
"""

import argparse
import logging
from pathlib import Path
from typing import Union, Optional, Literal, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from common import settings

from models.UNETPP_WCD.model import (
    ImprovedUNetPlusPlus, val_extra_transforms
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Параметры нормализации — те же, что в скрипте нормализации датасета
# ---------------------------------------------------------------------------
FOV_THRESHOLD = 10        # яркость выше этого — "внутри диска"
MARGIN = 5                # отступ от bbox в пикселях
PAD_MODE = "constant"     # добивка до квадрата: constant (чёрный) / reflect
TARGET_SIZE = (1280, 1280)

# ---------------------------------------------------------------------------
# Глобальное состояние — модель загружается один раз
# ---------------------------------------------------------------------------
_MODEL = None
_VAL_TRANSFORM = None
_PRIORITY_ORDER = [3, 1, 2, 4, 0]


def _load_model(fold: int = 1):
    global _MODEL, _VAL_TRANSFORM  # pylint: disable=global-statement

    if _MODEL is None:
        model_path = ImprovedUNetPlusPlus.get_model_save_path(fold)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Модель не найдена по пути {model_path}. "
                f"Сначала обучите модель или укажите корректный fold."
            )
        _MODEL = ImprovedUNetPlusPlus()
        _MODEL.load(str(model_path))
        logger.info("Модель загружена из %s", model_path)

        transforms = list(val_extra_transforms) if val_extra_transforms else []
        transforms.append(ToTensorV2())
        _VAL_TRANSFORM = A.Compose(transforms)
        logger.info("val-transform собран")

    return _MODEL, _VAL_TRANSFORM


# ---------------------------------------------------------------------------
# Нормализация «сырого» изображения: FOV crop -> square -> resize
# ---------------------------------------------------------------------------
def _find_fov_bbox(img_rgb: np.ndarray,
                   threshold: int = FOV_THRESHOLD,
                   margin: int = MARGIN) -> Tuple[int, int, int, int]:
    """
    Возвращает (x, y, w, h) — bbox не-чёрной области.
    Работает по яркости (max по каналам), ищет крупнейшую компоненту.
    """
    H, W = img_rgb.shape[:2]
    gray = img_rgb.max(axis=2)
    mask = (gray > threshold).astype(np.uint8) * 255

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        # не нашли FOV — вернём всё изображение
        return 0, 0, W, H

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))

    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    w = stats[best, cv2.CC_STAT_WIDTH]
    h = stats[best, cv2.CC_STAT_HEIGHT]

    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(W - x, w + 2 * margin)
    h = min(H - y, h + 2 * margin)
    return int(x), int(y), int(w), int(h)


def _crop_and_square(img: np.ndarray, bbox, pad_mode: str = PAD_MODE) -> np.ndarray:
    x, y, w, h = bbox
    crop = img[y:y + h, x:x + w]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)

    pad_top = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    pad_left = (side - cw) // 2
    pad_right = side - cw - pad_left

    border = cv2.BORDER_REFLECT_101 if pad_mode == "reflect" else cv2.BORDER_CONSTANT

    if crop.ndim == 2:
        return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  border, value=0)
    return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                              border, value=(0, 0, 0))


def _normalize_raw_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Сырое изображение -> 1280x1280 RGB uint8 (та же логика, что в скрипте
    нормализации датасета).
    """
    H, W = img_rgb.shape[:2]
    if H < 16 or W < 16:
        raise ValueError(f"Изображение слишком маленькое: {H}x{W}")

    bbox = _find_fov_bbox(img_rgb)
    img_crop = _crop_and_square(img_rgb, bbox)

    if img_crop.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        img_out = cv2.resize(img_crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    else:
        img_out = img_crop

    return img_out


# ---------------------------------------------------------------------------
# Нарезка / сборка
# ---------------------------------------------------------------------------
def _iter_patch_coords(H, W, patch, stride):
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    for y in ys:
        for x in xs:
            yield y, x


def _reconstruct(patch_preds, coords, H, W, num_classes, priority_order):
    votes = np.zeros((num_classes, H, W), dtype=np.int32)
    for pred, (y, x) in zip(patch_preds, coords):
        ph, pw = pred.shape
        y_end = min(y + ph, H)
        x_end = min(x + pw, W)
        ph_ = y_end - y
        pw_ = x_end - x
        if ph_ <= 0 or pw_ <= 0:
            continue
        pred_crop = pred[:ph_, :pw_]
        for c in range(num_classes):
            votes[c, y:y_end, x:x_end] += (pred_crop == c).astype(np.int32)

    votes_perm = votes[priority_order, :, :]
    idx = np.argmax(votes_perm, axis=0)
    return np.array(priority_order, dtype=np.uint8)[idx]


def _mask_to_rgb(mask_idx: np.ndarray) -> np.ndarray:
    h, w = mask_idx.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in settings.COLOR_MAP.items():
        rgb[mask_idx == cls] = np.array(color, dtype=np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Вход
# ---------------------------------------------------------------------------
def _read_input_image(image) -> np.ndarray:
    """Принимает str/Path/PIL.Image/np.ndarray -> np.ndarray (H,W,3) RGB uint8."""
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Изображение не найдено: {p}")
        arr = np.array(Image.open(p).convert("RGB"))
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"np.ndarray должен быть (H,W,3), получено {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"np.ndarray должен быть uint8, получено {image.dtype}")
        arr = image
    else:
        raise ValueError(
            "image должен быть str/Path, PIL.Image или np.ndarray (H,W,3) uint8"
        )
    return arr


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------
ReturnType = Literal["mask", "mask_idx", "overlay", "image"]


def predict(
    image: Union[str, Path, Image.Image, np.ndarray],
    fold: int = 1,
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
    batch_size: Optional[int] = None,
    priority_order: Optional[list] = None,
    return_type: ReturnType = "mask",
    alpha: float = 0.5,
) -> Image.Image:
    """
    Принимает СЫРОЕ изображение (любого размера), внутри:
      FOV-crop -> square -> resize 1280 -> нарезка на патчи -> модель ->
      сборка голосованием -> PIL.Image.

    Args:
        image: путь, PIL.Image или np.ndarray (H,W,3) RGB uint8 — СЫРОЕ.
        fold: какой фолд модели использовать.
        patch_size/stride/batch_size: если None — из settings.
        priority_order: приоритет классов при равенстве голосов.
        return_type: "mask" (цветная маска), "mask_idx" (0..4),
                     "overlay", "image".
        alpha: прозрачность для overlay.

    Returns:
        PIL.Image.Image в размере TARGET_SIZE (1280x1280).
    """
    model, val_transform = _load_model(fold)

    ps = patch_size or settings.PATCH_SIZE
    st = stride or settings.STRIDE
    bs = batch_size or settings.BATCH_SIZE
    prio = priority_order or _PRIORITY_ORDER

    assert st < ps, f"stride ({st}) должен быть меньше patch_size ({ps})"
    assert set(prio) == set(range(settings.NUM_CLASSES)), \
        f"priority_order должен содержать все {settings.NUM_CLASSES} классов"

    # 1) читаем сырое
    raw_rgb = _read_input_image(image)
    logger.info("Сырое изображение: %s", raw_rgb.shape)

    # 2) нормализуем: FOV-crop -> square -> resize 1280
    img_rgb = _normalize_raw_image(raw_rgb)
    logger.info("После нормализации: %s", img_rgb.shape)

    H, W = img_rgb.shape[:2]
    if H < ps or W < ps:
        raise ValueError(
            f"Нормализованное изображение {H}x{W} меньше патча {ps}x{ps}."
        )

    # 3) патчи + модель
    coords = list(_iter_patch_coords(H, W, ps, st))
    patch_preds = []

    for i in range(0, len(coords), bs):
        batch_coords = coords[i:i + bs]
        batch_imgs = [img_rgb[y:y + ps, x:x + ps] for (y, x) in batch_coords]
        batch_arr = np.stack(batch_imgs, axis=0)   # (B, ps, ps, 3) uint8

        tensors = [val_transform(image=batch_arr[b])["image"]
                   for b in range(batch_arr.shape[0])]
        tensor = torch.stack(tensors, dim=0)

        preds = model.predict(tensor)
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        preds = preds.detach().cpu().numpy().astype(np.uint8)
        for p in preds:
            patch_preds.append(p)

    # 4) сборка
    mask = _reconstruct(patch_preds, coords, H, W, settings.NUM_CLASSES, prio)
    mask_rgb = _mask_to_rgb(mask)

    # 5) возврат
    if return_type == "image":
        return Image.fromarray(img_rgb)

    if return_type == "mask_idx":
        return Image.fromarray(mask)

    if return_type == "mask":
        return Image.fromarray(mask_rgb)

    if return_type == "overlay":
        img = img_rgb.astype(np.float32)
        mrgb = mask_rgb.astype(np.float32)
        non_bg = (mask != 0)[..., None]
        blended = img * (1 - alpha * non_bg) + mrgb * (alpha * non_bg)
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    raise ValueError(
        f"Неизвестный return_type: {return_type}. "
        f"Ожидалось одно из: mask, mask_idx, overlay, image"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Инференс модели сегментации сосудов (сырое изображение)"
    )
    parser.add_argument("image", type=str, help="Путь к СЫРОМУ изображению глазного дна")
    parser.add_argument("--out", type=str, default="prediction_out",
                        help="Директория для сохранения")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    common_kwargs = {
        "fold": args.fold,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
    }

    predict(img_path, return_type="image", **common_kwargs).save(
        out_dir / f"{stem}_normalized.png")
    predict(img_path, return_type="mask", **common_kwargs).save(
        out_dir / f"{stem}_mask.png")
    predict(img_path, return_type="mask_idx", **common_kwargs).save(
        out_dir / f"{stem}_mask_idx.png")
    predict(img_path, return_type="overlay", **common_kwargs).save(
        out_dir / f"{stem}_overlay.png")

    print("Готово.")
    print(f"  сырое изображение: {img_path}")
    print(f"  результаты:        {out_dir.resolve()}")
    print("  файлы:             *_normalized.png, *_mask.png, "
          "*_mask_idx.png, *_overlay.png")


if __name__ == "__main__":
    main()
