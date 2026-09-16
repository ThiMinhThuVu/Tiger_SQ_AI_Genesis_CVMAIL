from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

STATIONS = (
    "6L", "6R", "7L", "7R", "8", "9", "10L", "10R",
    "11L", "11R", "12L", "12R", "13L", "13R",
)

TEST_CASES = (
    ("center_1_case_10", "center_1_case_15"),
    ("center_1_case_11", "center_1_case_6"),
    ("center_1_case_12", "center_1_case_7"),
    ("center_1_case_13", "center_1_case_8"),
    ("center_1_case_14", "center_1_case_9"),
)

# The previous two-case validation fold is now the held-out test fold.  The
# validation case rotates to the first case in the next test pair, so it is
# always drawn from that fold's training pool and no case crosses splits.
VALIDATION_CASES = (
    ("center_1_case_11",),
    ("center_1_case_12",),
    ("center_1_case_13",),
    ("center_1_case_14",),
    ("center_1_case_10",),
)

EXPECTED_LABELMAP_SHA256 = "3cd640283a0e7157f096d85e72961a6deb117380dfe413f906b98d09c2cf4671"
EXPECTED_VISIBILITY_SHA256 = "be72045c2ea4d6d5a6ff44721482b107e5581d77560bba334c046c755eeb8434"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_from_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[0].startswith("center_"):
        raise ValueError(f"Unrecognized TIGER frame name: {name}")
    return parts[0]


@dataclass(frozen=True)
class LabelMap:
    fine_names: tuple[str, ...]
    fine_rgb: np.ndarray
    fine_weights: np.ndarray
    fine_to_coarse: np.ndarray
    coarse_names: tuple[str, ...]
    coarse_rgb: np.ndarray
    coarse_weights: np.ndarray
    _fine_lut: np.ndarray
    _coarse_lut: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        rows = list(csv.DictReader(Path(path).open(newline="")))
        fine_ids = [int(row["fine_id"]) for row in rows]
        if fine_ids != list(range(31)):
            raise ValueError(f"fine_id must be exactly 0..30, got {fine_ids}")
        merged_ids = sorted({int(row["merged_id"]) for row in rows})
        if merged_ids != list(range(16)):
            raise ValueError(f"merged_id must cover exactly 0..15, got {merged_ids}")

        fine_rgb = np.asarray(
            [[int(row[f"fine_{c}"]) for c in "rgb"] for row in rows], dtype=np.uint8
        )
        fine_to_coarse = np.asarray([int(row["merged_id"]) for row in rows], dtype=np.uint8)
        fine_weights = np.asarray([float(row["weight"]) for row in rows], dtype=np.float32)

        coarse_rgb = np.zeros((16, 3), dtype=np.uint8)
        coarse_names: list[str | None] = [None] * 16
        coarse_weights = np.zeros(16, dtype=np.float32)
        for row in rows:
            merged_id = int(row["merged_id"])
            rgb = np.asarray([int(row[f"merged_{c}"]) for c in "rgb"], dtype=np.uint8)
            if coarse_names[merged_id] is not None and not np.array_equal(coarse_rgb[merged_id], rgb):
                raise ValueError(f"Inconsistent RGB for merged_id={merged_id}")
            coarse_rgb[merged_id] = rgb
            coarse_names[merged_id] = row["merged_name"]
            coarse_weights[merged_id] = max(coarse_weights[merged_id], float(row["weight"]))

        fine_lut = np.full(1 << 24, 255, dtype=np.uint8)
        coarse_lut = np.full(1 << 24, 255, dtype=np.uint8)
        for idx, (r, g, b) in enumerate(fine_rgb.astype(int)):
            fine_lut[(r << 16) | (g << 8) | b] = idx
        for idx, (r, g, b) in enumerate(coarse_rgb.astype(int)):
            coarse_lut[(r << 16) | (g << 8) | b] = idx
        return cls(
            tuple(row["fine_name"] for row in rows), fine_rgb, fine_weights,
            fine_to_coarse, tuple(str(x) for x in coarse_names), coarse_rgb,
            coarse_weights, fine_lut, coarse_lut,
        )

    @staticmethod
    def _packed(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB mask, got {rgb.shape}")
        return (
            (rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2].astype(np.uint32)
        )

    def decode_fine(self, rgb: np.ndarray) -> np.ndarray:
        ids = self._fine_lut[self._packed(rgb)]
        if np.any(ids == 255):
            unknown = np.unique(rgb[ids == 255].reshape(-1, 3), axis=0)[:10]
            raise ValueError(f"Unknown fine-mask RGB values: {unknown.tolist()}")
        return ids

    def decode_coarse(self, rgb: np.ndarray) -> np.ndarray:
        ids = self._coarse_lut[self._packed(rgb)]
        if np.any(ids == 255):
            unknown = np.unique(rgb[ids == 255].reshape(-1, 3), axis=0)[:10]
            raise ValueError(f"Unknown coarse-mask RGB values: {unknown.tolist()}")
        return ids

    def encode_fine(self, ids: np.ndarray) -> np.ndarray:
        if ids.min(initial=0) < 0 or ids.max(initial=0) >= 31:
            raise ValueError("Fine prediction contains IDs outside 0..30")
        return self.fine_rgb[ids]

    def encode_coarse(self, ids: np.ndarray) -> np.ndarray:
        if ids.min(initial=0) < 0 or ids.max(initial=0) >= 16:
            raise ValueError("Coarse prediction contains IDs outside 0..15")
        return self.coarse_rgb[ids]


def load_visibility(path: str | Path) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            station = row["annotated frame (e.g. 6R)"].strip()
            visible = {item.strip() for item in row["visible lymph stations"].split(",")}
            if not visible <= set(STATIONS):
                raise ValueError(f"Unknown stations for {row['case']}: {sorted(visible - set(STATIONS))}")
            if station not in visible:
                raise ValueError(f"Annotated station {station} is not visible for {row['case']}")
            name = f"{row['case']}_{station}.png"
            if name in output:
                raise ValueError(f"Duplicate visibility row: {name}")
            output[name] = np.asarray([s in visible for s in STATIONS], dtype=np.float32)
    return output


@dataclass(frozen=True)
class Record:
    name: str
    case_id: str
    image: Path
    fine_mask: Path
    visibility: np.ndarray


def records_for_fold(data_root: str | Path, fold: int, split: str) -> list[Record]:
    if fold not in range(5):
        raise ValueError("fold must be in 0..4")
    root = Path(data_root)
    visibility = load_visibility(root / "lymph_node_station_visibility.csv")
    canonical_split = {"val": "validation"}.get(split, split)
    if canonical_split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    split_cases = {
        "validation": set(VALIDATION_CASES[fold]),
        "test": set(TEST_CASES[fold]),
    }
    all_held_out = split_cases["validation"] | split_cases["test"]
    records: list[Record] = []
    for image in sorted((root / "images").glob("*.png")):
        case_id = case_from_name(image.name)
        belongs = (
            case_id not in all_held_out if canonical_split == "train"
            else case_id in split_cases[canonical_split]
        )
        if not belongs:
            continue
        fine_mask = root / "masks_fine" / image.name
        if not fine_mask.is_file() or image.name not in visibility:
            raise FileNotFoundError(f"Missing target for {image.name}")
        records.append(Record(image.name, case_id, image, fine_mask, visibility[image.name]))
    expected = {"train": 98, "validation": 14, "test": 28}[canonical_split]
    if len(records) != expected:
        raise ValueError(f"Fold {fold} {split} expected {expected} frames, got {len(records)}")
    expected_cases = {"train": 7, "validation": 1, "test": 2}[canonical_split]
    if len({r.case_id for r in records}) != expected_cases:
        raise ValueError(f"Fold {fold} {split} has incorrect case count")
    return records


def visibility_pos_weight(records: Sequence[Record]) -> np.ndarray:
    targets = np.stack([record.visibility for record in records])
    positives = targets.sum(0)
    negatives = len(records) - positives
    return np.clip(negatives / np.maximum(positives, 1), 1, 8).astype(np.float32)


class TigerMultitaskDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Record],
        labelmap: LabelMap,
        height: int = 512,
        width: int = 896,
        training: bool = False,
    ) -> None:
        self.records = list(records)
        self.labelmap = labelmap
        self.height = int(height)
        self.width = int(width)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + (high - low) * torch.rand(()).item()

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if torch.rand(()).item() < 0.75:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import affine

            angle = self._uniform(-10.0, 10.0)
            translation = [
                round(self._uniform(-0.05, 0.05) * self.width),
                round(self._uniform(-0.05, 0.05) * self.height),
            ]
            scale = self._uniform(0.90, 1.10)
            image = affine(image, angle, translation, scale, [0.0, 0.0],
                           interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = affine(mask, angle, translation, scale, [0.0, 0.0],
                          interpolation=InterpolationMode.NEAREST, fill=0)
        if torch.rand(()).item() < 0.75:
            image = ImageEnhance.Brightness(image).enhance(self._uniform(0.82, 1.18))
        if torch.rand(()).item() < 0.75:
            image = ImageEnhance.Contrast(image).enhance(self._uniform(0.85, 1.15))
        if torch.rand(()).item() < 0.50:
            image = ImageEnhance.Color(image).enhance(self._uniform(0.88, 1.12))
        return image, mask

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image) as handle:
            image = handle.convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        with Image.open(record.fine_mask) as handle:
            fine_ids = self.labelmap.decode_fine(np.asarray(handle.convert("RGB")))
        mask = Image.fromarray(fine_ids, mode="L").resize(
            (self.width, self.height), Image.Resampling.NEAREST
        )
        if self.training:
            image, mask = self._augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (
            image_array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        fine = np.asarray(mask, dtype=np.int64)
        coarse = self.labelmap.fine_to_coarse[fine].astype(np.int64)
        return {
            "image": torch.from_numpy(image_array.transpose(2, 0, 1).copy()),
            "fine": torch.from_numpy(fine.copy()),
            "coarse": torch.from_numpy(coarse.copy()),
            "visibility": torch.from_numpy(record.visibility.copy()),
            "name": record.name,
            "case_id": record.case_id,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.int64),
        }


def audit_dataset(data_root: str | Path, full: bool = True) -> dict[str, object]:
    root = Path(data_root)
    folders = ("images", "masks_fine", "masks_coarse")
    names = {folder: {p.name for p in (root / folder).glob("*.png")} for folder in folders}
    if any(len(value) != 140 for value in names.values()):
        raise ValueError({folder: len(value) for folder, value in names.items()})
    if not (names["images"] == names["masks_fine"] == names["masks_coarse"]):
        raise ValueError("Image/fine/coarse filenames do not match")
    labelmap = LabelMap.load(root / "labelmap.csv")
    visibility = load_visibility(root / "lymph_node_station_visibility.csv")
    if set(visibility) != names["images"]:
        raise ValueError("Visibility rows do not match image filenames")

    cases = sorted({case_from_name(name) for name in names["images"]})
    if len(cases) != 10:
        raise ValueError(f"Expected 10 cases, got {len(cases)}")
    frames_per_case = {case: sum(case_from_name(name) == case for name in names["images"]) for case in cases}
    if set(frames_per_case.values()) != {14}:
        raise ValueError(frames_per_case)

    folds = []
    for fold in range(5):
        train = records_for_fold(root, fold, "train")
        val = records_for_fold(root, fold, "validation")
        test = records_for_fold(root, fold, "test")
        name_sets = [{r.name for r in records} for records in (train, val, test)]
        case_sets = [{r.case_id for r in records} for records in (train, val, test)]
        if any(name_sets[i] & name_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Leakage in fold {fold}")
        if any(case_sets[i] & case_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Case leakage in fold {fold}")
        folds.append({
            "fold": fold, "train_frames": len(train), "validation_frames": len(val),
            "test_frames": len(test), "train_cases": sorted(case_sets[0]),
            "validation_cases": sorted(case_sets[1]), "test_cases": sorted(case_sets[2]),
            "frame_overlap": 0, "case_overlap": 0,
        })

    unknown_fine = unknown_coarse = coarse_mismatch = pixels = 0
    checked = sorted(names["images"]) if full else [sorted(names["images"])[0]]
    for name in checked:
        fine_rgb = np.asarray(Image.open(root / "masks_fine" / name).convert("RGB"))
        coarse_rgb = np.asarray(Image.open(root / "masks_coarse" / name).convert("RGB"))
        fine = labelmap._fine_lut[labelmap._packed(fine_rgb)]
        coarse = labelmap._coarse_lut[labelmap._packed(coarse_rgb)]
        unknown_fine += int((fine == 255).sum())
        unknown_coarse += int((coarse == 255).sum())
        valid = fine != 255
        coarse_mismatch += int((labelmap.fine_to_coarse[fine[valid]] != coarse[valid]).sum())
        pixels += int(fine.size)
    if unknown_fine or unknown_coarse or coarse_mismatch:
        raise ValueError({
            "unknown_fine_pixels": unknown_fine,
            "unknown_coarse_pixels": unknown_coarse,
            "fine_to_coarse_mismatched_pixels": coarse_mismatch,
        })

    result: dict[str, object] = {
        "status": "PASS", "full_pixel_audit": full, "frames": 140, "cases": 10,
        "frames_per_case": frames_per_case, "visibility_rows": len(visibility),
        "visibility_dimension": len(STATIONS), "pixels_checked": pixels,
        "unknown_fine_pixels": unknown_fine, "unknown_coarse_pixels": unknown_coarse,
        "fine_to_coarse_mismatched_pixels": coarse_mismatch, "folds": folds,
        "labelmap_sha256": sha256(root / "labelmap.csv"),
        "visibility_sha256": sha256(root / "lymph_node_station_visibility.csv"),
    }
    if result["labelmap_sha256"] != EXPECTED_LABELMAP_SHA256:
        raise ValueError("labelmap.csv hash differs from the protocol")
    if result["visibility_sha256"] != EXPECTED_VISIBILITY_SHA256:
        raise ValueError("visibility CSV hash differs from the protocol")
    return result


def write_visibility_ground_truth(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", *STATIONS])
        for record in sorted(records, key=lambda item: item.name):
            writer.writerow([Path(record.name).stem, *record.visibility.astype(int).tolist()])


def save_audit(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
