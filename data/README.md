# TIGER Challenge — Dataset Part 1 & 2

> **Dataset revision 2026-09-12:** the active training folders contain 517
> paired images/masks. Seven withdrawn frames are kept outside the active
> folders under `retired_20260912/` for recovery, and the corrected fine and
> coarse masks for `center_7_case_5_10L.png` were downloaded from Synapse.
> Exact provenance and entity IDs are recorded in
> `dataset_update_20260912.json`.

This dataset contains annotated thoracoscopic (RAMIE) surgery frames for the **semantic segmentation** task of the TIGER Surgical AI Challenge.

---

## Folder Structure

```
tiger_challenge_part1_and_2_filtered/
├── images/                      # Input images (PNG, RGB)
├── masks_fine/                  # Fine-grained segmentation masks (31 classes)
├── masks_coarse/                # Merged segmentation masks (16 classes)
└── labelmap.csv                 # Label definitions: fine-grained ↔ merged mapping + weights
```

---

## Images

- **Format:** PNG, RGB
- **Resolution:** 1280 × 720 (720p) or 1920 × 1080 (1080p)
- **Naming:** `{case_id}_{lymph_node_station}.png`  
  e.g. `center_1_case_10_6L.png` — case 10 from centre 1, annotated at station 6L
- **Cases:** 15 anonymised cases, all from centre 1 — restricted to the source videos
  - `center_1_case_2`, `center_1_case_4`, `center_1_case_5`
  - `center_1_case_6`, `center_1_case_7`, `center_1_case_8`
  - `center_1_case_9` … `center_1_case_17`
- **Stations per case:** up to 14 lymph node stations (6L, 6R, 7L, 7R, 8, 9, 10L, 10R, 11L, 11R, 12L, 12R, 13L, 13R)
- **Total:** 206 images

Each image is a representative frame selected from a surgical video at the moment when the annotated lymph node station is visible.

---

## Segmentation Masks

Both mask folders contain one mask per image, with identical filenames to `images/`.  
Each pixel is coloured according to the anatomical structure it belongs to.

### `masks_fine/` — Fine-grained (31 classes)

Columns `fine_id`, `fine_name`, `fine_r/g/b`, `weight`, `type` from `labelmap.csv`.

| Class | Structure | RGB | Type |
|------:|-----------|-----|------|
|  0 | Background | 0, 0, 0 | Non-anatomical |
|  1 | Instrument | 184, 61, 245 | Non-anatomical |
|  2 | Other | 134, 0, 251 | Non-anatomical |
|  3 | Trachea | 0, 255, 220 | Anatomical |
|  4 | Right main bronchus | 131, 224, 112 | Anatomical |
|  5 | Left main bronchus | 191, 224, 112 | Anatomical |
|  6 | Esophagus | 253, 189, 0 | Anatomical |
|  7 | Fatty tissue esophagus | 255, 224, 32 | Anatomical |
|  8 | Right inferior pulmonary ligament | 207, 164, 117 | Anatomical |
|  9 | Left inferior pulmonary ligament | 234, 204, 159 | Anatomical |
| 10 | Pleura | 15, 126, 87 | Anatomical |
| 11 | Pericardium | 153, 76, 13 | Anatomical |
| 12 | Inferior pulmonary vein | 51, 221, 255 | Anatomical |
| 13 | Right subclavian artery | 255, 142, 142 | Anatomical |
| 14 | Right vagal nerve | 255, 239, 179 | Anatomical |
| 15 | Aorta | 255, 0, 7 | Anatomical |
| 16 | Azygos vein | 0, 6, 255 | Anatomical |
| 17 | Superior caval vein | 50, 183, 250 | Anatomical |
| 18 | Lung | 17, 139, 57 | Anatomical |
| 19 | Lymph node | 255, 0, 204 | Anatomical |
| 20 | Fatty tissue | 250, 250, 55 | Anatomical |
| 21 | Left subclavian artery | 255, 136, 178 | Anatomical |
| 22 | Right bronchial artery | 255, 73, 162 | Anatomical |
| 23 | Pulmonary artery | 255, 161, 136 | Anatomical |
| 24 | Pool of blood | 173, 0, 0 | Anatomical |
| 25 | Resection area | 128, 128, 128 | Anatomical |
| 26 | Gastric conduit | 245, 147, 49 | Anatomical |
| 27 | Right recurrent laryngeal nerve | 218, 161, 69 | Anatomical |
| 28 | Left recurrent laryngeal nerve | 255, 217, 129 | Anatomical |
| 29 | Omentum | 219, 219, 56 | Anatomical |
| 30 | Thoracic duct | 201, 138, 118 | Anatomical |

### `masks_coarse/` — Merged groups (16 classes)

Columns `merged_id`, `merged_name`, `merged_r/g/b` from `labelmap.csv`. Fine-grained structures are merged into anatomical groups:

| Group | Merged Label | RGB | Fine-grained structures included |
|-------|-------------|-----|----------------------------------|
| 0 | Background | 0, 0, 0 | Background |
|  1 | Respiratory Tract | 0, 255, 220 | Trachea, Right/Left main bronchus |
|  2 | Gastroesophageal | 253, 189, 0 | Esophagus, Fatty tissue esophagus, Gastric conduit, Omentum |
|  3 | Pleura | 15, 126, 87 | Right/Left inferior pulmonary ligament, Pleura |
|  4 | Heart | 153, 76, 13 | Pericardium, Inferior pulmonary vein |
|  5 | Vessels | 255, 142, 142 | Right/Left subclavian artery, Right bronchial artery |
|  6 | Nerves | 255, 239, 179 | Right vagal nerve, Right/Left recurrent laryngeal nerve |
|  7 | Aorta | 255, 0, 7 | Aorta |
|  8 | Azygos Vein | 0, 6, 255 | Azygos vein |
|  9 | Superior Caval Vein | 50, 183, 250 | Superior caval vein |
| 10 | Lung | 17, 139, 57 | Lung |
| 11 | Lymphatic Tissue | 255, 0, 204 | Lymph node |
| 12 | Non-anatomical Other | 184, 61, 245 | Instrument, Other |
| 13 | Fatty Tissue | 250, 250, 55 | Fatty tissue |
| 14 | Pulmonary artery | 255, 161, 136 | Pulmonary artery |
| 15 | Anatomical Other | 173, 0, 0 | Pool of blood, Resection area, Thoracic duct |

---

## Reading Masks

Masks are standard RGB PNGs. To decode a pixel:

```python
from PIL import Image
import numpy as np

mask = np.array(Image.open("masks_fine/center_1_case_10_6L.png").convert("RGB"))
# mask[y, x] → (R, G, B) — look up fine_r/g/b in labelmap.csv to get the class name

mask_merged = np.array(Image.open("masks_coarse/center_1_case_10_6L.png").convert("RGB"))
# mask_merged[y, x] → (R, G, B) — look up merged_r/g/b in labelmap.csv to get the group name
```

---

## Task 3 — Lymph Node Station Visibility

Given a single image (one surgical frame), predict which lymph node stations are **visible** in that frame. This is a **multi-label classification** task: each of the 14 possible stations is an independent binary label.

### Labels

| Station | Station | Station | Station |
|---------|---------|---------|---------|
| 6L | 7L | 8 | 10L |
| 6R | 7R | 9 | 10R |
| 11L | 12L | 13L | |
| 11R | 12R | 13R | |

### Annotation file

`lymph_node_station_visibility.csv` — one row per image:

| Column | Description |
|--------|-------------|
| `case` | Anonymised case ID (e.g. `center_1_case_10`) |
| `annotated frame (e.g. 6R)` | The lymph node station that was surgically annotated in this frame |
| `visible lymph stations` | Comma-separated list of all stations visible in the frame |

Example:

```
case,annotated frame (e.g. 6R),visible lymph stations
center_1_case_7,6R,"6R, 6L, 7R, 10R"
```

The annotated station is always included in the visible list. Additional stations may be visible in the background.

**Coverage: 15 of 15 cases.** `center_1_case_6`–`case_17` have all 14 stations labeled. `center_1_case_2` and `case_4` have 9 of 14, and `case_5` has 8 of 14.

---

## Citation and Usage

**This dataset is provided exclusively for use within the TIGER Surgical AI Challenge. Any use outside the scope of this challenge — including publication, redistribution, or commercial application — is strictly prohibited without prior written consent from the dataset authors.**
