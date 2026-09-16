# TIGER SQ-AI full-version experiments

This namespace contains experiments trained on the 40-case dataset snapshot.
It is intentionally separate from the historical 10-case artifacts under
`artifacts/` and `outputs/`.

## Task namespaces

- `task1_merged_16cls/`: official Task 1, deterministic 16-ID merged output
  exported from the frozen fine-segmentation OOF predictions. The full40
  seed-2026 artifact contains 524 native-resolution RGB masks and five-fold
  metrics.
- `task2_fine_31cls/`: official Task 2, 31 output IDs including background
  (30 foreground anatomical classes).
- `task3_visibility/`: official Task 3, five-fold visibility heads trained on
  the 308 genuinely annotated frames. The 216 segmentation-only frames are
  excluded rather than interpreted as negative visibility labels.

Historical repository reports may call this output "Task 1". New full-version
artifacts must use the official challenge task numbering.
