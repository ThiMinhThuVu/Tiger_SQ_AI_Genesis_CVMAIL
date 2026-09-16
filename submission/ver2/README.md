# TIGER SQ-AI submission ver2 — separate Task 2 and Task 3 images

This version deliberately packages two independent images:

- `task2`: five-fold P2 (`640x1120`) fine-segmentation ensemble; local score `0.78691`.
- `task3`: five-fold P0 (`512x896`) encoders paired fold-by-fold with the visibility heads trained on the updated CSV; local score `0.83440`.

Build, offline-test, and export each image:

```bash
bash submission/ver2/build.sh task2
bash submission/ver2/test_container.sh task2 tigersqai26_genesis_cvmail:ver2-task2 INPUT_DIR EMPTY_OUTPUT_DIR
bash submission/ver2/save_image.sh task2

bash submission/ver2/build.sh task3
bash submission/ver2/test_container.sh task3 tigersqai26_genesis_cvmail:ver2-task3 INPUT_DIR EMPTY_OUTPUT_DIR
bash submission/ver2/save_image.sh task3
```

Task 2 writes only `/output/task2/`. Task 3 writes only `/output/task3.csv`.
Submit each archive only to its matching task.

On a host without Docker-daemon access, build loadable image archives directly:

```bash
bash submission/ver2/assemble.sh task2
bash submission/ver2/assemble.sh task3
```

## Submitted artifacts (2026-09-08)

| Task | Synapse entity | Submission | Evaluation status |
|---|---|---|---|
| Task 2 P2 | `syn77315428` | `9779893` | `RECEIVED` |
| Task 3 P0, updated CSV | `syn77319129` | `9779894` | `RECEIVED` |
| Task 2 P2 fix1 | `syn77417703` | `9780533` | `RECEIVED` |

Upload archives and SHA-256 digests are in `submission/dist/`:

- `tigersqai_Genesis_CVMAIL_ver2_task2.tar.gz`: `5f20b447d861a65b5af943349fe07cf3ed5ae1cdf6a614434e2bd53f2918c8d8`
- `tigersqai_Genesis_CVMAIL_ver2_task3.tar.gz`: `55c2e5bc45b767f50f1a221feb3363b06fc1dd8e2b56b4aee0111516da216d94`
- `tigersqai_Genesis_CVMAIL_task2_fix1.tar.gz`: `89456f4064914455d1f0bad27d0fa385ba9098b0f47c83ec7e007c2ec7f75a30`

Task 2 `fix1` makes its validator self-contained. It no longer imports the
combined-task `contract.py`, which expected `TASK1_PALETTE` and caused the
original standalone Task 2 image to fail during module import.

Both OCI manifests and final application-layer contents were audited. An
end-to-end `docker run` was not possible on this host because its Docker socket
is not accessible to the current account.
