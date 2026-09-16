# TIGER SQ-AI Task 1/2/3 submission — ver1

`ver1` packages the center-stratified 32/8 five-fold models:

- Task 1: independent coarse-only five-model ensemble.
- Task 2: Improved-v2 fine five-model ensemble.
- Task 3: five visibility heads paired with the Task-2 encoders.

Build and export:

```bash
bash submission/ver1/build.sh tigersqai26_genesis_cvmail:ver1
bash submission/test_container.sh tigersqai26_genesis_cvmail:ver1 INPUT_DIR EMPTY_OUTPUT_DIR
bash submission/ver1/save_image.sh tigersqai26_genesis_cvmail:ver1 Genesis_CVMAIL
```

The upload files are written as
`submission/dist/tigersqai_Genesis_CVMAIL_ver1_task123.tar.gz` and `.sha256`.
The existing ver0 archive and source files are not overwritten.

If Docker daemon access is unavailable, assemble an image archive from the
existing validated ver0 Docker archive (its dependency layer is reused):

```bash
bash submission/ver1/assemble.sh
```
