# TIGER SQ-AI 2026 combined submission container

This directory builds one offline `linux/amd64` image covering all three tasks:

- `/output/task1/`: 16-colour merged masks (IDs 0–15)
- `/output/task2/`: 31-colour full masks (IDs 0–30)
- `/output/task3.csv`: continuous probabilities for the 14 stations

The image ensembles the five completed
`mask2former_swin_small_clinical_hier_presence_v2` folds. Each segmentation
forward pass is also reused by its fold-specific visibility head. All weights
are copied into the image at build time and the architecture is constructed
locally; runtime network access is neither needed nor attempted.

## Build

Use the Synapse image naming rule and increment the version on every update:

```bash
./submission/build.sh tigersqai26_<team_name>:v1
```

The build script uses Docker BuildKit named contexts so the 1.4 GB of model
weights do not need to be duplicated into this directory. It also resolves
runtime packages into a CPython 3.11 vendor context before the Docker build;
the Dockerfile itself is COPY-only and performs no network access or install.

## Test with the exact offline interface

`INPUT_DIR` must be a flat directory of RGB PNG frames. `OUTPUT_DIR` must be
empty. A CUDA-capable Docker host is required for the full test.

```bash
./submission/test_container.sh \
  tigersqai26_<team_name>:v1 \
  /path/to/sanity/images \
  /path/to/empty/results
```

The test uses `--network none`, read-only `/input`, read-only container root,
and a writable `/output`, then runs the independent contract validator.

To validate already-generated output without rerunning inference:

```bash
PYTHONPATH=submission .venv/bin/python submission/validate_output.py \
  --input /path/to/images \
  --output /path/to/results
```

For the official metric script, place the method output one level below a
prediction root, for example `predictions/OurMethod/{task1,task2,task3.csv}`.

## Export for Synapse

```bash
./submission/save_image.sh tigersqai26_<team_name>:v1 <team_name>
```

This writes `submission/dist/tigersqai_<team_name>_task123.tar.gz` and its
SHA-256 file. Upload the archive to Synapse, declare that the image covers
Tasks 1, 2 and 3, and submit the required short method description. Declare
every external dataset and pretrained checkpoint used by the method.

## Protocol decisions

The current Synapse Docker Instructions (modified 11 August 2026) and current
official evaluator revision `3c052c6` both define Task 1 as merged segmentation
and Task 2 as full segmentation. Older local reports and evaluator revision
`f0b701c` used the reverse naming; this container intentionally follows the
current submission contract.

The linked CARUS document is a general guide from a previous challenge. Its
video input, task argument, registry push, video, and email workflow does not
replace the TIGER SQ-AI 2026 no-argument PNG interface and tar.gz upload rules.
