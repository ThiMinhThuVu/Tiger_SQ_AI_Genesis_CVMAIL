# TIGER SQ-AI Task 1 submission — ver2 D517

Standalone Task 1 image trained on the corrected 517-frame dataset snapshot.
It ensembles five center-stratified coarse-only Mask2Former Swin-Small
checkpoints and writes only `/output/task1/`.

Checkpoint-validation summary: mean Task-1 score `0.756590`, mean Dice
`0.713853`, mean nHD `0.200672`, and mean fold-level center-macro selection
score `0.759395`. The exact all-case OOF center-macro report is scheduled as
Slurm job `606`.

Submitted on 2026-09-13:

- Synapse entity: `syn77425596`
- Task-1 submission: `9780545`
- Archive SHA-256: `723cb6d264ba4727999c380f2cdfe7a9f8e45a0ed11fc1029de85a1e055a8139`

The daemonless build is:

```bash
bash submission/ver2_task1/assemble.sh
gzip -1 submission/dist/tigersqai_Genesis_CVMAIL_ver2_task1_d517.tar
```

The Docker build alternative is:

```bash
bash submission/ver2_task1/build.sh
```
