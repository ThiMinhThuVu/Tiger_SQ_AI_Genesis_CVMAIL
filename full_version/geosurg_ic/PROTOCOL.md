# GeoSurg-IC single-GPU deadline pilot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, run.
- Date: 2026-09-14 UTC.
- Verification status: COMPLETED. 10 implementation/metric tests, 8-step gradient
  audit, all four training arms and native-resolution evaluation completed.
- Hypothesis source: user-provided GeoSurg-IC description, not a verified published paper.
- Resource limit: exactly one visible GPU, physical GPU 3 on vishc-server-2.
- No submission is performed by this experiment.

## Frozen design

Start from each fold's D517 Task-2 P2 checkpoint, input 640 x 1120, original
31-class fine model and coarse/presence auxiliary heads. Freeze Swin encoder;
fine-tune decoder and existing heads at LR 1e-5. All arms run in eval mode with
autograd enabled so dropout cannot contaminate the mixed response. This is a
short decoder adaptation pilot, not full end-to-end GeoSurg-IC training.

Use the existing 32-train/8-validation case manifest, with no train-validation
case overlap. Validation was used to select the starting checkpoint: pilot
scores are exploratory validation, not independent test evidence. Do not use
historical held-out OOF or hidden challenge scores to tune this pilot.

Depth Anything V2 Small, revision 5426e4f0f36572d16453bbda7a8389317b1bef99,
runs locally and is frozen. Cache image-only pseudo-depth at 160 x 280. All
517 images may be cached, but each training fold accesses only its training
records. No fitting/calibration of depth uses validation labels.
Source: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

Choose a non-background touching semantic class pair with >=8 adjacent
transitions on the 160 x 280 routing grid, independently of depth. Make a
two-pixel boundary band. Within each side, estimate local mean depth in a
9 x 9 neighborhood; the absolute cross-side difference after 10--90% depth
normalization is evidence of separation. Select up to 25% of boundary pixels,
requiring normalized depth jump >=0.03. No geometry-based feature attraction
or repulsion is imposed. Coarse routing can miss very small interfaces.

Apply independent per-channel RGB gain (0.8--1.2) and offset (-0.05--0.05)
to pixels belonging to the two chosen classes. The two supports are disjoint;
the four views are clean, A, B, A+B. Supports include all instances of the
selected class in the image; this pilot does not isolate connected components.
No spatial augmentation is applied, preserving cached-depth alignment. This
choice is identical across all arms.

For implied semantic logit margin m = log(p_a) - log(p_b), compute

    delta = m(A+B) - m(A) - m(B) + m(clean)
    L_IC = sum_x w(x) delta(x)^2, sum_x w(x) = 1

Replay four deterministic forward passes to backpropagate the exact first
derivative while keeping only one view graph in memory. All four terms receive
gradients; the detached delta is the analytic upstream derivative, not a
stop-gradient approximation to selected views. Probes/depth do not run at
inference. Original supervised Mask2Former + coarse/hierarchy/presence loss
is retained; the supervised view cycles through the four views equally.

## Arms and budget

1. `aug`: matched appearance augmentation, no IC gradient.
2. `gt`: same IC, uniformly sampled GT interface pixels.
3. `geo`: depth-routed IC.
4. `shuffle`: same-class-pair shuffled routing, exactly preserving coverage.

GT and shuffle arms match the geometry arm's per-image selected-pixel count
(including skipping zero-evidence samples); thus GT is a count-matched
GT-location control, not a wholly depth-independent exposure schedule.
All arms share checkpoint, sample order, perturbation RNG, optimizer and
supervised point-sampling seeds. IC every second step. Fixed 192 optimizer
steps initially, no best-epoch search. IC coefficient **0.02** was locked after
the 8-step train-only audit, before inspecting pilot validation. At coefficient
1 the median IC/supervised gradient norm ratio was 4.99 (range 0.20--49.92).
This is a scale diagnostic, not an efficacy result; the audit updated a
disposable model and is never used as initialization. Audit runtime 29.6 s,
peak allocated VRAM 4083 MiB. All pilot arms reload the original P2 weights.

Screen fold 0 first. If geometry beats starting P2 and matched augmentation
by >=0.001 in center-macro fine task score, examine GT/shuffle and replicate
on fold 1. A deterioration >0.002 in worst-center fine score is a caution
against deployment. Boundary Dice (GT band) and native-resolution fine
Dice/nHD must be checked before promotion. No claim of improved boundary
generalization from one fold or selected validation scores alone.

Reserve the last two hours for evaluation/packaging. Stop launching training
when remaining time cannot cover completion plus this reserve. Existing P2
submission remains the fallback. Keep all pilot artifacts in this namespace.

## Execution record

- Depth cache completed: 517 images, approximately 2 minutes, 28 MB on disk.
- Audit: `artifacts/audit_v1/geo/fold_0/history.json`, 8 steps, not a candidate.
- Pilot: `sbatch full_version/geosurg_ic/pilot.sbatch`, Slurm job **668**.
- Native validation: `sbatch --dependency=afterok:668 full_version/geosurg_ic/native.sbatch`,
  Slurm job **669**. It cannot start until all four pilot arms finish.
- Both Slurm jobs request one GPU and pin the same physical GPU UUID. No arrays,
  parallel GPU workers, extra depth model at inference, or automatic submission.
- Exact pilot command per arm: `.venv/bin/python -u full_version/geosurg_ic/run_pilot.py
  --arm ARM --fold 0 --steps 192 --ic-every 2 --ic-weight 0.02
  --output full_version/geosurg_ic/artifacts/pilot_v1`.
- Train logs and JSON histories record supervised/IC losses, routed pixels,
  gradient norm, elapsed time and peak allocated VRAM. Controller aborts on
  errors; each arm has a 90-minute hard timeout.

## Final decision (2026-09-15)

**NO-GO for GeoSurg-IC promotion.** Native center-macro fine score is 0.794987
for geometry and 0.795670 for matched augmentation; geometry also has lower
GT-boundary-band Dice. The predeclared improvement-over-augmentation gate
failed. No fold-1 expansion or submission replacement was performed. See
`RESULTS.md` and `artifacts/pilot_v1/decision.json` for the completed comparison.
