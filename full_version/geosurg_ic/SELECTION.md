# Two selected candidates

Selected on 2026-09-15 in response to the request to choose two new versions
for submission:

1. **Matched augmentation (`aug`)** — highest fold-0 native center-macro fine
   score, 0.7956700107840776, and highest boundary-band Dice among the four
   new versions.
2. **GeoSurg-IC (`geo`)** — second-highest corresponding fine score,
   0.7949873070178106; retains the requested geometry-conditioned interaction
   mechanism.

The original NO-GO scientific conclusion is unchanged: the pilot does not
establish an advantage for geometry IC over matched augmentation. Selecting
the top two candidates for further work is not a claim that IC won that test.

Both candidates currently have fold-0 checkpoints at
`artifacts/pilot_v1/{aug,geo}/fold_0/best.pt`. Their inference architecture is
compatible with the existing standalone Task-2 packaging. That packaging
requires five distinct fold checkpoints. `selected_5fold.sbatch` prepares the
remaining eight runs sequentially on one GPU, preserving the pilot recipe.

The user subsequently explicitly requested both complete five-fold training
and challenge submission. Job **670** trains the remaining folds on one GPU;
job **671**, dependent on successful completion of 670, packages and smoke-tests
both complete ensembles. The pilot NO-GO is a scientific result, not a veto
on this explicit subsequent submission request.

Live challenge rules were checked: only the final submission per team/task is
scored. Both candidates will be submitted sequentially, with the higher
five-fold center-macro validation fine score last. This does not obtain two
independent hidden-test scores. Team 3602194 is registered, eligible and has
available submission quota in Task-2 evaluation 9619534 as of preparation.
