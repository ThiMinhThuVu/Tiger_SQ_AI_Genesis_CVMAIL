# SAFE-Graph knowledge currently in use

Status captured from the repository on 2026-08-10.

![Current SAFE-Graph knowledge structure](assets/safe_graph_knowledge_current.svg)

## Reading the diagram

- Green is the only knowledge path currently exercised by the Q2 evaluator.
- Amber is candidate knowledge or audit evidence. It is present in the repository,
  but is not runtime-approved.
- Red marks the two explicit release gates: the expert-review package is still
  pending, and the Q2 result is an oracle upper bound rather than a deployable
  verifier.

## Operational path today

For a validation component from outer fold `k`, Q2 loads only
`outer_fold_k_train/station_anatomy_support.csv`. It selects the rows whose
`context_basis` is `visible_station`, looks up the predicted anatomy class under
every ground-truth visible station, and computes:

```text
support(component) = max_station Beta(1,1)-smoothed P(anatomy visible | station)
station_risk       = 1 - support(component)
```

The station risk is evaluated alone and appended to local component features in
a leave-one-case-out logistic regression. This is leakage-safe with respect to
the outer fold, but it depends on ground-truth station labels and is therefore
an oracle upper bound, not a deployable runtime path.

## Knowledge inventory

| Layer | Current content | Current authority |
|---|---|---|
| Protocol station-anatomy seed | 14 stations, 49 `boundary_structure_of` edges | Candidate only; soft, absence is not a violation |
| Clinical anatomy relations | 19 rules across 12 relation types | Pending clinical review |
| Fold-safe empirical atlas | 5 outer-fold train scopes; each has 14 x 30 visible-station support values plus pair geometry/co-visibility tables | Only station-anatomy support is used by Q2 |
| All-data atlas | 140 frames / 10 cases | Descriptive audit only; prohibited for fitting |
| OOF confusion graph | 31 class nodes, 50 edges | Review/comparator evidence, not used by Q2 |
| Expert-review package | 115 station rows, 97 anatomy-relation rows, 50 confusion rows | `PENDING_EXPERT_REVIEW` |

The root KG manifest is `NOT_RUNTIME_APPROVED`. No frozen
`clinical_kg_v1.0.json` exists yet.

## Semantic guardrails

- `boundary_structure_of` does not mean mandatory visibility.
- `co_visible_with` does not mean containment or causality.
- An empirical zero does not mean `forbidden_in`.
- Missing anatomy is not automatically a violation.
- View-dependent directions remain soft and require stability plus clinical
  review.
- The single-frame graph does not infer hidden anatomy under an instrument.

## Source of truth

- Protocol seed: `knowledge_graph/clinical_station_anatomy_seed_v1.json`
- Anatomy relation seed: `knowledge_graph/clinical_anatomy_relations_seed_v1.json`
- Hybrid KG manifest: `artifacts/safe_graph_knowledge/kg_v1_20260808/manifest.json`
- Review status: `artifacts/safe_graph_knowledge/kg_v1_20260808/expert_review_v1/review_manifest.json`
- Active Q2 lookup/evaluation: `scripts/evaluate_safe_graph_station_oracle.py`
- Q2 runtime status: `artifacts/safe_graph_verifier/station_oracle_upper_bound_v1/manifest.json`

