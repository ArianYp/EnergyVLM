# Blinded counterfactual edit audit

210 prompt pairs, stratified by edit family and by whether the fixed image pair reverses, then shuffled. Presentation order within each pair is randomised, so **A is not always the original**.

Fill `review_sheet.csv` (or the JSON). Do not open `answer_key.json` until every row is answered.

## Questions

- **q1_single_atom: Do the two prompts differ in exactly ONE compositional atom (one colour, shape, texture, count, spatial relation, depth relation, or verb)? [yes/no]**
- **q2_semantic_change: Is that difference a real semantic change rather than a synonym or a no-op? [yes/no]**
- **q3_coherent: Is prompt B a coherent, physically plausible request? [yes/no]**
- **q4_which_atom: Which atom differs? [colour/shape/texture/count/spatial/depth/verb/none/other]**
- **q5_notes: free text**

## Why this matters

The counterfactual training arm orients its preference labels using the edited prompt. If the edits do not change exactly one atom, or are not real semantic changes, then a null correct-versus-counterfactual result cannot distinguish 'orientation does not matter' from 'the intervention was too weak to matter'. Report the pass rate per edit family, and report it separately for reversing and non-reversing records.

## Sampling

- 15 records per (edit family x reversal status) stratum, seed 20260807
- Strata present: 3d_spatial/norev, 3d_spatial/rev, color/norev, color/rev, count/norev, count/rev, shape/norev, shape/rev, spatial/norev, spatial/rev, texture/norev, texture/rev, verb/norev, verb/rev
