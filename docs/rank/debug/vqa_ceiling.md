# VQAScore ceiling for the structured negatives (O1 validity)

Judge: **clip-flant5-xxl** (external: not DINO, not the reference photograph).
Student: `checkpoints/phaseW/phaseW_CD_dinop_hard_3k-rewRi-s16-hp1-acc4_s0_145176/checkpoint_avg_last5.pt` (step 3000), 4 Euler steps
at guidance 1, ONE shared noise per caption seeded by the caption idx.
200 held-out captions (556 negatives, m=3); the 3k
training pool, the projector manifest (27000 captions) and
the replay photos (2000) are excluded.

For each caption the full VQAScore matrix `S[image of prompt i][prompt t]` is computed over the
positive and its negatives, and

* **contradiction** = the negative's IMAGE scores lower on the POSITIVE prompt than the positive's
  own image does, `S[neg][pos] < S[pos][pos]`. This is the premise any ranking signal needs.
* **edit realised** = the negative's IMAGE scores higher on ITS OWN prompt than the positive's image
  does, `S[neg][neg] > S[pos][neg]`. This is whether the student actually drew the edit.
* **both** = both hold.
* **DINO cos > 0.98** = the negative's image is essentially the positive's image again.
* **prompt-side** = `S[pos][pos] > S[pos][neg]`, a control on the judge: looking only at the
  positive's image, does VQAScore prefer the true caption to the edited one? A family that fails
  here is one the judge cannot read in the TEXT, not one the student failed to draw.

| family | n | contradiction | 95% CI | gap (pos prompt) | edit realised | 95% CI | gap (own prompt) | both | prompt-side | DINO cos>0.98 | DINO cos (med) | pixel MSE (med) |
|---|---:|---:|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| all | 556 | 0.790 | 0.75-0.83 | +0.2375 | 0.853 | 0.82-0.88 | +0.3214 | 0.665 | 0.939 | 0.014 | 0.886 | 0.0202 |
| color | 184 | 0.957 | 0.91-0.99 | +0.4741 | 0.973 | 0.94-0.99 | +0.5533 | 0.929 | 0.978 | 0.016 | 0.889 | 0.0289 |
| texture | 53 | 0.906 | 0.79-1.00 | +0.3457 | 0.906 | 0.83-0.97 | +0.3103 | 0.811 | 0.887 | 0.000 | 0.845 | 0.0196 |
| shape | 27 | 0.852 | 0.69-0.96 | +0.0891 | 0.667 | 0.46-0.85 | +0.1190 | 0.519 | 0.926 | 0.000 | 0.861 | 0.0153 |
| verb | 81 | 0.778 | 0.68-0.87 | +0.1806 | 0.889 | 0.81-0.95 | +0.2742 | 0.679 | 0.963 | 0.012 | 0.855 | 0.0220 |
| 3d_spatial | 22 | 0.682 | 0.45-0.86 | +0.0409 | 0.500 | 0.27-0.73 | -0.0158 | 0.273 | 0.864 | 0.091 | 0.886 | 0.0122 |
| count | 137 | 0.628 | 0.53-0.72 | +0.0512 | 0.803 | 0.72-0.88 | +0.2335 | 0.474 | 0.912 | 0.007 | 0.911 | 0.0163 |
| spatial | 52 | 0.538 | 0.40-0.67 | +0.0294 | 0.692 | 0.56-0.82 | +0.0650 | 0.308 | 0.923 | 0.019 | 0.904 | 0.0172 |

Chance is 0.500 for the first three rates; CIs are caption-clustered bootstrap (2000 resamples).
The positive's image scores 0.897 on its own caption on
average, and the judge prefers the true caption to the edited one on that image
0.939 of the time overall, so the negatives are genuine
contradictions *as text*.

## Interpretation

Over all 556 negatives the premise holds but not overwhelmingly: the negative's image scores
lower on the positive's caption than the positive's own image does
79.0% of the time (CI 0.75-0.83,
mean gap +0.237), the student actually draws the edit
85.3% of the time, and both hold together for 66.5%. So
79.0% is the ceiling for *any* scorer -- DINO, projector or oracle -- on this
negative set: about one negative in five is not a contradiction at all to an external judge, and a
ranking loss that insists on ordering it is training on noise. The failure is not that the student
redraws the same picture: only 1.4% of negatives are near-identical
to the positive (median DINO cosine 0.886, median pixel MSE 0.0202),
i.e. the shared noise gives paired images that do differ -- they just often do not differ in a way
that makes the caption less true. The split by family is sharp. Real contradictions come from
**color** (0.96, gap +0.474, 33% of negatives), **texture** (0.91, gap +0.346, 10% of negatives), **verb** (0.78, gap +0.181, 15% of negatives) -- these carry large VQAScore gaps and the student both violates the caption and draws
the edited attribute. Marginal: **shape** (0.85, gap +0.089, 5% of negatives), **count** (0.63, gap +0.051, 25% of negatives) -- above chance but with gaps an order of magnitude smaller, so the ordering is real yet easily swamped by scorer noise. No usable contradiction: **3d_spatial** (0.68, gap +0.041, 4% of negatives), **spatial** (0.54, gap +0.029, 9% of negatives) -- the contradiction rate does not clear chance. The
relational and numeric families are exactly the compositional ones the campaign is aimed at, and
they are the ones where the negatives are weakest; the colour-dominated mass of the negative set
(33% colour, 25% count) means the ranking term is
mostly learning colour. Note that these same weak families mostly pass the prompt-side control, so
the bottleneck is the student's generation, not the judge's reading of the edited text.

## Which individual rewrites carry the signal (n >= 5)

| family | edit | n | contradiction | gap (pos prompt) | edit realised |
|---|---|---:|---:|---:|---:|
| 3d_spatial | in front of -> behind | 17 | 0.765 | +0.0536 | 0.412 |
| 3d_spatial | behind -> in front of | 5 | 0.400 | -0.0023 | 0.800 |
| color | white -> brown | 22 | 1.000 | +0.3778 | 0.955 |
| color | brown -> red | 15 | 0.800 | +0.1572 | 0.800 |
| color | green -> yellow | 13 | 0.846 | +0.3901 | 1.000 |
| color | blue -> green | 11 | 1.000 | +0.4947 | 1.000 |
| color | black -> brown | 11 | 1.000 | +0.4061 | 0.909 |
| color | white -> red | 11 | 0.909 | +0.5053 | 1.000 |
| color | red -> blue | 11 | 1.000 | +0.7511 | 1.000 |
| color | black -> white | 11 | 1.000 | +0.5896 | 1.000 |
| color | red -> green | 9 | 1.000 | +0.7130 | 1.000 |
| color | brown -> blue | 8 | 0.875 | +0.5586 | 1.000 |
| color | blue -> yellow | 7 | 1.000 | +0.6898 | 1.000 |
| color | red -> yellow | 7 | 1.000 | +0.6419 | 1.000 |
| color | green -> purple | 6 | 1.000 | +0.3858 | 1.000 |
| color | black -> red | 6 | 1.000 | +0.5406 | 1.000 |
| color | yellow -> purple | 5 | 1.000 | +0.5230 | 1.000 |
| count | two -> three | 45 | 0.489 | -0.0098 | 0.822 |
| count | two -> four | 38 | 0.684 | +0.0330 | 0.895 |
| count | two -> five | 17 | 0.706 | +0.1046 | 1.000 |
| count | three -> two | 9 | 0.889 | +0.2816 | 0.667 |
| count | three -> four | 9 | 0.667 | +0.0417 | 0.667 |
| shape | large -> small | 10 | 0.900 | +0.0821 | 1.000 |
| shape | small -> big | 8 | 0.625 | +0.1055 | 0.625 |
| shape | little -> big | 6 | 1.000 | +0.1007 | 0.000 |
| spatial | next to -> far from | 29 | 0.483 | +0.0182 | 0.759 |
| spatial | on top of -> under | 9 | 0.556 | +0.0570 | 0.778 |
| spatial | near -> far from | 7 | 0.429 | -0.0129 | 0.714 |
| texture | wooden -> metal | 11 | 0.909 | +0.3116 | 1.000 |
| verb | standing -> sitting | 26 | 0.808 | +0.2849 | 0.923 |
| verb | sitting -> standing | 20 | 0.850 | +0.2059 | 0.900 |
| verb | holding -> dropping | 16 | 0.688 | +0.0684 | 0.875 |
| verb | walking -> running | 6 | 0.500 | +0.1109 | 0.667 |
