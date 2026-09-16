# Does B1's ranking head order prompts on the reward's states?

200 training captions of the 3k pool with negatives (`phaseW/rank/negatives_3k.json`), 419 positive/negative pairs.
Teacher: frozen base SD3.5-medium, K=8 Euler, CFG 7.0, selected candidate (argmax dino_patch_cos), noise seed_base+sel. Supervised score indices [4, 5, 6, 7]; the reward's 2 least-noisy student inputs are z_s with s ~ {6: 0.167425, 5: 0.3319275, 4: 0.3901225, 3: 0.110525}.
Projector: pretrained projector.pt. Head: `checkpoints/phaseW/phaseW_CD_dinop_hard_3k-rewRi-s16-rankB1-hp1-acc4_s0_151302/rank_head_final.pt` (step 3000).
K=8 sigmas: [1.0, 0.9475, 0.8828, 0.8008, 0.6938, 0.548, 0.338, 0.0089, 0.0]

Regimes: `x0_zS` = the student's one-step clean estimate at supervised input z_S (z5,z6 = delta 1 at k=6,7; z4,z5 = delta 2 at k=6,7); `x0_mean_dD` = the mean over the two states the reward averages; `rollout` = P(z_K) of the 4-step guidance-1 rollout, the regime the head was trained in.

## B1_s0 (`checkpoints/phaseW/phaseW_CD_dinop_hard_3k-rewRi-s16-rankB1-hp1-acc4_s0_151302/checkpoint_avg_last5.pt`, step 3000)

| regime | space | acc all | 3d_spatial | color | count | shape | spatial | texture | verb | margin | pos score | acc (shuffled anchor) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x0_z3 | raw | 0.630 | 0.562 | 0.654 | 0.590 | 0.471 | 0.646 | 0.667 | 0.675 | +0.00199 | 0.4163 | 0.473 |
| x0_z3 | shaped | 0.726 | 0.750 | 0.722 | 0.754 | 0.618 | 0.723 | 0.933 | 0.675 | +0.00894 | 0.3433 | 0.442 |
| x0_z4 | raw | 0.532 | 0.562 | 0.534 | 0.541 | 0.441 | 0.538 | 0.500 | 0.562 | +0.00025 | 0.4215 | 0.425 |
| x0_z4 | shaped | 0.749 | 0.750 | 0.752 | 0.705 | 0.794 | 0.662 | 0.967 | 0.750 | +0.00549 | 0.3842 | 0.446 |
| x0_z5 | raw | 0.513 | 0.375 | 0.549 | 0.410 | 0.441 | 0.646 | 0.367 | 0.537 | +0.00010 | 0.4184 | 0.430 |
| x0_z5 | shaped | 0.687 | 0.438 | 0.654 | 0.607 | 0.765 | 0.754 | 0.733 | 0.750 | +0.00310 | 0.3777 | 0.477 |
| x0_z6 | raw | 0.566 | 0.562 | 0.586 | 0.607 | 0.529 | 0.554 | 0.633 | 0.500 | +0.00036 | 0.4141 | 0.456 |
| x0_z6 | shaped | 0.594 | 0.438 | 0.617 | 0.557 | 0.588 | 0.554 | 0.633 | 0.637 | +0.00108 | 0.3463 | 0.480 |
| x0_mean_d1 | raw | 0.556 | 0.500 | 0.564 | 0.557 | 0.471 | 0.631 | 0.600 | 0.512 | +0.00023 | 0.4163 | 0.430 |
| x0_mean_d1 | shaped | 0.680 | 0.438 | 0.692 | 0.639 | 0.706 | 0.692 | 0.800 | 0.675 | +0.00209 | 0.3620 | 0.465 |
| x0_mean_d2 | raw | 0.535 | 0.562 | 0.519 | 0.508 | 0.471 | 0.615 | 0.400 | 0.588 | +0.00017 | 0.4200 | 0.403 |
| x0_mean_d2 | shaped | 0.752 | 0.688 | 0.752 | 0.656 | 0.794 | 0.785 | 0.833 | 0.762 | +0.00429 | 0.3810 | 0.453 |
| x0_mean_d3 | raw | 0.594 | 0.625 | 0.594 | 0.541 | 0.382 | 0.692 | 0.567 | 0.650 | +0.00112 | 0.4189 | 0.422 |
| x0_mean_d3 | shaped | 0.783 | 0.875 | 0.752 | 0.836 | 0.765 | 0.738 | 0.900 | 0.775 | +0.00722 | 0.3638 | 0.442 |
| rollout | raw | 0.480 | 0.438 | 0.489 | 0.393 | 0.500 | 0.477 | 0.433 | 0.550 | -0.00131 | 0.3912 | 0.518 |
| rollout | shaped | 0.671 | 0.438 | 0.639 | 0.721 | 0.676 | 0.600 | 0.767 | 0.750 | +0.04591 | 0.2735 | 0.537 |

Cosines (mean over captions): `head_identity_ref` 0.7755, `head_identity_rollout` 0.5492, `head_identity_x0_z3` 0.5885, `head_identity_x0_z4` 0.5772, `head_identity_x0_z5` 0.5758, `head_identity_x0_z6` 0.5773, `x0_z3_vs_rollout_all` 0.8201, `x0_z3_vs_rollout_pos` 0.8316, `x0_z3_vs_rollout_pos_shaped` 0.6772, `x0_z4_vs_rollout_all` 0.8410, `x0_z4_vs_rollout_pos` 0.8531, `x0_z4_vs_rollout_pos_shaped` 0.7170, `x0_z5_vs_rollout_all` 0.8468, `x0_z5_vs_rollout_pos` 0.8596, `x0_z5_vs_rollout_pos_shaped` 0.7305, `x0_z6_vs_rollout_all` 0.8470, `x0_z6_vs_rollout_pos` 0.8594, `x0_z6_vs_rollout_pos_shaped` 0.7351

## ours_s0 (`checkpoints/phaseW/phaseW_CD_dinop_hard_3k-rewRi-s16-hp1-acc4_s0_145176/checkpoint_avg_last5.pt`, step 3000)

| regime | space | acc all | 3d_spatial | color | count | shape | spatial | texture | verb | margin | pos score | acc (shuffled anchor) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x0_z3 | raw | 0.697 | 0.375 | 0.759 | 0.754 | 0.588 | 0.708 | 0.633 | 0.675 | +0.00316 | 0.4278 | 0.482 |
| x0_z3 | shaped | 0.566 | 0.562 | 0.519 | 0.721 | 0.559 | 0.554 | 0.533 | 0.550 | +0.00307 | 0.2515 | 0.470 |
| x0_z4 | raw | 0.797 | 0.750 | 0.850 | 0.803 | 0.765 | 0.754 | 0.767 | 0.775 | +0.00213 | 0.4427 | 0.527 |
| x0_z4 | shaped | 0.558 | 0.500 | 0.511 | 0.672 | 0.588 | 0.631 | 0.467 | 0.525 | +0.00149 | 0.2623 | 0.539 |
| x0_z5 | raw | 0.697 | 0.625 | 0.714 | 0.623 | 0.765 | 0.677 | 0.733 | 0.713 | +0.00100 | 0.4372 | 0.496 |
| x0_z5 | shaped | 0.516 | 0.438 | 0.534 | 0.492 | 0.559 | 0.569 | 0.433 | 0.487 | +0.00058 | 0.2583 | 0.523 |
| x0_z6 | raw | 0.625 | 0.625 | 0.662 | 0.557 | 0.706 | 0.492 | 0.667 | 0.675 | +0.00039 | 0.4213 | 0.527 |
| x0_z6 | shaped | 0.551 | 0.438 | 0.534 | 0.492 | 0.618 | 0.508 | 0.700 | 0.600 | +0.00028 | 0.2477 | 0.520 |
| x0_mean_d1 | raw | 0.697 | 0.625 | 0.714 | 0.656 | 0.706 | 0.631 | 0.767 | 0.738 | +0.00069 | 0.4292 | 0.468 |
| x0_mean_d1 | shaped | 0.537 | 0.438 | 0.541 | 0.459 | 0.676 | 0.538 | 0.567 | 0.537 | +0.00043 | 0.2530 | 0.516 |
| x0_mean_d2 | raw | 0.788 | 0.750 | 0.872 | 0.705 | 0.794 | 0.754 | 0.733 | 0.762 | +0.00156 | 0.4399 | 0.516 |
| x0_mean_d2 | shaped | 0.558 | 0.625 | 0.526 | 0.639 | 0.588 | 0.600 | 0.500 | 0.512 | +0.00103 | 0.2603 | 0.523 |
| x0_mean_d3 | raw | 0.757 | 0.625 | 0.820 | 0.803 | 0.618 | 0.723 | 0.733 | 0.738 | +0.00264 | 0.4353 | 0.487 |
| x0_mean_d3 | shaped | 0.582 | 0.562 | 0.511 | 0.738 | 0.618 | 0.615 | 0.500 | 0.575 | +0.00228 | 0.2569 | 0.480 |
| rollout | raw | 0.525 | 0.500 | 0.451 | 0.508 | 0.559 | 0.508 | 0.700 | 0.600 | +0.00257 | 0.4085 | 0.530 |
| rollout | shaped | 0.699 | 0.438 | 0.632 | 0.803 | 0.735 | 0.662 | 0.800 | 0.762 | +0.04424 | 0.2150 | 0.566 |

Cosines (mean over captions): `head_identity_ref` 0.7755, `head_identity_rollout` 0.5514, `head_identity_x0_z3` 0.5871, `head_identity_x0_z4` 0.5755, `head_identity_x0_z5` 0.5746, `head_identity_x0_z6` 0.5802, `x0_z3_vs_rollout_all` 0.8351, `x0_z3_vs_rollout_pos` 0.8464, `x0_z3_vs_rollout_pos_shaped` 0.6975, `x0_z4_vs_rollout_all` 0.8577, `x0_z4_vs_rollout_pos` 0.8671, `x0_z4_vs_rollout_pos_shaped` 0.7336, `x0_z5_vs_rollout_all` 0.8596, `x0_z5_vs_rollout_pos` 0.8694, `x0_z5_vs_rollout_pos_shaped` 0.7387, `x0_z6_vs_rollout_all` 0.8465, `x0_z6_vs_rollout_pos` 0.8572, `x0_z6_vs_rollout_pos_shaped` 0.7173


## Caption-level cluster bootstrap (20,000 resamples of the 200 captions)

`shaped` = <g(P(x)), g(u_ref)>, `raw` = <P(x), u_ref>, `shuf` = the shaped score against ANOTHER caption's photo.

| student | regime | shaped acc [95% CI] | raw acc [95% CI] | shaped - raw (p) | shaped - shuffled anchor (p) |
|---|---|---|---|---|---|
| B1_s0 | x0_z3 | 0.726 [0.674, 0.776] | 0.630 [0.575, 0.683] | +0.095 [+0.025, +0.164] p=0.0071 | +0.284 [+0.214, +0.354] p=0.0001 |
| B1_s0 | x0_z4 | 0.749 [0.700, 0.797] | 0.532 [0.477, 0.589] | +0.217 [+0.145, +0.288] p=0.0001 | +0.303 [+0.225, +0.378] p=0.0001 |
| B1_s0 | x0_z5 | 0.687 [0.636, 0.736] | 0.513 [0.456, 0.571] | +0.174 [+0.100, +0.246] p=0.0001 | +0.210 [+0.142, +0.277] p=0.0001 |
| B1_s0 | x0_z6 | 0.594 [0.538, 0.650] | 0.566 [0.507, 0.623] | +0.029 [-0.042, +0.100] p=0.4457 | +0.115 [+0.034, +0.195] p=0.0063 |
| B1_s0 | x0_mean_d1 | 0.680 [0.626, 0.733] | 0.556 [0.501, 0.609] | +0.124 [+0.057, +0.191] p=0.0003 | +0.215 [+0.137, +0.292] p=0.0001 |
| B1_s0 | x0_mean_d2 | 0.752 [0.699, 0.801] | 0.535 [0.478, 0.592] | +0.217 [+0.138, +0.292] p=0.0001 | +0.298 [+0.225, +0.371] p=0.0001 |
| B1_s0 | x0_mean_d3 | 0.783 [0.738, 0.825] | 0.594 [0.545, 0.644] | +0.189 [+0.128, +0.247] p=0.0001 | +0.341 [+0.275, +0.408] p=0.0001 |
| B1_s0 | rollout | 0.671 [0.617, 0.723] | 0.480 [0.421, 0.539] | +0.191 [+0.125, +0.257] p=0.0001 | +0.134 [+0.057, +0.212] p=0.0008 |
| ours_s0 | x0_z3 | 0.566 [0.510, 0.621] | 0.697 [0.644, 0.747] | -0.131 [-0.202, -0.062] p=0.0002 | +0.095 [+0.024, +0.165] p=0.0075 |
| ours_s0 | x0_z4 | 0.558 [0.501, 0.616] | 0.797 [0.749, 0.842] | -0.239 [-0.312, -0.165] p=0.0001 | +0.019 [-0.061, +0.099] p=0.6631 |
| ours_s0 | x0_z5 | 0.516 [0.454, 0.576] | 0.697 [0.643, 0.751] | -0.181 [-0.258, -0.106] p=0.0001 | -0.007 [-0.087, +0.074] p=0.8947 |
| ours_s0 | x0_z6 | 0.551 [0.495, 0.608] | 0.625 [0.568, 0.682] | -0.074 [-0.146, -0.002] p=0.0437 | +0.031 [-0.043, +0.104] p=0.4266 |
| ours_s0 | x0_mean_d1 | 0.537 [0.478, 0.595] | 0.697 [0.642, 0.751] | -0.160 [-0.234, -0.087] p=0.0001 | +0.021 [-0.056, +0.099] p=0.6154 |
| ours_s0 | x0_mean_d2 | 0.558 [0.498, 0.619] | 0.788 [0.737, 0.835] | -0.229 [-0.304, -0.154] p=0.0001 | +0.036 [-0.044, +0.117] p=0.3968 |
| ours_s0 | x0_mean_d3 | 0.582 [0.527, 0.638] | 0.757 [0.708, 0.803] | -0.174 [-0.242, -0.106] p=0.0001 | +0.103 [+0.030, +0.176] p=0.0064 |
| ours_s0 | rollout | 0.699 [0.648, 0.748] | 0.525 [0.470, 0.581] | +0.174 [+0.111, +0.237] p=0.0001 | +0.134 [+0.061, +0.206] p=0.0001 |

Reading. The head's ordering is NOT at chance on the reward's x0_hat inputs for the student that was
trained with the shaped reward (B1: 0.68-0.78, every x0 regime above its shuffled-anchor control), so B1's
transfer path is not inert by construction. But the same head applied to the head-naive `ours` student
*removes* ordering on those inputs (shaped 0.52-0.58 vs raw 0.70-0.80, every difference negative and
significant) while still adding ordering on 4-step ROLLOUTS (+0.174, p=1e-4), the regime it was trained in.
So the head's prompt-discrimination is tied to the P(z_K) rollout distribution; on x0_hat at sigma 0.34-0.80
it carries no prompt information of its own, and B1's high shaped accuracy there is the student having been
driven into the head's geometry by 3,000 updates of the shaped reward on exactly these captions and photos
(its raw-projector ordering collapsed from ours' 0.80 to 0.53 at z4, and its raw positive score fell from
0.4427 to 0.4215 while its shaped positive score rose from 0.2623 to 0.3842).

Caveats. (1) These are TRAINING captions and the head's own training negatives, so both students' numbers
are in-sample; the held-out probe (phaseW/rank/probe) is the out-of-sample statement. (2) The head was
trained against the REFRESHED projector, which run 151302 did not save (review finding 3), so the shaped
columns use the pretrained projector.pt and inherit that caveat. (3) The x0_hat states also differ from the
rollout endpoints in content, not only in conditioning: cos(P(x0_hat), P(z_K)) is only 0.82-0.86 raw and
0.68-0.74 shaped for the same caption and prompt.

