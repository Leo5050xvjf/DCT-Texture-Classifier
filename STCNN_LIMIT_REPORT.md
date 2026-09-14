# STCNN noise-robustness limit study

Date: 2026-09-15

## Objective

This study tests how far a compact Spatial Texture CNN (STCNN) can preserve the
clean-image texture/non-texture decision under noise. Unless explicitly stated,
STCNN means a spatial-domain patch CNN. The compact model has 39,777 parameters
and receives a 32x32 luminance context to classify its central 8x8 region.

## Fixed protocol

- Natural training set: 6,000 balanced contexts from DIV2K training images.
- Natural validation set: 2,000 balanced contexts from disjoint DIV2K validation images.
- Labels: hard pseudo-labels selected from clean-image high/low Sobel regions.
- Validation: five fixed noise realizations per corruption condition.
- Threshold: 0.5.
- Important limitation: these metrics reproduce the clean Sobel-extreme
  definition. They are not human texture GT and are not full-image accuracy.

## Experiments

1. Standard mixed-AWGN STCNN, trained with sigma uniformly sampled from 0 to 50.
2. Paired clean/noisy BCE plus prediction-consistency loss.
3. Noise-conditioned STCNN, with sigma encoded as an additional input.
4. A 633,729-parameter STCNN teacher with the same 32x32 input.
5. Logit distillation from the large teacher to the 39,777-parameter student.
6. Expansion of the training range from sigma 0-50 to sigma 0-100.
7. Family-shift tests: correlated Gaussian, Poisson, speckle,
   salt-and-pepper, and sinusoidal/banding-like interference.
8. Six-family noise training, with and without exact sigma conditioning.
9. Sigma-input error, jitter, and randomization tests.
10. Synthetic clean-signal anchors to expose the ambiguity between stochastic
    material texture and spatially correlated observation noise.

## AWGN results

Training only through sigma 50:

| Model | Params | Clean | sigma 50 | sigma 75 | sigma 100 |
|---|---:|---:|---:|---:|---:|
| Standard compact STCNN | 39,777 | 97.55% | 96.35% | 91.11% | 82.96% |
| Paired consistency STCNN | 39,777 | 98.50% | 97.13% | 90.23% | 79.86% |
| Paired + exact sigma | 42,401 | 98.75% | 96.94% | 93.45% | 89.79% |
| Large STCNN teacher | 633,729 | 98.60% | 97.42% | 94.17% | 88.10% |

Paired consistency improves clean and in-range sigma-50 accuracy, but without a
sigma input it becomes biased toward texture beyond the training range. At
sigma 100 its recall remains 98.38% while specificity falls to 61.34%.

Training through sigma 100:

| Model | Params | Clean | sigma 50 | sigma 75 | sigma 100 | sigma 150 |
|---|---:|---:|---:|---:|---:|---:|
| Paired compact STCNN | 39,777 | 98.20% | 97.25% | 95.55% | 93.23% | 85.20% |
| Paired + sigma | 42,401 | 98.05% | 97.11% | 95.29% | 93.15% | 87.00% |
| Large STCNN teacher | 633,729 | 97.85% | 97.40% | 96.43% | 94.33% | 86.57% |
| Paired distilled student | 39,777 | 98.00% | 97.35% | 96.07% | 93.35% | 84.70% |

The large improvement at sigma 100 after expanding the training range shows
that the earlier failure was mostly a coverage problem, not a hard 0.04M model
capacity limit. Distillation closes most of the sigma-75 teacher/student gap,
but does not consistently improve every sigma.

## Noise-family shift

| Model | Clean | AWGN 100 | Correlated G 50 | Poisson peak 10 | S&P 5% | Sine amp 30 |
|---|---:|---:|---:|---:|---:|---:|
| AWGN100 compact STCNN | 98.20% | 93.40% | 55.00% | 97.10% | 95.90% | 85.30% |
| Diverse distilled compact STCNN | 96.30% | 93.95% | 82.11% | 96.21% | 96.41% | 96.07% |
| Diverse + exact sigma (oracle) | 98.50% | 91.66% | 94.06% | 97.50% | 97.25% | 97.66% |
| Diverse large teacher | 97.65% | 94.07% | 89.27% | 97.09% | 97.58% | 95.13% |
| Diverse + sigma jitter | 97.35% | 92.85% | 82.91% | 95.88% | 95.60% | 96.94% |

An AWGN-only model is not generally noise robust. Spatially correlated noise
with approximately the same RMS amplitude collapses its specificity and is the
hardest tested family. Diverse training and distillation improve this condition
substantially, but a large deployable teacher still leads the compact
non-conditioned student by about seven percentage points.

The exact-sigma conditional experiment was repeated with three training seeds:

- Clean: 98.40% +/- 0.10%.
- AWGN 50: 97.24% +/- 0.35%.
- Correlated Gaussian 50: 94.18% +/- 0.98%.
- Sinusoidal amplitude 30: 97.70% +/- 0.40%.

## Noise-level input

An exact or oracle noise input gives a high research upper bound, but it can be
fragile. For true AWGN sigma 50, the diverse exact-sigma model obtains 97.41%
with the oracle per-patch RMS value. Supplying fixed values 25, 50, or 75 gives
62.57%, 93.96%, and 55.80%, respectively.

Training with sigma jitter of standard deviation 10 and 10% random sigma
replacement reduces peak performance, but makes the model much less sensitive.
For the same true sigma-50 input, supplied values 25, 50, and 75 give 94.91%,
95.39%, and 94.39%. Its advantage over the non-conditioned distilled student is
then small, so sigma conditioning is worthwhile only when a reasonably
calibrated estimator is available.

## Synthetic signal ambiguity

Controlled data used smooth planes as non-texture and two clean texture types:
periodic patterns and correlated stochastic material-like fields. Adding these
as training anchors improves recognition of clean synthetic textures, but does
not solve smooth-signal plus correlated-noise false positives.

This is an identifiability limit: a single 32x32 observation of a correlated
random clean texture can be statistically indistinguishable from a smooth
signal plus correlated random noise. A larger classifier cannot recover
information absent from its input. Exact noise metadata, multiple frames,
larger scene context, or a separately estimated noise model is required for a
guarantee in this case.

## Full-image diagnostic

The diverse distilled compact STCNN was tiled over five native-resolution demo
images. Mean binary agreement with its clean-input map was:

| Noise | Agreement | Probability correlation | MAE |
|---|---:|---:|---:|
| sigma 15 | 97.63% | 0.969 | 0.030 |
| sigma 50 | 93.15% | 0.817 | 0.081 |
| sigma 100 | 79.21% | 0.655 | 0.194 |

The original STCNN had only 83.55% agreement at sigma 50, so diverse training
and distillation materially improve full-image stability. However,
`piqsels_grass` receives only 1.7% texture on the clean input. The model may be
suppressing true fine stochastic texture together with noise; no dense GT is
available for those five images.

The sigma-jitter conditional model makes the opposite trade-off:
`piqsels_grass` receives 93.8% texture on clean input, but falls to 34.0% at
sigma 50. Across all five images, its clean-map agreement is 92.51%, 80.68%,
and 77.28% at sigma 15, 50, and 100. It therefore looks more intuitive on this
clean grass example but is less invariant than the non-conditioned distilled
student. Neither can be called correct without dense texture GT.

## Current conclusions

1. Sigma 50 AWGN is learnable by the compact STCNN; it is not near the capacity limit.
2. Training distribution is more important than parameter count for AWGN.
3. A large teacher helps most under noise-family shift, and distillation
   transfers part, but not all, of that advantage.
4. Exact sigma conditioning is powerful but can be dangerously brittle.
5. Noise diversity is necessary. AWGN accuracy alone materially overstates robustness.
6. The hardest limit is distinguishing true stochastic texture from correlated
   noise in one small observation. This cannot be solved solely by scaling the CNN.

## Selected checkpoints

- `checkpoints/stcnn_limit/stcnn_awgn100_paired.pt`: compact model for known AWGN-like use.
- `checkpoints/stcnn_limit/stcnn_diverse_distilled.pt`: compact model without a sigma input.
- `checkpoints/stcnn_limit/stcnn_diverse_sigma_robust.pt`: compact conditional model with sigma jitter.
- `checkpoints/stcnn_limit/stcnn_diverse_conditioned_oracle.pt`: research upper bound; requires accurate sigma.
- `checkpoints/stcnn_limit/stcnn_diverse_teacher.pt`: 633,729-parameter teacher.

Plots, compact metrics, and the full-image gallery are under
`results/stcnn_limit/`.
