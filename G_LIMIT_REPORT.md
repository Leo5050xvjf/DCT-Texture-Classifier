# Dense generator G noise/capacity limit study

## Question

This study asks how far the dense texture-map generator `G` can be pushed by:

1. extending the AWGN training range from sigma 50 to sigma 100;
2. supplying an explicit corruption-level scalar;
3. scaling model capacity;
4. training on multiple synthetic noise families; and
5. distilling a larger robust `G` into the original-size model.

The word *limit* is operational. All scores measure reproduction of the fixed
clean teacher map, not agreement with human texture labels.

## Fixed experimental material

- Train: 1,600 clean `128x128` luminance crops from 800 DIV2K training images.
- Validation: 200 clean `128x128` crops from 100 disjoint DIV2K validation images.
- Target: the original clean-only `8x8` frequency-domain DCT classifier,
  evaluated at stride 1 and overlap-averaged into one dense soft map.
- G input: clean or synthetically corrupted luminance in `[0,1]`.
- G output: one dense texture probability map at the input resolution.
- Main loss: paired clean/noisy pixel MSE to the same clean target, plus a
  `0.25` clean/noisy consistency term.
- Evaluation: full 200-crop validation set and three fixed corruption
  realizations. Threshold `0.5` is used for binary agreement.

The target map contains 58.76% positive pixels. Therefore a score of about
58.76% with balanced accuracy near 50% means the model has collapsed to
calling almost everything texture.

## Models

| Model | Base channels | Parameters | Approximate FP32 weights |
|---|---:|---:|---:|
| tiny G | 8 | 190,753 | 0.73 MiB |
| original-size G | 16 | 761,665 | 2.91 MiB |
| conditional G | 16 | 761,825 | 2.91 MiB |
| large G | 32 | 3,043,969 | 11.61 MiB |
| extra-large G | 64 | 12,170,497 | 46.43 MiB |

The conditional model concatenates a constant `sigma/100` channel with the
luminance input. For the controlled experiments it receives the per-crop RMS
difference from the known clean image. This is an oracle upper bound; a real
deployment needs a calibrated noise estimator or known acquisition setting.

## Three-seed results

The following values are mean binary agreement with the clean teacher, with
sample standard deviation across three independently trained models.

| Training/model | Clean | AWGN 50 | AWGN 100 | Correlated Gaussian 50 | Poisson peak 10 | Sinusoid amp. 30 |
|---|---:|---:|---:|---:|---:|---:|
| AWGN small | 90.39 +/- 0.24 | 84.07 +/- 0.48 | 78.17 +/- 0.57 | 58.77 +/- 0.00 | 84.18 +/- 0.17 | 77.88 +/- 1.77 |
| AWGN conditional | 91.50 +/- 0.98 | 84.36 +/- 0.47 | 78.89 +/- 0.22 | 59.37 +/- 0.45 | 83.66 +/- 0.28 | 80.72 +/- 2.38 |
| AWGN large | 93.00 +/- 1.39 | 84.27 +/- 0.26 | 79.10 +/- 0.54 | 58.76 +/- 0.00 | 84.75 +/- 0.13 | 75.76 +/- 4.00 |
| Diverse small | 87.11 +/- 1.30 | 82.19 +/- 0.53 | 75.80 +/- 0.55 | 68.99 +/- 3.41 | 82.69 +/- 0.27 | 82.22 +/- 1.06 |
| Diverse conditional | 91.68 +/- 1.46 | 79.84 +/- 1.88 | 73.56 +/- 3.26 | 73.34 +/- 1.60 | 79.69 +/- 1.75 | 84.50 +/- 0.61 |
| Diverse large | 90.74 +/- 1.30 | 83.34 +/- 0.61 | 78.37 +/- 0.59 | 72.83 +/- 0.49 | 83.74 +/- 0.76 | 83.09 +/- 1.59 |

## Findings

### 1. Covering sigma 100 in training matters more than supplying sigma

The old G, trained on AWGN through sigma 50, scores 62.1% at sigma 100.
With the new paired protocol, a fair sigma-50 small G reaches 74.1%; expanding
the training range to sigma 100 raises the three-seed mean to 78.17%.

For models trained only through sigma 50, explicit sigma raises the sigma-100
score from 74.1% to 75.4%. When both are trained through sigma 100, the
three-seed means are 78.17% and 78.89%. The help is real but small.

### 2. Capacity saturates around the 3M-parameter model

Across representative capacity runs, sigma-100 agreement for 0.19M, 0.76M,
3.04M, and 12.17M models is 77.9%, 77.8%, 79.2%, and 78.1% respectively.
Clean agreement continues to rise from 85.7% to 93.2%, but noisy recovery does
not. The extra-large model learns the clean teacher more precisely without
recovering more information from the highly corrupted observation.

The same saturation appears under diverse training. At correlated Gaussian
sigma 50, representative 3.04M and 12.17M models score 73.2% and 73.1%.
Because the extreme-size checkpoints were not repeated three times, this part
establishes a saturation trend rather than a precise statistically paired
capacity estimate; the small/large comparison above is the repeated result.

### 3. Noise-family coverage is the dominant out-of-distribution factor

All AWGN-only models collapse on correlated Gaussian sigma 50 to approximately
58.76%, effectively predicting the entire image as texture. Increasing G from
0.76M to 12.17M parameters does not fix this.

Six-family training (AWGN, correlated Gaussian, Poisson, speckle, impulse, and
sinusoidal/banding) raises the three-seed correlated-noise score to 68.99% for
small G and 72.83% for large G. This is a much larger effect than either model
scaling or scalar sigma conditioning.

### 4. A scalar sigma cannot describe noise structure

The diverse conditional G improves correlated-noise and sinusoidal results,
but loses AWGN and Poisson accuracy and has more training variance. Different
noise families can share the same RMS strength while having very different
spatial statistics; one scalar cannot identify that distinction.

For nominal AWGN sigma 100, clipping reduces mean observed RMS to about 76.87.
The AWGN-100 conditional model scores 79.3% when supplied 75, but only 69.3%
when supplied 100. The robust-sigma model is less brittle, although its peak
accuracy is lower. Conditional G therefore requires a calibrated definition
of sigma, not merely an approximate nominal label.

### 5. Large-to-small distillation transfers part of the robust behavior

In the selected run:

| Model | Parameters | Clean | AWGN 100 | Correlated Gaussian 50 | Sinusoid amp. 30 |
|---|---:|---:|---:|---:|---:|
| diverse small | 761,665 | 86.0 | 76.0 | 65.1 | 81.5 |
| diverse distilled small | 761,665 | 86.7 | 75.4 | 70.5 | 82.5 |
| diverse large teacher | 3,043,969 | 91.2 | 78.6 | 73.2 | 83.5 |

Distillation closes roughly two thirds of that selected correlated-noise gap,
at the cost of 0.6 percentage point on AWGN 100.

## Current conclusion

The current practical ceiling is not set primarily by parameter count.
For known AWGN through sigma 100, a compact or large model plateaus near 79%
binary agreement with the clean DCT teacher. For mixed synthetic noise, the
3.04M diverse-large G is the most stable overall model. The 0.76M distilled G
is the preferred compact compromise.

The unresolved limitation is information and supervision: one corrupted frame
may not contain enough evidence to distinguish stochastic clean texture from
structured noise, and the fixed clean DCT teacher itself is only an operational
pseudo-label. Stronger claims require real camera noise and human- or
task-validated texture ground truth, not simply a larger G.

## Reproduction

```bash
python train_g_limit.py
python evaluate_g_limit.py
python evaluate_g_sigma_input.py
python train_g_distill.py
python summarize_g_limit.py
python visualize_g_limit.py
python -m unittest discover -s tests -v
```

Detailed machine-readable metrics and galleries are under `results/g_limit/`.
