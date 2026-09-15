# Published experiment artifacts

This directory contains compact, reviewable copies of the completed experiment outputs. Large DIV2K-derived datasets, full-resolution per-pixel NumPy arrays, caches, and redundant intermediate images are intentionally excluded.

## `paper_v1`

- `metrics.json`: held-out DIV2K classifier and baseline metrics.
- `validation_analysis.json`: CNN/Sobel agreement analysis.
- `history.csv`, `run_config.json`: training trace and configuration.
- `diagnostics_gallery.jpg`: controlled clean-pattern tests.
- `div2k_gallery.jpg`, `external_grass_gallery.jpg`: selected full-resolution inference visualizations.
- `texturesam_compatibility_summary.json`: interface check only; TextureSAM masks are not ground truth.

## `robust_v2`

- `summary.json`: compact combined patch/dense/diagnostic summary.
- `patch_models_summary.json`: all patch model results by noise level.
- `conditional_accuracy_selected_summary.json`: corrected sigma-conditioned checkpoint selection run.
- `metrics.json`, `history.csv`, `patch_label_metrics.json`: dense generator results.
- `controlled_diagnostics.csv`: generator behavior on procedural inputs.
- `accuracy_vs_noise.png`: patch-model robustness curves.
- `validation_sigma_*.jpg`: dense natural-image validation examples.
- `g_diagnostics_sigma*_gallery.jpg`: successful and failed controlled cases.
- `freq_aware_seg_demo_noise_comparison.jpg`: five demo images at sigma 0/15/50.
- `dct_teacher_demo_noise_comparison.jpg`: the exact stride-1,
  overlap-averaged clean DCT teacher applied directly at sigma 0/15/50.
- `spatial32_demo_noise_comparison.jpg/.json`: blockwise full-image diagnostic
  for the Spatial 32x32 classifier at sigma 0/15/50.
- `map_generator_external_grass_gallery.jpg`: full-resolution external inference.
- `texturesam_compatibility_summary.json`: exact-size integration check only.

Model binaries are stored separately under `checkpoints/`.

## `stcnn_limit`

- `summary.json`, `key_metrics.csv`: compact results from the STCNN ablations.
- `stcnn_limit_summary.png`: AWGN limit curve and noise-family comparison.
- `diverse_distilled_demo_awgn.jpg/.json`: native-resolution full-image
  diagnostic at sigma 0/15/50/100.
- `sigma_robust_demo_awgn.jpg/.json`: matching diagnostic for the
  sigma-jitter conditional STCNN.

The complete interpretation and limitations are documented in
`STCNN_LIMIT_REPORT.md`.

## `g_limit`

- `summary.json`, `replicate_summary.csv`: compact capacity, distillation, and
  three-seed summaries.
- `evaluation.json/.csv`: all model/noise measurements.
- `replicate_evaluation.json/.csv`: three independently trained checkpoints for
  each key model.
- `sigma_sensitivity.json/.csv`: supplied-sigma calibration ablation.
- `distillation_evaluation.json/.csv`: large-to-small G comparison.
- `g_limit_summary.png`: capacity, noise-family, repeated-run, and sigma plots.
- `g_limit_awgn_100_gallery.jpg`: representative validation crops at AWGN 100.
- `g_limit_corr_awgn_50_gallery.jpg`: matching correlated-noise comparison.

The complete interpretation is documented in `G_LIMIT_REPORT.md`.

Selected model binaries are under `checkpoints/g_limit/`.
