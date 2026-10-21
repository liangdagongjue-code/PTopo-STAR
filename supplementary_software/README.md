# PTopo-STAR paper experiment

This directory contains the dominant-family transfer experiment for TPMS-274. Synthetic pretraining excludes the held-out primitive; labelled source mixtures can contain it as a non-dominant component. The included `latticetopo_pilot` module supplies the geometry dependency.

## Frozen protocol

- Eight outer folds use the dominant coefficient among Gyroid, Diamond, Primitive, IWP, Neovius, Fischer-Koch S, FRD and PMY.
- In fold `k`, every unlabeled pretraining geometry has coefficient `k == 0` exactly. The loader refuses any cache that violates this condition.
- Physical labels from the held-out dominant family are absent except for the deterministic 0/1/2/4-shot support set. All methods share the same support IDs.
- Five encoder seeds and 20 support draws are algorithmic repetitions; they are not counted as additional physical experiments.
- Primary endpoint: official TPMS-274 energy dissipation. Secondary endpoints: peak stress and the full 120-point stress trace.

## SSL ablations

- `masked_only`: random block reconstruction.
- `periodic_only`: spatially clustered boundary-wrapping masks plus synchronized periodic-shift consistency.
- `topology_only`: random masks plus geometry-derived periodic topology descriptors.
- `ptopo_ssl`: boundary-wrapping masks, periodic consistency and topology descriptors.

All ablations use the same 3-D encoder capacity, initialization rule, unlabeled sample count and optimization schedule.

## PTopo-STAR adaptation

The explicit predictor uses the eight published TPMS mixture weights. The augmented predictor uses weights plus the frozen PTopo code. For the full-curve task, every method additionally receives that sample's known 120-point strain trace as query coordinates; stress is always the prediction target. Their nested difference is the residual:

`r = f_aug(weights, z) - f_exp(weights)`

The final prediction is `f_exp + g(d, u) r`. Distance `d` and bootstrap disagreement `u` use 90th-percentile training reference values. For the outer few-shot predictor, that reference includes source and target-support designs. With the residual enabled, each penalty starts attenuating above 1. Gate hyperparameters use leave-one-source-family-out validation and never inspect held-out-family test labels.

Baselines include explicit Ridge, fixed 200-tree ExtraTrees models, handcrafted periodic descriptors, random-encoder fusion, masked-only fusion and the two single-objective SSL ablations. The scalar tree predicts energy and peak stress from the eight explicit weights; the curve tree predicts the 120 stress values from the weights and known strain trace. This keeps the strain conditioning identical in principle across all curve models and prevents variation in experimental query grids from being misattributed to geometry.

## Reproduction

For a new training run, place this directory at `RUN_ROOT/experiments/ptopo_star_full`, the included geometry module at `RUN_ROOT/experiments/latticetopo_pilot`, and the separately obtained source data at `RUN_ROOT/datasets/processed/tpms274_raw/dataset`. The original scripts infer `RUN_ROOT` from their directory. Run the following from `RUN_ROOT/experiments/ptopo_star_full`:

```bash
python orchestrate_all.py pretraining --config config.yaml --results-dir results_paper_v1
python orchestrate_all.py downstream --config config.yaml --results-dir results_paper_v1
python aggregate_results.py --config config.yaml --shard-dir results_paper_v1/downstream_shards --output-dir results_paper_v1/aggregate
```

Individual fold/seed shards are resumable:

```bash
python run_pretraining.py --config config.yaml --output-dir results_paper_v1/pretraining --fold 0 --seed 20260901
python run_downstream.py --config config.yaml --pretraining-dir results_paper_v1/pretraining --output-dir results_paper_v1/downstream_shards --fold 0 --seed 20260901
```

The supplied verification reports concern the recovered formal outputs. They do not claim that model training was repeated during this revision.

## Data provenance

- Physical dataset commit: `ce432eda188c2eb0b470cf0203442d89c129eaa6`
- Official modeling-code commit: `c4f3deaaad05a9d81faea4534887aa8f5544369f`
- Dataset ZIP SHA-256: `C983F31CE9C32C106BA78B068C6E7A83369A1C87BF89BAAEAFCFBB0F8F3386B2`
- Source paper DOI: `10.1145/3721238.3730759`

The stored energy label is verified sample-by-sample against the authors' loading-minus-reversed-unloading trapezoidal objective. Slice-100K and V7-40 are not used in this paper experiment.

## Revised framework

Figure 1 is edited in `figures/figure1_framework.pptx`. Preserve the author-edited connectors by exporting that slide directly from PowerPoint as `figures/figure1_framework.pdf`. Run `python make_framework_figure.py --package-root PATH_TO_PACKAGE` to derive SVG, PNG and TIFF files from the exported vector PDF. This script does not redraw the layout or change the PPTX/PDF. PNG and TIFF are rendered at 1200 dpi for a 180 mm figure width.
