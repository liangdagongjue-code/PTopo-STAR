# PTopo-STAR research data and code

Version 1.0.0 accompanies **Periodic-topology self-supervised learning for few-shot mechanical prediction of additively manufactured TPMS metamaterials**.

## Download the complete research package

The code and small analysis tables can be browsed in this GitHub repository.
The complete package, including model checkpoints, features and prediction
archives, is attached to the versioned release:

- Release: https://github.com/liangdagongjue-code/PTopo-STAR/releases/tag/v1.0.0
- Full package: https://github.com/liangdagongjue-code/PTopo-STAR/releases/download/v1.0.0/PTopo_STAR_data_code_v1.0.0.zip

Download and extract the release ZIP before running the verification commands
below. The source-only GitHub ZIP does not include the large result archives.

## Contents

| Directory | Contents |
|---|---|
| formal_results/pretraining | 160 original checkpoints, 200 feature archives, histories and fold manifests |
| formal_results/downstream_shards | 40 predicted stress-trace archives for support draw 0 and 40 split tables |
| formal_results/aggregate | 1,715,340 scalar predictions, 857,670 curve-metric rows, run and family summaries, inference |
| derived_data | Gate choices, 488 unique support/test splits, paper tables and plotting data |
| supplementary_software | Frozen training, prediction, aggregation and figure scripts, geometry implementation, configurations and tests |
| figures, supplementary_figures, tables | English publication assets, editable framework, and table fragments |
| tools | Source-data restoration, isolated training-layout preparation, package verification |
| verification | Independent numerical checks, executable after restoration |

The scalar predictions and curve metrics cover every formal run. Full predicted
stress traces were archived only for support draw 0 (not all 20 support draws).
The public curve archives contain metadata and predictions. Raw experimental
stress, strain and labels are linked to the original source, not copied into
the deposit. DATA_DICTIONARY.txt specifies the schema and removed columns.

## Fast verification without retraining

Python 3.11 or later is recommended. From the extracted package directory:

```bash
python -m pip install -r requirements_verification.txt
python tools/verify_package.py
python tools/restore_experimental_data.py --work-dir WORK
python WORK/restored/verification/verify_recovered_results.py
python WORK/restored/verification/verify_prediction_errors.py
python WORK/restored/verification/independent_data_review.py
```

Replace WORK with a writable directory outside this package (on Windows use
a D-drive path). The restoration command downloads the 9.4 MB pinned source
archive, validates its SHA-256, and creates a separate working copy. It does
not change the public package. A local --source-zip option avoids downloading.
Verification recalculates errors, inference, gate statistics and Figure 4;
it does not train a network. These checks can use several GB of RAM.

## Training from scratch

The formal configuration is in supplementary_software/config.yaml. Install
the pinned training dependencies from supplementary_software/requirements.txt
in a separate environment with a compatible NVIDIA GPU and CUDA runtime.
The historical experiment used an RTX 3090 with 24 GB of memory.

```bash
python tools/prepare_training.py --run-root RUN
cd RUN/experiments/ptopo_star_full
python orchestrate_all.py pretraining --config config.yaml --results-dir results_paper_v1
python orchestrate_all.py downstream --config config.yaml --results-dir results_paper_v1
python aggregate_results.py --config config.yaml --shard-dir results_paper_v1/downstream_shards --output-dir results_paper_v1/aggregate
```

RUN must be a new or compatible writable directory outside this deposit. This
setup downloads the separately sourced experimental records and arranges the
relative paths expected by the original scripts. Training is a separate,
resource-intensive operation; no new training was performed for this release.

## Study and provenance

There are 274 physical designs in eight dominant TPMS families. Pretraining
uses only synthetic geometry, with the held-out primitive coefficient fixed
at zero. Five encoder seeds and 0/1/2/4 target supports define the evaluations.
Every curve model also receives the measured strain path as query coordinates.
The source-selected gate scales the difference between geometry-augmented and
explicit predictions. Algorithmic repeats are not independent physical tests.

Source dataset: https://github.com/maxineAPS/energy-dissipation-metamaterial-dataset/tree/ce432eda188c2eb0b470cf0203442d89c129eaa6

Source paper: https://doi.org/10.1145/3721238.3730759

## Licensing and citation

Data, model parameters, figures and documentation: CC BY 4.0 (LICENSE_DATA.txt).
Software: MIT (LICENSE_CODE.txt). THIRD_PARTY.txt defines the source-data boundary.
Cite the GitHub v1.0.0 release at https://github.com/liangdagongjue-code/PTopo-STAR/releases/tag/v1.0.0
and also cite the upstream experiment. Creators are listed in CITATION.cff.
This release has a versioned HTTPS URL; no DOI has been assigned.

The submission manuscript, editorial correspondence, passwords, server access
details and local working logs are not included. FILE_MANIFEST.csv records each
distributed file's size and SHA-256. RELEASE_REPORT.json records the packaging
checks. REPRODUCTION_CHECKS.json records the tested source-data restoration and
numerical checks. No repository publication is claimed by these local reports.
