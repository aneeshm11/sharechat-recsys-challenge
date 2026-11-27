# Trained model artefacts

All files are in [`models/`](models); SHA-256 checksums are in [`models/SHA256SUMS.txt`](models/SHA256SUMS.txt).
The boosters are XGBoost 3.4 binary models (`.ubj`), loadable with `xgboost.Booster().load_model(path)`.

## track_a - `models/track_a/`

| File | What it is |
|---|---|
| `B_xgb_d5_s11.ubj` ... `s15.ubj` | 5 boosters, depth 5, 1640 rounds, seeds 11-15 |
| `B_xgb_d7_s21.ubj` ... `s23.ubj` | 3 boosters, depth 7, 1355 rounds, seeds 21-23 |
| `B_xgb_d5.json`, `B_xgb_d7.json` | parameters, rounds, seeds and the 751 column names in order |
| `artifacts.json` | the numeric column list and the category vocabularies (hashed value -> code) built from `train.parquet` |

Submission = logit average of the 8 boosters with equal weight.

## track_b - `models/track_b/`

| File | Features | Training rows | Rounds | Weight |
|---|---|---|---|---|
| `pf_wfud_lr030_s61.ubj` ... `s66.ubj` (6) | 1,697 | all training rows, features at training density | 2298 | 3 |
| `pf_wfum_lr030_s81.ubj` | 1,697 | the same plus three test-like-density copies | 2298 | 3 |
| `pf_wfut_lr030_s91.ubj` | 1,697 | three test-like-density copies only | 2298 | 3 |
| `pf_wfd_lr030_s51.ubj` ... `s53.ubj` (3) | 1,592 | all training rows, features at training density | 1840 | 1.5 |

| File | What it is |
|---|---|
| `track_b_manifest.json` | the list of boosters, their feature configuration and ensemble weight, as read by the notebook |
| `prod_config_*.json` | per configuration: learning rate, rounds and the feature names in order |
| `artifacts.json`, `fut_cols.json` | category vocabularies from `train.parquet`; the 89 post-level columns used by the counter blocks |

Submission = weighted logit average of the 11 boosters.

## The same files on Kaggle

| | Kaggle location |
|---|---|
| track_a notebook | https://www.kaggle.com/code/cassinicraft/sharechat-track-a (version 3) |
| track_a models | https://www.kaggle.com/datasets/cassinicraft/sharechat-rowonly-b-models |
| track_b notebook | https://www.kaggle.com/code/cassinicraft/sharechat-track-b (version 3) |
| track_b models and code | https://www.kaggle.com/datasets/cassinicraft/sharechat-track-b-models |

The Kaggle notebooks and datasets are private; we will share them with any Kaggle account you name. The files in this repository are
byte-identical copies of the dataset files.

## Intermediate artefacts not included

Feature matrices (about 27 GB) are not stored; both notebooks rebuild them from the competition files. No competition data is included
in this repository.
