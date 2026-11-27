# track_b - how every booster was trained

Everything runs through one file. `pipeline_b.py` is the readable source; `pack.py` prepends the worker scripts (`xgbjob.py`, `feats2.py`,
`catjob.py`) to it and writes the single file `track_b_pipeline.py`, which is what was executed and what the Kaggle notebook runs.
Behaviour is selected with environment variables. Inputs: the two competition parquet files, `elig_feat.py`, `trace_feat.py`,
`artifacts.json`, `fut_cols.json` (set `SC_DS` to their folder). All runs shared one scratch folder (`SC_TMP`). Hardware: 2 x T4.

No stage reads `request_id`.

## Which file is which

| File | Status |
|---|---|
| `track_b_pipeline.py` | **the file the submitted notebook ran**, byte-identical to the copy in the Kaggle dataset `sharechat-track-b-models` |
| `pipeline_b.py` | the readable source embedded in that file. Identical to it except for the header comment (see below) |
| `feats2.py`, `xgbjob.py`, `catjob.py`, `elig_feat.py`, `trace_feat.py`, `pack.py` | the modules the single file embeds or imports, byte-identical to the dataset copies |
| `build_ensemble.py` | assembled the submission from saved test predictions during training |
| `pipeline_b_later_experiments.py` | a later copy of the source with two experiment additions made after the notebook's file was frozen (a sharpened same-user block and a CatBoost fit on a row subsample). Both are listed under "tested and not used" in the write-up; neither touches the submission |

One correction to read alongside the code: the header comment of the executed file says "no grouping by user". It was written before the
same-user block (`user_mirror`, the UM features) was added and is out of date. The executed file is left untouched; the header of
`pipeline_b.py` has been corrected, and that comment is the only difference between the two.

Each booster was trained on 25 November with the version of the file current when it was launched. The final file contains every stage
they used (`feat`, `go2`).

## 1. Features (about 30 minutes)

```
SC_STAGES=feat  python track_b_pipeline.py
```
Builds the 751 row features of both files, then the cross-row blocks TL, FU, W on (a) the full joint train+test timeline and
(b) three test-like copies (70 % of training rows kept, two 5-hour holes, followed by the test rows). The FH and UM blocks are built
on first use by the stages below. Posts are split into partitions that fit a 16 GB GPU; eigendecompositions run on the CPU.

## 2. Boosters (learning rate 0.03, depth 6; early stopping on the test-like hold-out, final rounds = 1.15 x)

| Manifest file | Command (`SC_STAGES=go2 SC_LR=0.03` in all) | Hold-out (test-like / training density) | Final rounds |
|---|---|---|---|
| `pf_wfd_lr030_s51..s53` | `SC_CFG=wf SC_TRAINSET=dense SC_HSEEDS=51 SC_FSEEDS=51,52,53` | 0.89832 / 0.90598 at 1600 rounds | 1840 |
| `pf_wfud_lr030_s61..s63` | `SC_CFG=wfu SC_TRAINSET=dense SC_HSEEDS=61 SC_FSEEDS=61,62,63` | 0.89992 / 0.90752 at 1998 rounds | 2298 |
| `pf_wfud_lr030_s64, s65` | `SC_CFG=wfu SC_TRAINSET=dense SC_ROUNDS=2298 SC_FSEEDS=64,65` | (round count reused) | 2298 |
| `pf_wfud_lr030_s66` | `SC_CFG=wfu SC_TRAINSET=dense SC_ROUNDS=2298 SC_FSEEDS=66` | (round count reused) | 2298 |
| `pf_wfum_lr030_s81` | `SC_CFG=wfu SC_TRAINSET=mix SC_ROUNDS=2298 SC_FSEEDS=81` | (round count reused) | 2298 |
| `pf_wfut_lr030_s91` | `SC_CFG=wfu SC_TRAINSET=thin SC_ROUNDS=2298 SC_FSEEDS=91` | (round count reused) | 2298 |

`SC_CFG`: `wf` = row + TL + FU + W + FH (1,592 features); `wfu` = the same + UM (1,697).
`SC_TRAINSET`: `dense` = all training rows with features at training density; `thin` = the three test-like copies; `mix` = both.
Other parameters: `subsample 0.8, colsample_bytree 0.4, min_child_weight 50, reg_lambda 5, max_bin 256, tree_method hist, device cuda`.

## 3. Ensemble and submission file

Logit average with weight 3 for the eight `wfu` boosters and 1.5 for the three `wf` boosters (`models/track_b/track_b_manifest.json`).
During training the file was assembled by `build_ensemble.py` from the saved test predictions; the notebook reproduces it from the raw data:

```
SC_STAGES=feat,predict SC_NO_COPIES=1 SC_NO_V=1 SC_MODELS=<folder with the boosters and the manifest>  python track_b_pipeline.py
```
`SC_NO_COPIES` skips the thinned copies (only needed for training); `SC_NO_V` skips an unused feature block.

## Other stages in the file

`go`, `abl`, `prod` (first-round ablations at learning rate 0.1), `xc`, `x1`, `x2`, `x3` (feature and hyper-parameter experiments quoted in the
write-up), `cb` (CatBoost experiment). None of them produces a model used in the submission. Logs of all runs are in `logs/`.
