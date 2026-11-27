# ShareChat RecSys Challenge - solution write-up (team AM)

## 1. What we submitted

We selected two submissions for the private leaderboard and describe both here.

| | **track_b** | **track_a** |
|---|---|---|
| Kaggle notebook | `sharechat-track-b`, version 3 | `sharechat-track-a`, version 3 |
| Public score | **0.89382** | **0.83441** |
| Uses `request_id` or any request-derived feature | **no** | **no** |
| Reads rows other than the one being scored | **yes** (same post; same user at another time) | **no** |
| Model | 11 XGBoost boosters, logit average | 8 XGBoost boosters, logit average |
| Features | 1,697 (three boosters use 1,592) | 751 |

**Neither track reads `request_id`, counts rows per request, compares rows of one request, or groups rows by (user, timestamp), in training or
in inference.** This can be checked directly: `grep -rn "request_id" track_a track_b` returns only doc-strings and the exclusion list in
`elig_feat.py`.

**track_b is not row-independent, and we want that to be unambiguous.** Its extra features are computed from *other rows of the train and test
files*: later and earlier rows of the same post, and rows of the same user at a different timestamp. The Evaluation page asks for predictions from
"only the features in that row"; your eligibility email asks whether request-ID-based features were used. track_b satisfies the second and not the
first. track_a satisfies both. We submit both so that you can rule; the notebooks carry the same disclosure in their first lines.

### Every submission we made (public scores)

**With `request_id`** means the file uses `request_id` or a feature derived from request grouping (request size, within-request statistics).
**Without `request_id`** means it uses neither.

| Date (UTC) | Submission | Public score | `request_id` | Selected |
|---|---|---|---|---|
| 24 Nov 09:27 | `sub1_xgb_cfg0.csv` | 0.88677 | with | no |
| 24 Nov 11:13 | `sub2_blend.csv` | 0.88915 | with | no |
| 24 Nov 14:47 | `sub4_r2_trace_x12_s101.csv` | 0.91408 | with | no |
| 24 Nov 15:37 | `sub5_r2_trace_T16_d6x3.csv` | 0.92316 | with | no |
| 24 Nov 19:55 | notebook `sharechat-nb-requestsize-trace`, version 2 | 0.92029 | with | no |
| 25 Nov 05:38 | notebook `sharechat-nb-darkgray-trace`, version 3 | 0.88967 | without | no |
| 25 Nov 11:35 | `track_b_test1_2models.csv` | 0.89178 | without | no |
| 25 Nov 14:37 | `track_b_final.csv` (the track_b model, uploaded as a file) | 0.89382 | without | no |
| 25 Nov 16:29 | notebook `sharechat-track-a`, version 3 | 0.83441 | without | **yes** |
| 25 Nov 16:57 | notebook `sharechat-track-b`, version 3 | 0.89382 | without | **yes** |

All five submissions with `request_id`, including our three highest public scores (0.92316, 0.92029, 0.91408), were not selected. We disclosed
them in every notebook. Our final entry is the last two rows: 0.89382 and 0.83441, both without `request_id`.

Private scores were not visible to us when this was written.

## 2. Data facts that shaped the solution

* `train.parquet`: 1,537,638 rows over 72 h. `test.parquet`: 1,037,090 rows over the following 78 h. 624 masked columns. Watch rate 4.98 %.
* The seven user and seven post embedding columns (`feature26-32`, `feature40-46`, 32 values each) are the state of an online-trained factorisation
  model, with matched heads user 0->post 1, 1->0, 2->5, 3->2, 4->6, 5->4, 6->3 and bias scalars `feature33-39` (user) and `feature47-53` (post).
* The test file has about 0.67x the row density of the training file and a 5-hour hole (hours 96-100 after the start of train).
* 16 % of test rows are on posts never seen in training (9 % in our hold-out), and such rows are harder for every model we trained.

## 3. Validation

* **Hold-out:** fit on the first 48 h of `train.parquet`, score the last 24 h. No random split: rows of one post or user are strongly dependent.
* **Test-like hold-out (track_b only):** the cross-row features get weaker when a post is re-sampled less often. We therefore re-featurise the
  hold-out rows on three thinned copies of the timeline (70 % of training rows kept at random, two 5-hour holes removed, followed by the label-free
  test rows) and report AUC on those. This is the number we use to predict the leaderboard.
* **Agreement with the leaderboard:** every model family lost about 0.006 from hold-out to public score.

| File | Hold-out | Public | Difference |
|---|---|---|---|
| track_b, 2 models (test submission) | 0.8980 | 0.89178 | -0.0062 |
| track_b, final 11 models | 0.9001 | 0.89382 | -0.0063 |
| track_a | 0.8411 | 0.83441 | -0.0067 |

We traced that difference to the composition of the test period (more rows on new posts), not to a modelling error we could find
(`diagnostics/holdout_vs_leaderboard_offset.py`, `diagnostics/oracle_post_rate.py`).

## 4. track_a: row-independent model

### Features (751) - `track_a/elig_feat.py`
Every value is a function of one input row plus vocabularies frozen from the training file. Full list: `features/track_a_features.tsv`.

| Block | Count | Content |
|---|---|---|
| numeric | 584 | every numeric masked column of the row, unchanged |
| time and age | 7 | minute of day (IST) with sin/cos; post age and account age at request time from the row's own timestamps (`feature1`, `feature57`, `feature58`), raw and log |
| category | 16 | `feature6-14, 16, 17, 19-22, 25` mapped to an integer by frequency rank **in the training file**; unseen = -1, null = -2 |
| embedding geometry | 126 | the row's 7 user x 7 post heads: 49 dot products, 49 cosines, 14 norms, 14 validity flags |
| platform scores | 18 | matched-head scores `<u_a, p_b> + user bias + post bias` (7), `<u_a, p_b> + sum of biases` (7), their max and mean, sum of user biases, sum of post biases |

No target statistics, no test-file statistics, no ID embeddings.

### Model and training
XGBoost `binary:logistic`, histogram method, true `watched` labels only.

| Setting | Depth | Learning rate | min_child_weight | colsample | lambda | subsample | Rounds | Seeds |
|---|---|---|---|---|---|---|---|---|
| d5 | 5 | 0.02 | 50 | 0.5 | 5 | 0.8 | 1640 | 11-15 |
| d7 | 7 | 0.02 | 200 | 0.3 | 20 | 0.8 | 1355 | 21-23 |

Rounds were set by early stopping on the hold-out and multiplied by 1.15 for the refit on all 72 h. The eight boosters are averaged in logit space.
Exact parameters, rounds, seeds and column order are in `models/track_a/B_xgb_d5.json` and `B_xgb_d7.json`.

### What we tried for this track (hold-out AUC)
| Experiment | Result |
|---|---|
| Seven XGBoost settings; their average | 0.8405 - 0.8410; 0.8411 |
| Neural networks with piecewise-linear numeric embeddings; blend with trees | 0.838; 0.8412 |
| Train-frozen post / author target statistics | 0.84095 vs 0.84099 |
| Independent re-implementation, two settings at learning rate 0.05 | 0.84030 / 0.83988; rank correlation 0.9994 with the submitted file |
| Distillation from a cross-row teacher, training time only (baseline 0.84030) | 0.84036; a row-only teacher as control gives 0.84049 |
| Distillation from a request-size-aware teacher, training time only (coarser setting, baseline 0.83935) | 0.83987 |
| Pseudo-labelling the target rows (same coarser setting, baseline 0.83935) | 0.83972 (soft), 0.8256 (hard) |
| Oracle: true watch rate of the same post within +-1 h / +-3 h as a feature | 0.83947 -> 0.83958 / 0.83950 |

The row-only family is saturated at about 0.841. None of the last four rows is used in the submission.

## 5. track_b: cross-row model without request size

### The signals
1. **Embedding trace (same post).** When a user views a post, the online model moves the post's head *b* along the viewing user's paired head *a*
   by `lr * (label - prediction)`. The update lands 3-10 minutes later. The same post appears again in later rows with its updated embedding, so
   `embedding(post, later row) - embedding(post, this row)` contains a small copy of this row's user vector if this row's user engaged.
2. **Counter self-inclusion (same post).** Public counters of the post at a later row include this row's own view.
3. **Other sessions of the same user.** At the user's next session the user's counters and embedding heads reflect what happened in this one.
   Only 2.85 % of test rows have a later session of the same user in the file.

### Features (1,697) - full list in `features/track_b_features_full_1697.tsv`
| Block | Count | Key | Code | Content |
|---|---|---|---|---|
| row | 751 | - | `elig_feat.py` | identical to track_a |
| TL | 55 | same post | `trace_feat.build_trace` | latency-weighted matched filter of the post's later embedding increments against the row's user vector, in a per-post noise-whitened space (global user covariance -> remove global-mean and audience-mean directions -> per-post increment covariance); per head pair (u4p6, u6p3, u5p4) and combined; cumulative versions at 7 horizons; gaps |
| FU | 270 | same post | `trace_feat.build_future` | change of 89 post-level columns at the post's next row, at the first row >= 1 h later, and since its previous row; 3 gaps |
| W | 156 | same post | `feats2.build_w` | aggregates (max / sum / signed root-sum-square) of per-increment whitened cosines and amplitudes on a 15-point time grid, over all increments and latency-restricted windows; cumulative whitened cosines at 15 horizons |
| FH | 360 | same post | `pipeline_b.fut_fixed` | change of the same 89 columns at the first row of the post >= 5 / 10 / 20 / 40 minutes later; 4 gaps |
| UM | 105 | same user, **different timestamp** | `pipeline_b.user_mirror` | change of 40 user-level counters at the user's next and previous session; change of the user's embedding heads at the next session projected on this row's post heads; gaps |

Three of the eleven boosters use every block except UM (1,592 features, `features/track_b_features_counter_only_1592.tsv`).

The UM block skips rows that share the row's own timestamp and never counts rows, so it does not reconstruct request size
(section 4 of `docs/ORGANISER_NOTES.md` shows that counting rows by user and timestamp *would* reconstruct it exactly; we do not do that).

### One timeline for both files
All cross-row blocks are computed once on the concatenation of train and test, sorted by (post, time) or (user, time). No label is involved.
On the hold-out the same feature set scored 0.8978 with features computed inside the training file only (our earlier notebook, finer fit) and
0.9022 on the joint timeline (coarser fit), mostly because rows near the end of a file no longer lose their later snapshots.

### Model and training
XGBoost `binary:logistic`, depth 6, learning rate 0.03, subsample 0.8, colsample 0.4, min_child_weight 50, lambda 5, 256 bins, GPU histogram.
True `watched` labels only. Rounds from early stopping on the test-like hold-out, x1.15 for the refit on all 72 h.

| Boosters | Features | Training rows | Hold-out rounds -> final | Seeds | Ensemble weight |
|---|---|---|---|---|---|
| 6 | 1,697 | all training rows, features at training density | 1998 -> 2298 | 61-66 | 3 each |
| 1 | 1,697 | the same rows plus three test-like-density copies | 2298 | 81 | 3 |
| 1 | 1,697 | three test-like-density copies only | 2298 | 91 | 3 |
| 3 | 1,592 (no UM) | all training rows, features at training density | 1600 -> 1840 | 51-53 | 1.5 each |

Predictions are averaged in logit space with those weights (`models/track_b/track_b_manifest.json`).
Exact commands for every booster: `track_b/TRAINING.md`. Hardware: one Kaggle session with 2 x T4.

### What each step was worth (test-like hold-out)
| Step | AUC | Note |
|---|---|---|
| Row-only model | 0.841 | |
| + TL + FU on the joint timeline, coarse fit (learning rate 0.1, one seed) | 0.8951 | |
| + W | 0.8958 | +0.0007 |
| learning rate 0.1 -> 0.03 | 0.8975 | |
| + FH | 0.8983 | +0.0008, confirmed on two seeds |
| + UM | 0.8999 | hold-out has 6.4 % coverage against 2.85 % in test, so about half carries over |
| final 11-model blend | 0.9001 | public 0.89382 |

### Tested and not used
| Idea | Result (test-like hold-out) |
|---|---|
| Other embedding head pairs | +0.0002 |
| Training only on test-like-density copies | 0.89580 vs 0.89582 |
| Second counter block (more horizons, baseline windows, ratios) | 0.89667 vs 0.89663 |
| Author-keyed counter deltas | 0.89665 vs 0.89663 |
| User bias scalars at the next session | 0.89804 vs 0.89820 |
| Sharpened user-side embedding signal | 0.89840 vs 0.89820 |
| Depth 8 | 0.89629 vs 0.89663 |
| CatBoost member (0.8966 alone) blended in | +0.0002 |
| Distillation from a request-size-aware teacher | 0.8984 vs 0.8982 |
| Pseudo-labelling the target rows | 0.8991 vs 0.8982 on one coarse run; not validated at final settings, not used |

### Where the limit is
The trace lifts AUC by +0.08 on posts next seen 30-120 minutes later and by only +0.02 on posts next seen within 30 seconds: the limit is the
amount of other traffic on a post between two snapshots. The same model scores 0.9046 at the training file's density and 0.8974 at the test file's.

## 6. Features in training versus features at prediction time

You asked us to disclose any difference. **The feature definitions and the code are the same for training rows and evaluation rows in both tracks.**
The points a reviewer should know:

1. **track_a.** Category vocabularies are built from `train.parquet` only and applied to both files. Nothing else is fitted on data.
2. **track_b, joint timeline.** A training row's "later rows of the same post" can be rows of the test file, and a test row's "earlier rows" can be
   rows of the training file. Only their feature columns are read, never a label.
3. **track_b, thinned copies.** Two of the eleven boosters (seeds 81 and 91) were trained on copies of the training timeline in which 30 % of the
   rows and two 5-hour windows were removed before the cross-row features were computed, to imitate the test file's density. Evaluation rows are
   always featurised once, on the full joint timeline.
4. **Labels.** No feature in either track is derived from `watched` (no target encoding, no pseudo-labels, no teacher model).
5. **`request_id`.** Not read by any training or inference code of either track. The scripts under `diagnostics/` do read it, for the measurements
   in `docs/ORGANISER_NOTES.md`; they produce no feature or model used in a submission.

## 7. Reproducing the submissions

* **Inference from the saved boosters** (what the Kaggle notebooks do): `track_a/sharechat-track-a.ipynb` (CPU, about 4 minutes) and
  `track_b/sharechat-track-b.ipynb` (CPU, about 31 minutes; it rebuilds every feature from the two raw files). The track_b notebook reproduced the
  file built during training with rank correlation 1.0000000 and mean absolute difference 2.7e-8.
* **Training track_b from scratch:** `track_b/TRAINING.md` (about 2.5 hours on 2 x T4). GPU histogram training is not bit-reproducible; seeds are fixed.
* **Training track_a:** see the note below.

### One gap we want to state plainly
The eight track_a boosters were trained on 24 November on a different machine, and that training script was not kept with the rest of the work.
What we have and provide: the exact feature code (`track_a/elig_feat.py`), the exact parameters, round counts, seeds and column order
(`models/track_a/B_xgb_*.json`), and the boosters themselves. `track_a/retrain_track_a.py` is an independent re-implementation we ran on
25 November (same features, the two settings at learning rate 0.05, three seeds each); its test predictions have rank correlation 0.9994 with the
submitted file. The same script contains the distillation experiments, which are not part of the submission.

## 8. Notes on request size

While checking our own work we measured what request size is worth, whether it can be obtained without the `request_id` column (it can, exactly),
and how to test a submission for it with and without labels. Those notes are in `docs/ORGANISER_NOTES.md` and in both notebooks.
