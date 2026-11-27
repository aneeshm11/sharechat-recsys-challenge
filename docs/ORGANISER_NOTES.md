
---
# Reading notes for the organisers: request size, row-dependence and the shape of the leaderboard

These notes report **diagnostics**, run on 25 Nov 2026. They are not part of this notebook's inference. Some of them read `request_id` of the
*training* file to measure what request size is worth; neither of our two tracks uses it. Everything below is either measured by us (stated with
numbers) or labelled as inference. We have not seen any other team's code, and **nothing here is a statement about a particular team**.

Hold-out used throughout: models fit on the first 48 h of `train.parquet`, scored on the last 24 h. XGBoost depth 6, learning rate 0.1, one seed
(the same coarse setting for every row of every table, so differences are comparable; finer settings add about 0.002 to every row).
Two of our files have been scored publicly, at 0.0062 and 0.0063 below their hold-out, so `hold-out - 0.006` is our estimate of the public score.

## 1. The public leaderboard has four score groups separated by empty gaps (snapshot 25 Nov 15:10 UTC, 120 teams)

| Public score group | Teams | Share | Gap to the next group below |
|---|---|---|---|
| 0.9188 - 0.9260 | 4 | 3.3 % | 0.0127 (nobody between 0.9061 and 0.9188) |
| 0.9001 - 0.9061 | 2 | 1.7 % | 0.0103 (nobody between 0.8898 and 0.9001) |
| 0.8679 - 0.8898 (median 0.8849, middle half 0.8835 - 0.8863) | 39 | 32.5 % | **0.0315** (nobody between 0.8363 and 0.8679) |
| 0.8323 - 0.8363 (median 0.8348) | 47 | 39.2 % | |
| below 0.832 | 28 | 23.3 % | |

**45 of 120 teams (37.5 %) score above the 0.8363 group.** One of the 45 is this team: our best public score (0.92316) comes from a file that uses
request size, which we disclosed and do not put forward. All four public notebooks of this competition are row-independent (we read their code: none
uses `request_id` as a model input) and report 0.82 - 0.842 locally, so no public recipe produces a score above the 0.8363 group.

## 2. What each kind of information reaches on the same hold-out

| Information given to the model | Hold-out AUC | Estimated public | Where that lands |
|---|---|---|---|
| The row's own columns only (track_a) | 0.8394 | 0.834 | the 0.8323 - 0.8363 group |
| Row + request size *predicted from the row* (out-of-fold) | 0.8371 | 0.831 | same group: request size is not in the row (section 3) |
| Row + later counters of the same post (no embedding trace) | 0.8449 | 0.838 | just above that group, inside the empty gap |
| Row + cross-row trace, counters, same-user sessions (track_b), no request size | 0.8982 (test-like density) | 0.892 | measured: our files scored **0.89178** and **0.89382** |
| **Row + request size (3 columns: n, 1/n, log n)** | **0.8878** | 0.882 | the 0.8679 - 0.8898 group |
| Row + request size + comparisons within the request | 0.8883 | 0.883 | the 0.8679 - 0.8898 group |
| Row + request size + later counters of the same post | 0.8912 | 0.885 | upper part of the 0.8679 - 0.8898 group |
| track_b blocks + request size | 0.9250 (test-like density) | 0.919 | the groups above 0.90 |
| track_b blocks + request size + within-request comparisons of the trace | 0.9307 (training density) | | measured: our disclosed request-size files scored 0.91408, 0.92029, 0.92316 |

Reading: the three request-size columns alone are worth **+0.0484** on top of the full row-level model, and that single step is what crosses the empty gap between
0.8363 and 0.8679. Without request size, the highest score we could build after a full day of exhaustive cross-row work is 0.89382.

## 3. Request size is an outcome of the session and cannot be derived from a row

* Watch rate by request size in the training file: n = 1, 2, 3, 5, 10+: 15.2 % | 11.1 % | 7.5 % | 3.3 % | 0.6 %. Request size alone, used as a score, has AUC 0.757.
* A model predicting request size from the row's own 751 features reaches R2 = 0.29 on log(n) (Spearman 0.55); feeding that prediction to the
  row-level model changes the hold-out from 0.8394 to 0.8371. The true value moves it to 0.8878.
* So a model cannot learn request size from a row. It has to be given rows of the same request.

## 4. Request size can be rebuilt exactly WITHOUT reading the `request_id` column

Rows of one request share the same user id and the same timestamp, and are adjacent in both files. We measured:

| File | Rows | Requests | Rows where `count(rows with same user and timestamp)` equals the true request size | Same, counting adjacent rows only |
|---|---|---|---|---|
| train.parquet | 1,537,638 | 439,041 | 100.0000 % | 100.0000 % |
| test.parquet | 1,037,090 | 297,707 | 100.0000 % | 100.0000 % |

**A solution can therefore contain request size while never touching the `request_id` column.** Checking for the column name is not sufficient; any
grouping or counting by (user, timestamp), by a window of identical timestamps, or by file position is the same feature. Neither of our tracks groups
rows by user and timestamp: track_b's same-user block only looks at rows of that user at a *different* timestamp, and never counts rows.

## 5. Two tests the organisers can run on any submission

**5a. A fingerprint that needs no labels.** Take a submission file and the request size of every test row. A model that was given request size puts
almost all of its confident rows into small requests. Measured on `test.parquet` for five of our own files with known public scores:

| Our file | Public score | Spearman(score, request size) | Share of each request-size group (1 / 2 / 3 / 4 / 5 / 6+) inside the file's top 5 % | Ratio n = 1 over n = 6+ |
|---|---|---|---|---|
| row-only (track_a family, 6 models) | about 0.835 (not scored) | -0.103 | 0.094 / 0.085 / 0.078 / 0.054 / 0.053 / 0.022 | 4 |
| track_b final (cross-row, no request size) | 0.89382 | -0.127 | 0.096 / 0.085 / 0.074 / 0.055 / 0.049 / 0.024 | 4 |
| our disclosed file with request size (sub1) | 0.88677 | -0.463 | 0.243 / 0.150 / 0.070 / 0.021 / 0.005 / 0.000 | 739 |
| our disclosed file with request size (sub2) | 0.88915 | -0.458 | 0.246 / 0.150 / 0.069 / 0.020 / 0.005 / 0.000 | 764 |
| our disclosed file with request size + trace (sub4) | 0.91408 | -0.440 | 0.213 / 0.138 / 0.076 / 0.032 / 0.014 / 0.003 | 83 |

The two files without request size have the same fingerprint (correlation about -0.1, ratio about 4) although one scores 0.835 and the other 0.894;
the three files with request size are at -0.44 to -0.46 with ratios of 80 to 760. The fingerprint therefore flags request size itself, not a high score
and not row-dependence in general. Request size has 44 % of test rows in the 6+ group, so the statistic is very stable.

**5b. A test with the labels.** Fit, on half of the requests, a logistic regression of the label on `[logit(score), log(request size), their product]`;
measure AUC on the other half. If the score already contains request size the gain is nil; if it does not, the gain is large. On our hold-out:

| Model behind the score | AUC of the score | AUC with request size added | Gain |
|---|---|---|---|
| row only (track_a) | 0.8395 | 0.8779 | **+0.0384** |
| row + later counters, no request size | 0.8453 | 0.8818 | **+0.0365** |
| track_b (cross-row, no request size) | 0.9066 | 0.9255 | **+0.0189** |
| row + request size | 0.8879 | 0.8879 | **-0.0000** |
| row + counters + request size | 0.8910 | 0.8910 | **-0.0000** |
| track_b blocks + request size | 0.9310 | 0.9310 | **-0.0000** |

The same split is visible without any fitting: the mean prediction of a request-size model falls with n as steeply as the true rate (15 % at n = 1 to
below 1 % at n = 10+), whereas a model without it stays much flatter.

## 6. Ideas the hosts were asked about and did not rule on - measured, and not used by either track

| Idea | Row-only model (0.8394) | Cross-row model |
|---|---|---|
| Pseudo-labelling the target-period rows with the model's own soft predictions | 0.8397 (hard labels: 0.8256) | 0.8991 vs 0.8982 |
| Distillation from a request-size-aware teacher (out-of-fold, training time only), weight 0.5 / 1.0 | 0.8399 / 0.8390 | 0.8984 vs 0.8982 |

Distillation moves nothing in either model. Pseudo-labelling moves the row-only model by +0.0004 (noise level for one seed) and the cross-row model by
+0.0009 at this coarse setting. We did not use it in either track: track_a is trained on true training labels only by design, and for track_b it could
not be validated at the final settings before the deadline.

## 7. What we conclude, and what we cannot

* **Measured:** no row-independent model we could build exceeds 0.841 on the hold-out (seven XGBoost settings, neural networks, distillation, pseudo-labelling,
  oracle features); 47 teams sit in the matching public group and nobody sits in the 0.0315 gap above it.
* **Measured:** request size, three columns, moves the same model to 0.8878, into the next group.
* **Measured:** the best model we could build without request size, using other rows of the same post and other sessions of the same user, scores 0.89382
  publicly. Six teams (including our own disclosed request-size file) are above that.
* **Inference:** every score above 0.8363 uses information from other rows. For the 39 teams at 0.8679 - 0.8898 the simplest explanation is request size;
  a cross-row trace without request size (our track_b) also lands there, so a score alone cannot tell the two apart - the test in section 5 can.
* **Inference:** scores above about 0.895 need request size (by `request_id` or by an equivalent key, section 4) together with further cross-row signal.
* **We cannot** verify how any other team obtained its score.
