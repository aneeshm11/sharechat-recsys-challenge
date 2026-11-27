# Features used to make predictions on the evaluation set

Complete lists, one line per feature in the order the models read them, with the block each belongs to:

* [`features/track_a_features.tsv`](features/track_a_features.tsv) - 751 features (all 8 track_a boosters)
* [`features/track_b_features_full_1697.tsv`](features/track_b_features_full_1697.tsv) - 1,697 features (8 of the 11 track_b boosters)
* [`features/track_b_features_counter_only_1592.tsv`](features/track_b_features_counter_only_1592.tsv) - the first 1,592 of those (3 track_b boosters)

No feature in either track is derived from `request_id`, from request size, from a comparison of rows of one request, or from the label.

## Row-level features - both tracks (751)

Computed by `elig_feat.py`. Each value depends on one input row and on vocabularies frozen from the training file.

| Block | Count | Names | Definition |
|---|---|---|---|
| numeric | 584 | `feature3` ... (masked names) | every numeric column of the row except `id`, `request_id`, `watched` and the timestamp `feature1`, cast to float32 |
| time and age | 7 | `t_min, t_sin, t_cos, post_age_h, post_age_log, user_age_h, user_age_log` | minute of day in IST from `feature1`; hours between `feature1` and the post timestamp `feature58` / the account timestamp `feature57` |
| category | 16 | `cat_feature6` ... `cat_feature25` | integer code = frequency rank of the hashed value **in train.parquet**; unseen -1, null -2 |
| embedding geometry | 126 | `dot_u{a}p{b}`, `cos_u{a}p{b}`, `nu{a}`, `np{b}`, `vu{a}`, `vp{b}` | dot products and cosines of the row's 7 user heads x 7 post heads; norms; validity flags |
| platform scores | 18 | `plat_u{a}p{b}_bu{a}_bp{b}`, `plat_u{a}p{b}_sumb`, `plat_max`, `plat_mean`, `bias_u_sum`, `bias_p_sum` | matched-head dot product plus the row's bias scalars |

## Cross-row features - track_b only (946)

All are computed on one timeline made of the rows of `train.parquet` followed by the rows of `test.parquet`. They read feature columns of
other rows; they never read a label. "Later row of the same post" always means a row with the same `feature18` and a later `feature1`.

| Block | Count | Grouping key | Names | Definition |
|---|---|---|---|---|
| TL | 55 | post (`feature18`) | `tr_z_*`, `tr_amp_*`, `tr_info_*`, `tr_zbest_*`, `tr_un_*`, `tr_c{h}_*`, `tr_gap_next`, `tr_gap_900`, `tr_ninc` | matched filter of the post's later embedding increments against this row's user head, with a latency prior peaking at 5-7.5 min, in a per-post noise-whitened space; head pairs u4p6, u6p3, u5p4 and their combinations; cumulative cosines at 300-3600 s |
| FU | 270 | post | `fn_<col>`, `f1_<col>`, `fp_<col>`, `fu_gap_*` | for 89 post-level columns: value at the post's next row minus own; value at the first row >= 1 h later minus own; own minus value at the previous row |
| W | 156 | post | `w_ninc_*`, `w_nval_*`, `wa_*`, `wc_n{h}_*` | on a grid of later snapshots (0, 30 s ... 2 h): whitened cosine and amplitude of each increment with the row's user head, aggregated (max, sum, signed root-sum-square) over all increments and over latency windows; cumulative cosines at 15 horizons |
| FH | 360 | post | `fh{300,600,1200,2400}_<col>`, `fh{h}_gap` | for the same 89 columns: value at the first row of the post >= h seconds later minus own |
| UM | 105 | user (`feature5`), rows at a **different** timestamp only | `um_gap_*`, `um_dn_<col>`, `um_dp_<col>`, `um_cos_*`, `um_dot_*`, `um_dnorm_*`, `um_pcos_*` | for 40 user-level columns: value at the user's next session minus own, and own minus value at the previous session; change of the user's embedding heads at the next session projected on this row's post heads |

The 89 post-level columns are listed in `models/track_b/fut_cols.json`; the 40 user-level columns are the `um_dn_*` names in the feature list.

### What the UM block does not do
Rows of one request share the user and the timestamp. The UM block looks only at rows of the same user whose timestamp differs from the
row's own, and it never counts rows, so it contains no request size. `docs/ORGANISER_NOTES.md`, section 5a, shows the request-size
fingerprint of the final track_b file on the test set: it is the same as that of a row-only model.
