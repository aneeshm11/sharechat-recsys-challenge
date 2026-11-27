# track_a - row-independent track

| File | Purpose |
|---|---|
| `sharechat-track-a.ipynb` | the submitted notebook (version 3): builds the 751 row features of `test.parquet` and averages the 8 boosters in `models/track_a/` |
| `elig_feat.py` | the feature code, used identically for training and inference. Reads one row at a time; `request_id` is in its exclusion list and is never read |
| `retrain_track_a.py`, `xgbjob.py` | independent re-training of the two settings (run on 25 November) and the distillation experiments quoted in the write-up |

## About the training code of the submitted boosters

The eight submitted boosters were trained on 24 November on another machine; that script was not kept. Their exact configuration is in
`models/track_a/B_xgb_d5.json` and `B_xgb_d7.json` (parameters, rounds, seeds, column order), and the procedure was: features from
`elig_feat.py`, true `watched` labels, early stopping on the last 24 h of `train.parquet`, refit on all 72 h with 1.15 x the stopped rounds.

`retrain_track_a.py` follows the same procedure with a learning rate of 0.05 instead of 0.02 and three seeds per setting. Its test predictions
have rank correlation 0.9994 with the submitted file (hold-out 0.84030 / 0.83988 for the two settings, 0.84057 blended).

To run it, the worker script has to be embedded first (this is what `track_b/pack.py` does for the track_b pipeline):

```
python -c "import base64; b = base64.b64encode(open('xgbjob.py','rb').read()).decode(); open('run_retrain.py','w').write('MODE = \'S\'\nXGBJOB_B64 = \'' + b + '\'\n' + open('retrain_track_a.py').read())"
SC_DS=<folder with elig_feat.py, trace_feat.py, artifacts.json, fut_cols.json>  python run_retrain.py
```

The script also trains a cross-row teacher on the training file for the distillation experiment; that part needs `trace_feat.py` from `track_b/`.
The pure hard-label models it writes (`submission_S_pure.csv`) do not depend on it.
