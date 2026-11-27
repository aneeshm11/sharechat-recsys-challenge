"""Label-free request-size fingerprint on the TEST file for our own submissions with known public scores (diagnostic only)."""
import glob, json, numpy as np, pandas as pd, pyarrow.parquet as pq
TEST = glob.glob('/kaggle/input/**/test.parquet', recursive=True)[0]
t = pq.read_table(TEST, columns=['id', 'request_id']); ids = t.column('id').to_numpy(); rq = pd.factorize(t.column('request_id').to_numpy())[0]
n = pd.Series(rq).map(pd.Series(rq).value_counts()).values; nb = np.minimum(n, 6)
NB = '/kaggle/input/notebooks/cassinicraft/notebookb2a18421d6/subs'
F = {'row-only (track_a family, 6 models)': ('/kaggle/working/run/s1/out/submission_S_pure.csv', 'about 0.835 (not scored)'),
     'track_b final (cross-row, no request size)': ('/kaggle/working/subs/track_b_final.csv', '0.89382'),
     'our disclosed file with request size (sub1)': (f'{NB}/sub1_xgb_cfg0.csv', '0.88677'),
     'our disclosed file with request size (sub2)': (f'{NB}/sub2_blend.csv', '0.88915'),
     'our disclosed file with request size + trace (sub4)': (f'{NB}/sub4_r2_trace_x12_s101.csv', '0.91408')}
out = {}
print('test rows by request size 1..6+:', ' '.join(f'{(nb == k).mean():.3f}' for k in range(1, 7)))
for nm, (p, lbs) in F.items():
    s = pd.read_csv(p).set_index('id').loc[ids, 'watched'].values.astype(np.float64)
    r = pd.Series(s).rank(pct=True).values                                   # rank-based, so files on different scales are comparable
    m = [float(s[nb == k].mean()) for k in range(1, 7)]; mr = [float(r[nb == k].mean()) for k in range(1, 7)]
    top = [float((r[nb == k] > 0.95).mean()) for k in range(1, 7)]           # share of the request-size group that is in the file's top 5 %
    out[nm] = {'public': lbs, 'mean_pred': m, 'mean_rank': mr, 'top5_share': top, 'index_top5_n1_over_n6': top[0] / max(top[5], 1e-9), 'spearman_with_n': float(pd.Series(s).corr(pd.Series(n.astype(float)), method='spearman'))}
    print(f'{nm:52s} public {lbs:22s} | share of rows in the file\'s top 5 % by request size 1..6+: ' + ' '.join(f'{x:.3f}' for x in top) + f' | ratio n=1 / n=6+: {top[0] / max(top[5], 1e-9):6.1f} | Spearman(score, n) {out[nm]["spearman_with_n"]:+.3f}')
json.dump(out, open('/kaggle/working/run/diag1/diag3.json', 'w'), indent=1)
