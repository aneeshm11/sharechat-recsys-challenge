import numpy as np, pandas as pd
B = '/kaggle/input/datasets/cassinicraft/recsysbackup'
m = pd.read_parquet(f'{B}/meta.parquet')
H = 3600_000
tr = m[m.split == 0]; te = m[m.split == 1]
print('train rows', len(tr), 'hours', tr.ts.min() / H, tr.ts.max() / H, '| test rows', len(te), 'hours', te.ts.min() / H, te.ts.max() / H)
print('train target rate', tr.watched.mean(), 'requests', tr.rq.nunique(), te.rq.nunique())
# rows per hour, by absolute hour
hr = (m.ts // H).astype(int)
c = m.groupby([m.split, hr]).size().unstack(0).fillna(0).astype(int)
print('rows per hour (index = hours since train start; cols split 0/1)')
print(c.T.to_string(max_cols=200))
# hour-of-day matched ratio: test hour h vs mean of train hours with same hour-of-day
ctr = c[0][c[0] > 0]; cte = c[1][c[1] > 0]
hod_tr = ctr.groupby(ctr.index % 24).mean()
ratio = cte / hod_tr.reindex(cte.index % 24).values
print('test/train rows-per-hour ratio by test hour: quantiles', np.round(ratio.quantile([0, .1, .25, .5, .75, .9, 1]).values, 3))
print(np.round(ratio.values, 2))
# posts
for name, d in (('train', tr), ('test', te), ('val(48h+)', tr[tr.ts >= 48 * H]), ('fit', tr[tr.ts < 48 * H])):
    d = d.sort_values(['feature18', 'ts'])
    gap = (d.ts.shift(-1) - d.ts).where(d.feature18.shift(-1) == d.feature18) / 1000.0
    has10 = None
    print(f'{name}: posts {d.feature18.nunique()} rows/post median {d.groupby("feature18").size().median():.0f} | gap-next NaN {gap.isna().mean():.4f} '
          f'median {gap.median():.0f}s q25 {gap.quantile(.25):.0f} q75 {gap.quantile(.75):.0f} | P(gap<600s) {(gap < 600).mean():.4f} P(gap<180s) {(gap < 180).mean():.4f}')
# number of later rows of same post within (180s, 600s]
def cnt_window(d, lo, hi):
    d = d.sort_values(['feature18', 'ts']); key = d.feature18.values.astype(np.int64) * 10**10 + d.ts.values
    a = np.searchsorted(key, key + lo * 1000, side='right'); b = np.searchsorted(key, key + hi * 1000, side='right')
    return (b - a)
for name, d in (('fit', tr[tr.ts < 48 * H]), ('val', tr[tr.ts >= 48 * H]), ('test', te)):
    k = cnt_window(d, 180, 600)
    print(f'{name}: later same-post rows in (3,10] min: mean {k.mean():.2f} median {np.median(k):.0f} P(0) {(k == 0).mean():.4f} P(>=3) {(k >= 3).mean():.4f}')
# user overlap / post overlap
print('test rows with post seen in train', te.feature18.isin(tr.feature18.unique()).mean())
