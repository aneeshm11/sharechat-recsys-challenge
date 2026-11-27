# ShareChat RecSys Challenge - team AM

Code, trained models and write-up for our two selected submissions.

| | Kaggle notebook | Public score | `request_id` or request-derived features | Reads other rows |
|---|---|---|---|---|
| **track_b** | [`sharechat-track-b`](https://www.kaggle.com/code/cassinicraft/sharechat-track-b) v3 | 0.89382 | none | yes: same post, and same user at another time |
| **track_a** | [`sharechat-track-a`](https://www.kaggle.com/code/cassinicraft/sharechat-track-a) v3 | 0.83441 | none | no |

Neither track uses `request_id`, request size, or any grouping of rows by request, in training or in inference.
track_b is **not** row-independent; track_a is. Section 1 of the write-up says exactly what that means.

## Where to find what you asked for

| Your request | Where it is |
|---|---|
| **1. Detailed write-up** (approach, features, model, training, validation) | [`WRITEUP.md`](WRITEUP.md) |
| Every feature used to predict the evaluation set | [`FEATURES.md`](FEATURES.md) and the complete lists in [`features/`](features) (one line per feature, in model order) |
| **2. Differences between training and evaluation features** | [`WRITEUP.md`, section 6](WRITEUP.md#6-features-in-training-versus-features-at-prediction-time) |
| **3. Full code for the final submissions** | track_b: [`track_b/`](track_b) - start with [`track_b/TRAINING.md`](track_b/TRAINING.md). track_a: [`track_a/`](track_a) |
| The notebooks that produced the submitted files | [`track_b/sharechat-track-b.ipynb`](track_b/sharechat-track-b.ipynb), [`track_a/sharechat-track-a.ipynb`](track_a/sharechat-track-a.ipynb) |
| **4. Trained model artefacts** | [`models/`](models), indexed in [`ARTIFACTS.md`](ARTIFACTS.md) |
| Check that request-ID features were not used | [`WRITEUP.md`, section 1](WRITEUP.md#1-what-we-submitted) and the tests in [`docs/ORGANISER_NOTES.md`](docs/ORGANISER_NOTES.md), section 5 |

## Repository layout

```
WRITEUP.md            the write-up
FEATURES.md           feature blocks of both tracks
ARTIFACTS.md          model files, how each was trained, checksums, Kaggle locations
track_a/              row-independent track: notebook, feature code, re-training script
track_b/              cross-row track: notebook, pipeline (readable sources + the single file that was run), training commands
models/               29 files: 8 + 11 boosters, configurations, manifest, vocabularies
features/             complete feature lists (751 / 1,697 / 1,592)
diagnostics/          measurements quoted in the write-up and in the organiser notes, with their result files
docs/ORGANISER_NOTES.md   what we measured about request size and the shape of the leaderboard
logs/                 logs of every training and experiment run
```

## Reproducing a submission

Both notebooks run on Kaggle without internet and without a GPU, from the competition files plus the files in `models/`:

* track_a: about 4 minutes. Output is bit-identical to the submitted file.
* track_b: about 31 minutes; it rebuilds all features from the two raw files. Output matches the submitted file with rank correlation 1.0000000.

Training from scratch is described in `track_b/TRAINING.md` and in section 7 of the write-up, which also states one gap: the original
training script of the track_a boosters was not kept, and what we provide in its place.

## Disclosure

Our team's highest public scores (0.91408, 0.92029, 0.92316) came from earlier files that use request size. They are disclosed in both
notebooks and were not selected. The scripts in `diagnostics/` read `request_id` to measure what it is worth; nothing in `track_a/` or
`track_b/` does.
