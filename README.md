# Track danceability prediction (NTU Machine Learning, final project, 2023)

Final project for NTU's Machine Learning course (Spring 2023, nicknamed "HTML"),
done by a team of three. The task was an in-class Kaggle competition: predict
a track's **danceability**, an integer from 0 to 9. Inputs are Spotify audio
features, YouTube engagement counts and categorical metadata. About 15% of the
values in every feature column are missing, and the public and private parts
of the hidden test set were distributed differently. Submissions were scored
by mean absolute error (MAE).

We compared three model families:

- scikit-learn's HistGradientBoostingRegressor
- CatBoost
- a small PyTorch feed-forward network

For each one we varied the imputation and scaling, tuned it with Optuna, and
recommended **CatBoost with iterative imputation**. It scored 1.8356 MAE on the
public leaderboard and 2.1045 on the later-released private labels, the best of
the three on both.

**Read [REPORT.md](REPORT.md)** for the full write-up: data exploration,
categorical feature selection, imputation and scaling experiments, all result
tables, analysis of the public/private shift, and lessons learned.

## Repository contents

| File | What it is |
|---|---|
| `REPORT.md` | Technical write-up of the project (the content of our team report, rewritten). |
| `nn_regressor.py` | Standalone version of the neural-network pipeline from section 4.3 of the report. |
| `requirements.txt` | Python dependencies for `nn_regressor.py`. |

**Not included:**

- **The dataset.** It was provided by the course and is not redistributed here.
- **The original Colab notebooks.** They were exploratory, built on a
  course-provided template, and full of dead code.
- **Code for the gradient-boosting models.** HGBR, CatBoost, LightGBM, XGBoost
  and their Optuna studies were run in separate notebooks that are not part
  of this repository. Their configurations and results are documented in
  REPORT.md.

## Running the neural-network pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put the competition files here (not included):
#   data/train.csv  - must contain id, Danceability, the 13 numeric feature columns, Composer
#   data/test.csv   - same columns without Danceability
python nn_regressor.py --data-dir data --out pred.csv
```

The defaults match the final network configuration in the report:

- Adam optimizer
- one hidden layer with 100 units, dropout 0.3
- min-max scaling and mean imputation
- `CosineAnnealingWarmRestarts(T_0=2, T_mult=2)` learning-rate schedule
- early stopping after 100 epochs without improvement

Each experiment axis from the report has its own flag:

```bash
python nn_regressor.py --optimizer rmsprop --scaler standard   # Table 5
python nn_regressor.py --impute knn                            # Table 6 (mean | median | mean-median | knn | iterative)
python nn_regressor.py --hidden 128 64                         # deeper variants
```

The script splits `train.csv` 80/20 with a fixed seed. It fits the imputer,
scaler and Composer target encoding on the training split only, and reports
the validation MAE of rounded, clipped predictions. It writes a submission CSV
with the columns `id,Danceability`.

**Reference run.** These numbers come from the default settings on CPU with
Python 3.13, PyTorch 2.15 and scikit-learn 1.9:

```
train (13736, 14)  valid (3434, 14)  test (6315, 14)  device cpu
best validation MAE 1.7068 at epoch 222 (stopped at 322)
```

This validation MAE comes from a random split of the training data. It is not
comparable to the leaderboard numbers in REPORT.md, because the test set was
distributed differently from the training data.

### Differences from the original notebooks

`nn_regressor.py` was rewritten after the course from our Colab notebooks. It
keeps the method and fixes the following issues (see REPORT.md, section 7):

- The Composer target encoding is fitted on the training split only, so it no
  longer leaks validation labels. It is added as a new feature instead of
  overwriting `Key`.
- The imputer and scaler are fitted on the training split and then applied
  to the validation and test data. Previously the test set was scaled using
  its own statistics.
- Predictions are rounded and clipped to 0-9 instead of truncated.
- Early stopping uses validation MAE, the competition metric, instead of the
  smooth-L1 validation loss.

## Notes

- This was a team project. The experiments and the original report were a
  joint effort.
- All model scores in REPORT.md come from our final report and were not
  re-run. The dataset statistics (row counts, missing-value rates, skewness)
  were recomputed from the course CSVs.
