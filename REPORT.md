# Predicting track danceability: gradient-boosted trees vs. a small neural network

Technical write-up of a team final project for NTU's Machine Learning course
(Spring 2023). We were a team of three, and each of us owned one model family:
**I built the feed-forward neural network (section 4.3)**, while my teammates
built the HistGradientBoosting and CatBoost models. The data analysis, validation
protocol and final comparison were shared. This document covers the same ground
as our submitted report: the data, the three model families we compared, how
we validated them, the numbers we got, and what we would do differently.

All scores below are **mean absolute error (MAE)**; lower is better.

---

## 1. The task

Each example is a music track described by Spotify audio features, YouTube
engagement statistics and some metadata. The label, `Danceability`, is an
integer from 0 to 9, so the problem is ordinal regression. Predictions were
scored by MAE on a hidden test set through an in-class Kaggle competition.

The competition ran in two stages:

- **Stage 1.** Train on the labelled training set and submit predictions for the
  test set. A public leaderboard showed the score on part of the test set.
- **Stage 2.** Labels were released for part of the stage-1 private test set.
  Below, "partial private" means a model's score on those released labels.
  These models were trained without the released labels. The score turned out
  to be the most honest signal we had, because the public and private parts of
  the test set were not distributed the same way.

Dataset size (from the course-provided CSVs):

| | rows | columns |
|---|---|---|
| training set | 17,170 | 29 (28 inputs + label) |
| stage-1 test set | 6,315 | 28 |

The ten label values are close to balanced in the training set, with between
1,639 and 1,860 rows each.

The input columns fall into three groups:

- **13 numeric features:** Energy, Key, Loudness, Speechiness, Acousticness,
  Instrumentalness, Liveness, Valence, Tempo, Duration_ms, Views, Likes, Stream.
- **Low-cardinality categorical features:** Composer, Artist, Album_type,
  Licensed, official_video. Key is stored as a number but has only 11 distinct
  values, so we also tried treating it as a category.
- **Identifiers, free text and the rest:** Track, Album, Uri, Spotify/YouTube
  URLs, Description, Title, Channel, Comments. We did not use these (but see
  section 7 about Comments).

---

## 2. Data exploration

### 2.1 Missing values

Every feature has missing values in both the training and the test set, at a
rate of about 14-16% per column. The gaps are scattered across rows rather than
concentrated in a few, so dropping incomplete rows was not an option. Our
report put the loss at 93.8% of the training data. When I re-checked later, 87.9%
of training rows had at least one gap in the 13 numeric columns alone.

Imputation strategies we compared:

| Strategy | Notes |
|---|---|
| Mean / median | Per-column fill. For the neural network we also tried the average of the two. |
| KNN imputer | Fill from the 5 nearest neighbours (scikit-learn `KNNImputer`). |
| Iterative imputer | Models each incomplete feature as a function of the others, round-robin (scikit-learn `IterativeImputer`). |
| None | HistGradientBoosting and CatBoost both handle `NaN` natively. |

### 2.2 Feature scaling

Many numeric features are far from normally distributed. Duration, views,
likes and streams are strongly right-skewed: their sample skewness on the
training set ranges from about 4 to 23. We compared three scalers:

1. Min-max scaling to [0, 1].
2. Standardization (zero mean, unit variance).
3. Robust / quantile scaling by the interquartile range (25th to 75th percentile).

Scaling made no visible difference for the tree models, as expected, because
trees split on thresholds and are not affected by monotonic rescaling. It
mattered for the neural network (section 4.3). Scaling the data *before* KNN
or iterative imputation made the results worse for every model, so we always
imputed on unscaled features.

### 2.3 Which categorical features help?

We ran this experiment with CatBoost because it handles categorical features
natively. Setup: default parameters, all 13 numeric features, no scaling or
imputation, and categorical features added in groups. Scores are on a 20%
validation split of the training data.

**Table 1 - Categorical feature selection (CatBoost, default parameters)**

| Categorical features added | 20% validation MAE |
|---|---|
| none | 2.109809 |
| Composer | 1.990503 |
| Composer, Artist, Album_type, Key | 1.962641 |
| Composer, Artist, Album_type, Key, Licensed, official_video | 1.962462 |

Composer alone cuts the error by 5.7%. Artist, Album_type and Key add a little
more (7.0% in total). Licensed and official_video change the MAE by less than
0.0002. We therefore used **Composer, Artist, Album_type and Key** for both
tree models. The neural network needs numeric inputs, so it got only a
target-encoded Composer feature, which was the single most useful one.

---

## 3. Validation protocol

- **Hyperparameter tuning.** We used [Optuna](https://optuna.org/) on an 80/20
  train/validation split of the training data. It was more efficient than
  grid or random search.
- **Model selection.** In stage 1 we chose models based on the public
  leaderboard and the same 80/20 validation split. This was a mistake (see
  section 7). After stage 2 we re-ran the key comparisons against the partial
  private labels.

The result tables use these columns:

| Column | Meaning |
|---|---|
| 20% val | Held-out 20% of the training set (in-distribution) |
| Public | Stage-1 public leaderboard |
| Partial private | Labels released in stage 2 (see section 1) |

---

## 4. Models

### 4.1 HistGradientBoostingRegressor (HGBR)

scikit-learn's histogram-based gradient boosting. It bins each feature into a
small number of histogram buckets and stores per-bin statistics (such as the
sum of gradients). Split finding then scans the bins instead of sorting the raw
values, which makes training much faster on tens of thousands of rows with
little or no loss in accuracy.

Scaling had no effect on HGBR, so the comparison came down to the imputation
method. Hyperparameters were tuned with Optuna and no scaling was applied.

**Table 2 - HGBR, imputation method (tuned parameters, no scaling)**

| Imputation | Public | Partial private |
|---|---|---|
| none (native NaN) | 1.86759 | 2.2139 |
| mean | 1.93886 | 2.1822 |
| median | 1.94995 | 2.2456 |
| iterative | 1.88501 | 2.1822 |
| **KNN** | 1.88755 | **2.1616** |

Letting HGBR handle missing values itself gave the best public score but one
of the worst partial-private scores. KNN imputation generalized best.

### 4.2 CatBoostRegressor

CatBoost was designed for data with categorical features. It addresses two
problems that other boosting libraries have:

- **Ordered target statistics.** A common way to encode a categorical value is
  to replace it with the mean label of that category. Computed naively on the
  full training set (greedy target statistics), this leaks the label into the
  feature. Holdout and leave-one-out variants either waste data or still leak.
  CatBoost draws a random permutation of the training set as an artificial
  "time". Each example is then encoded using only the examples before it.
- **Ordered boosting.** The same permutation idea is applied to the
  gradient-boosting procedure itself. This reduces the prediction shift that
  appears when residuals are computed with models that have already seen the
  example.

Given these properties we expected CatBoost to make the best use of Composer
and Artist.

**Other boosting libraries.** In stage 1 we also tried XGBoost and LightGBM.
All three scored about the same on the public leaderboard. We dropped XGBoost
because it trained much more slowly and was harder to tune. LightGBM was
slightly ahead of CatBoost on the public set, but it lost more ground once the
private labels arrived:

**Table 3 - CatBoost vs. LightGBM (no imputation, tuned parameters)**

| Model | Public | Partial private |
|---|---|---|
| CatBoost | 1.84098 | **2.103011** |
| LightGBM | **1.82166** | 2.16164 |

**Imputation.** In stage 1 we were happy with the public score, so we let
CatBoost handle `NaN` on its own. After the data shift we ran the imputation
comparison we should have run earlier. We used default parameters and scored
each method on both the 20% validation split and the partial private labels:

**Table 4 - CatBoost, imputation method (default parameters)**

| Imputation | 20% val | Partial private |
|---|---|---|
| none (native NaN) | 1.962640 | 2.1489 |
| mean | 1.969952 | 2.12678 |
| median | 1.996107 | 2.17591 |
| **iterative** | **1.945246** | 2.1109 |
| KNN | 1.988196 | **2.0919** |

The two evaluation sets disagree. KNN is best on partial private but
second-worst on the in-distribution split. Iterative imputation is best on the
split and second-best on partial private. Mean imputation is also reasonable on
both. We chose **iterative imputation** for the final CatBoost model because it
was consistently good on both sets rather than best on one.

### 4.3 Feed-forward neural network

*This is the part I built.*

This model is called "Linear Neural Network (LN)" in our original report. It
is a small multilayer perceptron in PyTorch:

```python
self.layers = nn.Sequential(
    nn.Linear(input_dim, hidden),   # hidden = 100
    nn.ReLU(),
    nn.Dropout(p=dropout),          # dropout = 0.3
    nn.Linear(hidden, 1),
)
```

- **Inputs:** the 13 numeric features plus a target-encoded Composer feature.
- **Depth:** We also tried two and three hidden layers. One hidden layer was best,
  so we kept it.
- **Learning-rate schedule:** `CosineAnnealingWarmRestarts(T_0=2, T_mult=2,
  eta_min=1e-5)`, stepped per batch. The cosine decay lets training settle
  into a minimum. The periodic restarts briefly raise the learning rate again,
  which can help it escape a poor one.
- **Stopping:** at most 6,000 epochs, with early stopping after 100 epochs
  without improvement on the validation split. In practice runs converged
  after roughly 150-500 epochs.
- **Training details from the notebooks:** smooth-L1 loss, Adam with weight
  decay 1e-3, batch size 256. The learning rate, depth, width and dropout were
  searched with Optuna.

We first compared the optimizer and the scaler. Settings for this comparison:
one hidden layer with 100 units, dropout 0.3, mean imputation.

**Table 5 - Optimizer and scaling**

| Optimizer | Scaling | Public | Partial private |
|---|---|---|---|
| **Adam** | **min-max** | **1.85239** | 2.25269 |
| Adam | standardize | 1.87076 | 2.32013 |
| Adam | quantile (IQR) | 1.92334 | 2.29734 |
| RMSprop | min-max | 1.98891 | 2.37555 |
| RMSprop | standardize | 2.04719 | **2.24762** |
| RMSprop | quantile (IQR) | 1.96800 | 2.31895 |

Adam beat RMSprop on the public set for every scaler, and min-max scaling was
the best choice with Adam on both columns. The partial private column is
noisier. RMSprop with standardization has the worst public score but a
slightly better private score than Adam + min-max.

Next we compared imputation methods using Adam and min-max scaling:

**Table 6 - Imputation method (Adam, min-max)**

| Imputation | Public | Partial private |
|---|---|---|
| **mean** | 1.85239 | **2.25269** |
| median | 1.85144 | 2.25902 |
| average of mean and median | **1.84542** | 2.26409 |
| iterative | 1.92049 | 2.34325 |
| KNN | 1.88026 | 2.2929 |

Mean, median and their average are within about 0.01 of each other. The
model-based imputers, which worked well for the tree models, clearly hurt the
network.

---

## 5. Final configurations and results

**Final configuration for each model family**

| Model | Hyperparameters | Preprocessing |
|---|---|---|
| HGBR | `learning_rate=0.1423723421700089, max_depth=27, min_samples_leaf=1, l2_regularization=0.5521495769875121, max_iter=301` | no scaling, KNN imputation |
| CatBoost | `iterations=1131, depth=10, colsample_bylevel=0.09984973243776221, reg_lambda=4.132212730052845, bagging_temperature=9.316758579281807, eval_metric='MAE'` | no scaling, iterative imputation |
| Neural network | Adam, 1 hidden layer, 100 units, dropout 0.3 | min-max scaling, mean imputation |

**Table 7 - Final models (tuned with Optuna on the 80/20 split)**

| Model | Public | Partial private |
|---|---|---|
| HGBR | 1.88755 | 2.1616 |
| **CatBoost** | **1.8356** | **2.1045** |
| Neural network | 1.85239 | 2.25269 |

**Recommendation: CatBoost.** It had the best score on both the public
leaderboard and the partial private labels. For the stage-2 submission we kept
the configuration above and retrained it on the merged data: the original
training set plus the newly labelled part of the test set. Looking back after
the competition ended, this was the right choice.

---

## 6. Analysis

**Every model got worse on the private data, and the network got worst.** From
public to partial private, the MAE increased by:

| Model | Increase in MAE |
|---|---|
| CatBoost | 0.269 |
| HGBR | 0.274 |
| LightGBM | 0.340 |
| Neural network | 0.400 |

The network was last of these four on the private labels.

**Public rankings were often reversed on the private labels:**

- LightGBM beat CatBoost on public but lost on private (Table 3).
- HGBR without imputation was best on public but fourth of five on private
  (Table 2).
- RMSprop + standardize was worst on public but best on private (Table 5).

Because these reversals were common, the public leaderboard alone was a poor
basis for choosing a model.

**Imputation is a modelling decision, not a cleanup step.** For both tree
models, the choice of imputer moved the partial-private MAE by about 0.08
(HGBR: 2.1616-2.2456; CatBoost: 2.0919-2.17591). The best imputer also
depended on the model. Compared with native `NaN` handling, KNN and iterative
imputation helped both tree models on the private labels. For the network,
both did worse than a simple mean fill.

**Categorical metadata carried a lot of signal.** Adding a single column,
Composer, cut CatBoost's validation MAE by 5.7%. This is one reason CatBoost,
with its built-in categorical handling, ended up ahead.

---

## 7. What went wrong and what I would do differently

1. **We had no private hold-out in stage 1.** We split the data 80/20 for
   tuning and then chose models by the public leaderboard, which is effectively
   tuning on the test set. We should have set aside about 10% of the training
   data as our own test set and used it only for final model selection.
2. **We postponed the imputation study.** In stage 1 we let the boosters handle
   `NaN` because the public score looked fine. The comparison in Table 4 was
   cheap and should have come first.
3. **We used single-split estimates.** Many of the differences in Tables 2-6
   are a few thousandths of MAE, measured on one split. In hindsight I would
   use k-fold cross-validation and report the spread before trusting gaps
   that small.
4. **The neural-network notebooks had problems that I only found while
   cleaning them up for this repository:**
   - The Composer target encoding was computed on all training rows,
     including the validation split, which leaks labels into the
     validation score.
   - The encoded value overwrote the `Key` column instead of being added as
     a new feature.
   - In at least one notebook version, the test set was min-max scaled using
     its own minimum but the training set's range.
   - Predictions were truncated (`int()`) rather than rounded before
     submission.

   These issues may account for part of the network's weaker scores.
   `nn_regressor.py` in this repository fixes all four.
5. **We left a usable column out.** `Comments` is a numeric count, but we typed
   it as text and never used it. On the training set its Spearman correlation
   with the label is 0.12, which is small but non-zero and worth testing.

---

## References

- "Optuna: A next-generation hyperparameter optimization framework." KDD 2019.
- "CatBoost: unbiased boosting with categorical features." NeurIPS 2018.
- scikit-learn documentation:
  [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html),
  [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html),
  [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html).
- PyTorch documentation:
  [CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html).
- [CatBoost](https://github.com/catboost/catboost) on GitHub.
