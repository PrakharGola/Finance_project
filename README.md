# NIFTY Options IV Surface Reconstruction

This project imputes missing implied-volatility values in a NIFTY options surface and writes a Kaggle-style submission.

## Run

Place `dataset.csv` in this folder, then run the direct model-selection pipeline:

```bash
python src/model_selection_iv_solution.py --input dataset.csv --output submission.csv
```

Defaults are supported:

```bash
python src/model_selection_iv_solution.py
```

Outputs:

- `filled_dataset.csv`: original wide dataset with all option IV columns filled.
- `submission.csv`: rows for originally missing cells only, with `id,value`.

The submission id format is:

```text
datetime||column_name
```

Example:

```text
07-01-2026 09:15||NIFTY27JAN2624100PE
```

## What The Pipeline Does

`model_selection_iv_solution.py` parses each option ticker into expiry, strike, and option type, then builds validation masks from known observed IV cells. It hides those known values, predicts them, compares predictions with the actual known values, and reports MSE.

It also simulates a Kaggle-style public/private split inside each validation scenario:

- random observed cells
- contiguous time block
- strike block
- sampled expiry-day cells

The final model is selected by private-style validation performance before it is used to fill the real missing cells.

The feature set includes:

- strike, CE/PE flag, moneyness, distance from spot, absolute distance, log-moneyness
- hour, minute, minute of day, day index, minutes/days to expiry, expiry-day flag
- same-row surface statistics by option type
- nearest lower/upper strike IVs and ATM distance rank

It compares several candidate imputers:

- cross-sectional strike interpolation with polynomial/spline smoothing when available
- past-only time-series estimates using forward fill, EWM, and rolling median
- row/type median surface fallback
- optional ML surface model using LightGBM, XGBoost, scikit-learn, or a deterministic NumPy online ridge fallback
- adaptive hybrid that uses the same-row surface when enough strikes are observed, and past-only time behavior when the row/side is sparse

The script prints public, private, and full validation MSEs for every candidate and chooses the robust private-style winner.

## Look-Ahead Bias Controls

The script does not read any solution file and trains only on observed non-missing cells from `dataset.csv`.

Time movement is past-only:

- no backward fill
- no two-sided time interpolation
- forward fill, EWM, and rolling median use only current/past rows
- ML predictions are generated with expanding chronological training data, so a row only sees earlier rows plus observed same-timestamp option-chain values

Same-timestamp strike interpolation is allowed because the option-chain snapshot at that timestamp is assumed available.

# Model Selection For NIFTY IV Reconstruction

Kaggle's public leaderboard uses only around 30% of hidden test targets, while the final/private ranking uses the remaining 70%. A model can look good on the public 30% and still generalize poorly to the private 70%.

This project therefore simulates public/private validation. For each validation experiment, known non-missing IV cells are hidden, then split into pseudo-public 30% and pseudo-private 70%. Models are selected by a robust score rather than by one public-style MSE.

## Validation Strategies

- `random_cell_mask`
- `missing_pattern_like_mask`
- `public_private_like_mask`
- `strike_block_mask`
- `time_block_mask`
- `expiry_day_mask`
- `grouped_day_mask`

The validation runs across seeds: `7, 11, 21, 42, 77, 101, 2026`.

## Robust Selection

The selected model minimizes:

```text
private_70_mean_mse
+ 0.25 * public_private_gap_mean
+ 0.20 * mse_std
+ 0.20 * expiry_day_mse
+ 0.10 * worst_case_mse
```

CE/PE balance is reported separately and used as a tie-breaker after the robust score.

## Run

```bash
python model_selection_iv_solution.py --input dataset.csv --output submission.csv
```

Outputs:

- `validation_leaderboard.csv`
- `best_model_config.json`
- `filled_dataset.csv`
- `submission.csv`

`submission.csv` contains only originally missing cells and uses:

```text
id,value
datetime||contract,value
```
