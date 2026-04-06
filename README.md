# ParrotLabs
Deep Learning course @ HSG

## Dataset
Download the competition data from https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data and place the extracted files inside a `data` directory at the repository root:

```
data/prudential-life-insurance-assessment/train.csv
data/prudential-life-insurance-assessment/test.csv
```

## Setup

```bash
uv sync
```

## Preprocessing Methodology
- Kaggle already separates `train.csv` (with `Response`) and `test.csv` (without labels). Use `split_prudential_training_df` or `load_and_prepare_prudential_training_data` to optionally carve out a *small* evaluation subset from `train.csv` only; the `eval_size` fraction is therefore relative to the Kaggle training file.
- Instantiate either `PrudentialPaperPreprocessor` (paper baseline) or `PrudentialKANPreprocessor` (enhanced pipeline) and pass it into the splitter. The helper fits on the remaining training rows and reuses the learned state for the evaluation subset.
- `PrudentialKANPreprocessor` performs stratified k-fold (default 5-fold) target encoding, minimizing categorical leakage while preserving class balance.
- Treat the returned `PrudentialDataSplits` object as the canonical source for both raw and processed splits when training/evaluating models; never refit on evaluation data if you intend to publish the numbers.

## Running Experiments

- `main.py` is the single entrypoint for both training and tuning.
- Every run is driven from one YAML experiment config that contains trainer, preprocessing, and model settings.
- Start from `configs/smoke_experiment.yaml`, `configs/experiments/kan_cheby_single.yaml`, or `configs/experiments/xgboost_paper_experiment.yaml` and adjust `trainer.train_csv` / `trainer.test_csv` for your machine.

### Train

```bash
uv run python main.py --stage train --config configs/smoke_experiment.yaml
uv run python main.py --stage train --config configs/experiments/kan_cheby_single.yaml
uv run python main.py --stage train --config configs/experiments/xgboost_paper_experiment.yaml
```

### Tune

```bash
uv run python main.py --stage tune --config configs/smoke_experiment.yaml --n-trials 20
uv run python main.py --stage tune --config configs/experiments/xgboost_paper_experiment.yaml --n-trials 25 --timeout-tune 3600
```

- Tune runs use the same `Trainer` pipeline as train runs.
- The tuned config is written to `sweeps/*_best.yaml`; train it with `python main.py --stage train --config <that-file>.yaml`.
- For the paper XGBoost model, tune stage searches parameters externally with Optuna and writes the winning fixed params back into the generated YAML.
- A fixed global random seed from the config is applied automatically to numpy/scikit-learn and PyTorch when available.

## Evaluate a Saved Checkpoint

```bash
uv run python src/evaluate.py --checkpoint path/to/model.ckpt --data_path data/prudential-life-insurance-assessment/train.csv --model_type chebykan
```

## Run Tests

```bash
uv run python -m pytest tests/ -v
```
