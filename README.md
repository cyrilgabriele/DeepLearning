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

### Via Hydra (TabKAN models)

```bash
# Train individual models
uv run python src/train.py model=chebykan
uv run python src/train.py model=fourierkan
uv run python src/train.py model=bsplinekan
uv run python src/train.py model=mlp
uv run python src/train.py model=xgb

# Train all models and print comparison table
uv run python src/train.py model=chebykan,fourierkan,bsplinekan,mlp,xgb train.max_epochs=5 -m

# Override hyperparameters from CLI
uv run python src/train.py model=chebykan model.degree=5 model.widths=[128,64] train.max_epochs=50
uv run python src/train.py model=fourierkan model.grid_size=8 model.lr=5e-4
uv run python src/train.py model=bsplinekan model.grid_size=10 model.spline_order=4

# Lightweight test run (5 epochs)
uv run python src/train.py model=chebykan train.max_epochs=5
```

### Via YAML config (Trainer pipeline)
- `main.py` accepts a required YAML config that captures **all** trainer, preprocessing, and model parameters.
- Start from `configs/smoke_experiment.yaml`, adjust every field (especially `trainer.train_csv`/`trainer.test_csv`) to match your environment, and launch via `python main.py --config configs/smoke_experiment.yaml`.
- A fixed global random seed (`42`) is applied automatically to numpy/scikit-learn (and PyTorch if installed), so every run is deterministic.
- The CLI instantiates `Trainer`, which fits the chosen preprocessor on the training subset, builds the requested model, and prints MAE/accuracy/F1 on the evaluation slice.

## Evaluate a Saved Checkpoint

```bash
uv run python src/evaluate.py --checkpoint path/to/model.ckpt --data_path data/prudential-life-insurance-assessment/train.csv --model_type chebykan
```

## Run Tests

```bash
uv run python -m pytest tests/ -v
```
