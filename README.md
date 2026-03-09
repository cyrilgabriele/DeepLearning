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

## Training Models

```bash
# Train individual models
uv run python src/train.py model=chebykan
uv run python src/train.py model=fourierkan
uv run python src/train.py model=bsplinekan
uv run python src/train.py model=mlp
uv run python src/train.py model=xgb

# Compare all neural models (Hydra multirun)
uv run python src/train.py model=chebykan,fourierkan,bsplinekan,mlp -m

# Override hyperparameters from CLI
uv run python src/train.py model=chebykan model.degree=5 model.widths=[128,64] train.max_epochs=50
uv run python src/train.py model=fourierkan model.grid_size=8 model.lr=5e-4
uv run python src/train.py model=bsplinekan model.grid_size=10 model.spline_order=4

# Lightweight test run (5 epochs)
uv run python src/train.py model=chebykan train.max_epochs=5
```

## Evaluate a Saved Checkpoint

```bash
uv run python src/evaluate.py --checkpoint path/to/model.ckpt --data_path data/prudential-life-insurance-assessment/train.csv --model_type chebykan
```

## Run Tests

```bash
uv run python -m pytest tests/ -v
```
