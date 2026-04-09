# ParrotLabs

Deep Learning course @ HSG.

## Dataset

Download the Prudential Kaggle data and place the extracted CSVs at:

```text
data/prudential-life-insurance-assessment/train.csv
data/prudential-life-insurance-assessment/test.csv
```

## Setup

```bash
uv sync
```

## Supported Orchestration

`main.py` is the only supported orchestration entrypoint for the current pipeline.

Supported stages:

- `train`
- `tune`
- `interpret`
- `retrain`
- `select`

## Config Entry Points

Model runs:

- `configs/model/chebykan_experiment.yaml`
- `configs/model/fourierkan_experiment.yaml`
- `configs/model/glm_experiment.yaml`
- `configs/model/xgboost_paper_experiment.yaml`

Tune runs:

- `configs/tune/kan_cheby/kan_cheby_tune.yaml`
- `configs/tune/kan_fourier/kan_fourier_tune.yaml`
- `configs/tune/xgboost_paper/xgboost_paper_tune.yaml`

## Commands

Train:

```bash
uv run python main.py --stage train --config configs/model/chebykan_experiment.yaml
uv run python main.py --stage train --config configs/model/fourierkan_experiment.yaml
uv run python main.py --stage train --config configs/model/xgboost_paper_experiment.yaml
```

Tune:

```bash
uv run python main.py --stage tune --config configs/tune/kan_cheby/kan_cheby_tune.yaml
uv run python main.py --stage tune --config configs/tune/kan_fourier/kan_fourier_tune.yaml
uv run python main.py --stage tune --config configs/tune/xgboost_paper/xgboost_paper_tune.yaml
```

Interpret:

```bash
uv run python main.py --stage interpret --config configs/model/glm_experiment.yaml
uv run python main.py --stage interpret --config configs/model/xgboost_paper_experiment.yaml
uv run python main.py --stage interpret --config configs/model/chebykan_experiment.yaml
```

Retrain selected KAN candidates:

```bash
uv run python main.py --stage retrain --candidate-manifest sweeps/kan-cheby-single-tune_candidates.json --seeds 13 29 47
```

Select final KANs from retraining outputs:

```bash
uv run python main.py --stage select --retrain-manifest artifacts/retrain/chebykan/default-selection/manifest.json
```

## Artifact Layout

Training artifacts:

- `artifacts/<experiment>/run-summary-*.json`
- `checkpoints/<experiment>/model-*.pt`
- `checkpoints/<experiment>/model-*.joblib`
- `checkpoints/<experiment>/model-*.manifest.json`

Namespaced evaluation and interpretability outputs:

- `outputs/eval/<recipe>/<experiment>/`
- `outputs/interpretability/<recipe>/<experiment>/`

Tune artifacts:

- `sweeps/*_best.json`
- `sweeps/*_best.yaml`
- `sweeps/*_candidates.json`

Retrain and selection artifacts:

- `artifacts/retrain/<family>/<selection_name>/manifest.json`
- `artifacts/selection/<family>_selection.json`

## Legacy Scripts

`src/evaluate.py` and `src/submit.py` are legacy entrypoints. They are intentionally not part of the supported workflow anymore. Use `main.py` instead.

## Tests

```bash
uv run python -m pytest tests -v
```
