# ParrotLabs
Deep Learning course @ HSG

## Dataset
Download the competition data from https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data and place the extracted files inside a `data` directory at the repository root.

## Preprocessing Methodology
- Kaggle already separates `train.csv` (with `Response`) and `test.csv` (without labels). Use `split_prudential_training_df` or `load_and_prepare_prudential_training_data` to optionally carve out a *small* evaluation subset from `train.csv` only; the `eval_size` fraction is therefore relative to the Kaggle training file.
- Instantiate either `PrudentialPaperPreprocessor` (paper baseline) or `PrudentialKANPreprocessor` (enhanced pipeline) and pass it into the splitter. The helper fits on the remaining training rows and reuses the learned state for the evaluation subset.
- `PrudentialKANPreprocessor` performs stratified k-fold (default 5-fold) target encoding, minimizing categorical leakage while preserving class balance.
- Treat the returned `PrudentialDataSplits` object as the canonical source for both raw and processed splits when training/evaluating models; never refit on evaluation data if you intend to publish the numbers.

## Running Experiments
- `main.py` now accepts a single argument: a required YAML config that captures **all** trainer, preprocessing, and model parameters. This keeps results reproducible by eliminating implicit defaults or ad-hoc overrides.
- Start from `configs/smoke_experiment.yaml`, adjust every field (especially `trainer.train_csv`/`trainer.test_csv`) to match your environment, and launch via `python main.py --config configs/smoke_experiment.yaml`.
- A fixed global random seed (`42`) is applied automatically to numpy/scikit-learn (and PyTorch if installed), so every run is deterministic; the only runtime-dependent variable is the detected compute device (CUDA, MPS, or CPU).
- Set `trainer.eval_size` to a float within `[0.0, 0.2]`: choose `0.0` to skip splitting or a small positive fraction to reserve a validation slice. `null` is not accepted so the config remains the single source of truth.
- The CLI instantiates `Trainer`, which fits the chosen preprocessor on the training subset, builds the requested model, and prints MAE/accuracy/F1 on the evaluation slice.
- When `trainer.test_csv` is set, the trainer also transforms Kaggle's `test.csv`, runs inference, and saves `test-predictions-<timestamp>.csv` inside the experiment's `artifacts/<name>` folder for quick submission packaging.
