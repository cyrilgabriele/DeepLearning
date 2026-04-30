# Interpretability Stage Runtime Configs

These YAML files persist the runtime controls passed to `main.py --stage interpret`.
They are separate from experiment configs because they do not define training or
tuning. They define how an already-trained checkpoint is interpreted.

Fields:

- `config`: experiment YAML used to resolve preprocessing/model metadata.
- `checkpoint`: trained checkpoint to interpret.
- `output_root`: root directory for eval and interpretability outputs.
- `pruning_threshold`: KAN pruning threshold.
- `qwk_tolerance`: maximum allowed QWK drop during pruning.
- `candidate_library`: symbolic fitting backend for KANs.
- `max_features`: number of top features to keep in reporting/plots.

Run from the project root:

```bash
uv run python main.py --stage interpret --interpret-config configs/interpretability_stage/stage_c_best/chebykan.yaml
uv run python main.py --stage interpret --interpret-config configs/interpretability_stage/stage_c_best/fourierkan.yaml
uv run python main.py --stage interpret --interpret-config configs/interpretability_stage/stage_c_best/xgboost.yaml
```
