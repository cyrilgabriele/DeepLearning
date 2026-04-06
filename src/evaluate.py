"""Standalone evaluation: load a checkpoint, optimize thresholds, report QWK."""
import argparse
from pathlib import Path

import numpy as np
import torch

from configs import set_global_seed
from src.data.dataset import PrudentialDataModule
from src.models.tabkan import TabKAN
from src.models.mlp import MLPBaseline
from src.metrics.qwk import optimize_thresholds


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--model_type", type=str, choices=["chebykan", "fourierkan", "mlp"],
                        required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = set_global_seed(args.seed)

    dm = PrudentialDataModule(data_path=args.data_path, batch_size=args.batch_size, seed=seed)
    dm.setup()

    if args.model_type == "mlp":
        model = MLPBaseline.load_from_checkpoint(args.checkpoint)
    else:
        model = TabKAN.load_from_checkpoint(args.checkpoint)
    model.eval()

    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            val_preds.append(model(x).cpu().numpy())
            val_targets.append(y.cpu().numpy())

    preds = np.concatenate(val_preds).flatten()
    targets = np.concatenate(val_targets).flatten()

    thresholds, qwk = optimize_thresholds(targets, preds)
    print(f"Optimized QWK: {qwk:.4f}")
    print(f"Thresholds: {thresholds}")


if __name__ == "__main__":
    main()
