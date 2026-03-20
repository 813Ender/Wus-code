
import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import torch

from utils.helpers import set_seed, collate_fn
from dataloader import MultiTaskDataset, build_loaders
from model import MultiTaskModel
from train import train_multitask


def main():
    parser = argparse.ArgumentParser(
        description="5-Fold Cross Validation for MultiTaskModel"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="../data/lesion-selected",
        help="Base path for dataset",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=90, help="Number of epochs per fold"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--time_loss_weight",
        type=float,
        default=0.5,
        help="Weight for time loss",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k lesion-positive slices per case for time aggregation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for fold logs and summary",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print("Loading all data from train/val/test...")
    all_samples = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(args.base_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping.")
            continue
        dataset = MultiTaskDataset(split_path)
        all_samples.extend(dataset.samples)

    if not all_samples:
        raise ValueError("No samples loaded from any split.")


    case_to_samples = defaultdict(list)
    case_to_time = {}
    for sample in all_samples:
        relative_path = sample[4]
        case = os.path.dirname(os.path.dirname(relative_path))
        case_to_samples[case].append(sample)
        case_to_time[case] = sample[3]

    all_cases = list(case_to_samples.keys())
    case_labels = np.array([case_to_time[c] for c in all_cases])

    skf = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )

    fold_metrics = []
    os.makedirs(args.output_dir, exist_ok=True)
    last_log_filename = None

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(all_cases, case_labels)
    ):
        print(f"\n=== Fold {fold + 1}/{args.n_splits} ===")
        fold_dir = os.path.join(args.output_dir, f"fold_{fold + 1}")

        train_cases = [all_cases[i] for i in train_idx]
        val_cases = [all_cases[i] for i in val_idx]

        train_samples = [s for c in train_cases for s in case_to_samples[c]]
        val_samples = [s for c in val_cases for s in case_to_samples[c]]

        train_loader, val_loader = build_loaders(
            train_samples,
            val_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = MultiTaskModel(dropout_rate=args.dropout_rate).to(device)

        log_filename, final_model_path, best_model_path, best_val_metrics = (
            train_multitask(
                model,
                train_loader,
                val_loader,
                args.epochs,
                args.lr,
                args.dropout_rate,
                args.weight_decay,
                time_loss_weight=args.time_loss_weight,
                log_dir=fold_dir,
                top_k=args.top_k,
            )
        )
        last_log_filename = log_filename

        if best_val_metrics is not None:
            fold_metrics.append(best_val_metrics)
            print(
                f"\nFold {fold+1} Best Val Results (case-level TIME, top_k={args.top_k}):"
            )
            print(f"Time AUC (case-level): {best_val_metrics['time_auc']:.4f}")
            print(f"Time Acc (case-level): {best_val_metrics['time_acc']:.4f}")
            print(f"Time F1  (case-level): {best_val_metrics['time_f1']:.4f}")
            print(
                f"Time Sens/Spec: {best_val_metrics['time_sensitivity']:.4f}/{best_val_metrics['time_specificity']:.4f}"
            )
            print(
                f"Time PPV/NPV: {best_val_metrics['time_ppv']:.4f}/{best_val_metrics['time_npv']:.4f}"
            )


    if fold_metrics:
        metrics_array = {
            key: np.array([m[key] for m in fold_metrics])
            for key in fold_metrics[0]
        }
        mean_metrics = {key: np.mean(vals) for key, vals in metrics_array.items()}
        std_metrics = {key: np.std(vals) for key, vals in metrics_array.items()}

        summary_text = (
            "\n=== 5-Fold Cross Validation Summary (mean ± std, case-level TIME) ===\n"
        )
        summary_text += f"Time - Accuracy: {mean_metrics['time_acc']:.4f} ± {std_metrics['time_acc']:.4f}\n"
        summary_text += f"Time - AUC: {mean_metrics['time_auc']:.4f} ± {std_metrics['time_auc']:.4f}\n"
        summary_text += f"Time - F1: {mean_metrics['time_f1']:.4f} ± {std_metrics['time_f1']:.4f}\n"
        summary_text += f"Time - Sensitivity: {mean_metrics['time_sensitivity']:.4f} ± {std_metrics['time_sensitivity']:.4f}\n"
        summary_text += f"Time - Specificity: {mean_metrics['time_specificity']:.4f} ± {std_metrics['time_specificity']:.4f}\n"
        summary_text += f"Time - PPV: {mean_metrics['time_ppv']:.4f} ± {std_metrics['time_ppv']:.4f}\n"
        summary_text += f"Time - NPV: {mean_metrics['time_npv']:.4f} ± {std_metrics['time_npv']:.4f}\n"

        print(summary_text)

        if last_log_filename:
            with open(last_log_filename, "a") as f:
                f.write(summary_text)

        summary_file = os.path.join(args.output_dir, "5fold_summary_mean_std.txt")
        with open(summary_file, "w") as f:
            f.write(summary_text)


if __name__ == "__main__":
    main()
