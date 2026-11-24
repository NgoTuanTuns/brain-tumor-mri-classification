import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from mri_classifier.data import build_dataloaders, download_dataset
from mri_classifier.evaluate import evaluate_model, format_report
from mri_classifier.models import build_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    use_early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 1e-4,
) -> tuple[nn.Module, dict]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_correct / len(train_loader.dataset)

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        epoch_val_loss = val_metrics["loss"]
        epoch_val_acc = val_metrics["acc"]

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} || "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        if use_early_stopping:
            if epoch_val_loss < best_val_loss - min_delta:
                best_val_loss = epoch_val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"-> Early stopping at epoch {epoch + 1}. "
                        f"Restoring best weights from epoch {best_epoch} (val_loss={best_val_loss:.4f})."
                    )
                    break

    if use_early_stopping:
        model.load_state_dict(best_weights)

    return model, history


def run_experiment(exp_name: str, cfg: dict, data_dir: Path, device: torch.device) -> dict:
    common_cfg = cfg["common"]
    aug_cfg = cfg["augmentation"]
    exp_cfg = cfg["experiments"][exp_name]
    es_cfg = exp_cfg.get("early_stopping", {})

    print(f"\n=== Running experiment: {exp_name} (device={device}) ===")
    set_seed(common_cfg["seed"])

    loaders = build_dataloaders(
        data_dir=data_dir,
        common_cfg=common_cfg,
        aug_cfg=aug_cfg,
        use_augmentation=exp_cfg["use_augmentation"],
    )

    model = build_model(exp_cfg, num_classes=common_cfg["num_classes"]).to(device)
    model, history = train_model(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        device=device,
        num_epochs=exp_cfg["num_epochs"],
        lr=exp_cfg["lr"],
        weight_decay=exp_cfg["weight_decay"],
        use_early_stopping=es_cfg.get("enabled", False),
        patience=es_cfg.get("patience", 5),
        min_delta=es_cfg.get("min_delta", 1e-4),
    )

    test_metrics = evaluate_model(model, loaders["test_loader"], nn.CrossEntropyLoss(), device)
    print(
        f"[{exp_name}] Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Acc: {test_metrics['acc']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}\n"
    )
    print(format_report(test_metrics["y_true"], test_metrics["y_pred"], loaders["classes"]))

    return {"history": history, "test_metrics": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI CNN experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--exp", type=str, default="combined_es", help="Experiment name or 'all'")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional local dataset path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or download_dataset(cfg["common"]["dataset"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_names = list(cfg["experiments"].keys()) if args.exp == "all" else [args.exp]
    for name in exp_names:
        run_experiment(name, cfg, data_dir, device)


if __name__ == "__main__":
    main()