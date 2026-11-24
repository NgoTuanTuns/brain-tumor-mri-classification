import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)

    avg_loss = running_loss / len(loader.dataset)
    acc = float((y_pred_arr == y_true_arr).mean())
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "y_probs": y_probs_arr,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr),
    }


def format_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    return classification_report(y_true, y_pred, target_names=classes, digits=4)