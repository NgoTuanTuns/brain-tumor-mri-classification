from pathlib import Path

import kagglehub
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _to_rgb(img):
    return img.convert("RGB")


def build_transforms(
    img_size: int = 128,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    aug_cfg: dict | None = None,
):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    aug_cfg = aug_cfg or {}
    jitter_cfg = aug_cfg.get("color_jitter", {})

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    aug_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(_to_rgb),
            transforms.RandomHorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
            transforms.RandomRotation(degrees=aug_cfg.get("rotation_degrees", 15)),
            transforms.ColorJitter(
                brightness=jitter_cfg.get("brightness", 0.15),
                contrast=jitter_cfg.get("contrast", 0.15),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return eval_transform, aug_transform


def split_train_val_indices(
    targets: list[int],
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    indices = list(range(len(targets)))
    return train_test_split(
        indices,
        test_size=val_split,
        stratify=targets,
        random_state=seed,
    )


def download_dataset(dataset_slug: str = "masoudnickparvar/brain-tumor-mri-dataset") -> Path:
    return Path(kagglehub.dataset_download(dataset_slug))


def build_dataloaders(
    data_dir: str | Path,
    common_cfg: dict,
    aug_cfg: dict,
    use_augmentation: bool = False,
) -> dict:
    data_dir = Path(data_dir)
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    norm_cfg = common_cfg["normalize"]
    eval_transform, aug_transform = build_transforms(
        img_size=common_cfg["img_size"],
        mean=norm_cfg["mean"],
        std=norm_cfg["std"],
        aug_cfg=aug_cfg,
    )

    full_train_base = datasets.ImageFolder(train_dir, transform=eval_transform)
    train_idx, val_idx = split_train_val_indices(
        targets=full_train_base.targets,
        val_split=common_cfg["val_split"],
        seed=common_cfg["seed"],
    )

    train_tf = aug_transform if use_augmentation else eval_transform
    train_dataset = Subset(datasets.ImageFolder(train_dir, transform=train_tf), train_idx)
    val_dataset = Subset(datasets.ImageFolder(train_dir, transform=eval_transform), val_idx)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    batch_size = common_cfg["batch_size"]
    num_workers = common_cfg["num_workers"]
    pin_memory = torch.cuda.is_available()

    return {
        "train_loader": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val_loader": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test_loader": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "classes": full_train_base.classes,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }