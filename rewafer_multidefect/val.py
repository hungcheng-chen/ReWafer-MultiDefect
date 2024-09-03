import argparse

import timm
import torch
from datasets import BASIC_TRANSFORMS, ImageFolder
from opts import opts
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from utils import seed_everything


def main(opt: argparse.Namespace):
    seed_everything(opt.seed)  # Set random seed
    val_dataset = ImageFolder(
        root=opt.data_dir,
        is_train=False,
        seed=opt.seed,
        transform=BASIC_TRANSFORMS(),
    )  # Load validation dataset with basic transforms
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
    )  # Validation data loader
    model = timm.create_model(
        opt.model_name, pretrained=False, num_classes=opt.num_classes
    )  # Create model
    model.load_state_dict(torch.load(opt.load_model_path))  # Load model weights
    model = model.to(opt.device)  # Move model to device

    # Accuracy metric
    acc_metric = Accuracy(task="multiclass", num_classes=opt.num_classes).to(opt.device)

    model.eval()  # Switch to evaluation mode
    acc_metric.reset()  # Reset accuracy metric
    with torch.no_grad():  # Disable gradient computation
        pbar = tqdm(val_loader, desc=f"[{opt.device}]")  # Progress bar
        for source, targets in pbar:
            # Move data to device
            source, targets = source.to(opt.device), targets.to(opt.device)
            output = model(source).squeeze()  # Get model output
            acc_metric.update(output, targets)  # Update accuracy metric
            # Display accuracy
            pbar.set_description(f"Accuracy {100 * acc_metric.compute().item():.2f}%")


if __name__ == "__main__":
    """Test mode must be enabled and model weight path specified."""
    opt = opts().parse()
    main(opt)
