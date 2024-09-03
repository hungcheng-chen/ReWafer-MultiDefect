import argparse

import timm
from datasets import AUGMENTATION_TRANSFORMS, BASIC_TRANSFORMS, ImageFolder
from logger import Logger
from opts import opts
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import seed_everything


def main(opt: argparse.Namespace, logger: Logger):
    seed_everything(opt.seed)  # Set random seed
    # Load training dataset with augmentations
    train_dataset = ImageFolder(
        root=opt.data_dir,
        is_train=True,
        seed=opt.seed,
        transform=AUGMENTATION_TRANSFORMS(),
    )
    # Load validation dataset without augmentations
    val_dataset = ImageFolder(
        root=opt.data_dir,
        is_train=False,
        seed=opt.seed,
        transform=BASIC_TRANSFORMS(),
    )
    # Training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    # Validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    # Create model
    model = timm.create_model(
        opt.model_name, pretrained=opt.pretrained, num_classes=opt.num_classes
    )
    # Initialize trainer
    trainer = Trainer(opt, model, train_loader, val_loader, logger)
    trainer.run()  # Start training and validation
    logger.close()  # Close logger


if __name__ == "__main__":
    opt = opts().parse()
    logger = Logger(opt)
    main(opt, logger)
