import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet101_Weights
from torchmetrics import Accuracy

from lib.datasets import CIFAR10_MEAN, CIFAR10_STD
from lib.models import ResNet101
from lib.engines import train_one_epoch, eval_one_epoch
from lib.utils import save_model

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="finetune", type=str)
    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=5e-2, type=float)
    args = parser.parse_args()
    return args


def main(args):
    # -------------------------------------------------------------------------
    # Set Logger
    # -------------------------------------------------------------------------
    logging.basicConfig(
        filename=f"{args.title}.log",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    # -------------------------------------------------------------------------
    # Data Processing Pipeline
    # -------------------------------------------------------------------------
    train_transform = T.Compose(
        [
            T.RandomCrop(size=(32, 32), padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.RandAugment(num_ops=2, magnitude=5),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomErasing(p=0.25),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_data = CIFAR10(
        args.data, train=True, download=True, transform=train_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    val_transform = T.Compose(
        [
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    val_data = CIFAR10(args.data, train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = ResNet101(
        weights=ResNet101_Weights.IMAGENET1K_V2, num_classes=args.num_classes
    )
    model = model.to(args.device)

    # -------------------------------------------------------------------------
    # Performance Metic, Loss Function, Optimizer
    # -------------------------------------------------------------------------
    metric_fn = Accuracy(task="multiclass", num_classes=args.num_classes)
    metric_fn = metric_fn.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        # train one epoch
        train_summary = train_one_epoch(
            model, train_loader, metric_fn, loss_fn, args.device, optimizer, scheduler
        )

        # evaluate one epoch
        val_summary = eval_one_epoch(model, val_loader, metric_fn, loss_fn, args.device)

        # write log
        log = (
            f"epoch {epoch+1}, "
            + f'train_loss: {train_summary["loss"]:.4f}, '
            + f'train_accuracy: {train_summary["accuracy"]:.4f}, '
            + f'val_loss: {val_summary["loss"]:.4f}, '
            + f'val_accuracy: {val_summary["accuracy"]:.4f}'
        )
        logging.info(log)

        # save model
        checkpoint_path = f"{args.title}_last.pt"
        save_model(checkpoint_path, model, optimizer, scheduler, epoch + 1)


if __name__ == "__main__":
    args = get_args()
    main(args)
