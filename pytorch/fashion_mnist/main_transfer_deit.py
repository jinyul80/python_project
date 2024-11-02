import logging
from datetime import datetime
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torchmetrics import Accuracy

from datasets import get_dataloader

# from models import MyDeiTModel
from engines import train_one_epoch, eval_one_epoch, show_train_history

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n##################################################")
print(f"using PyTorch version: {torch.__version__}, Device: {DEVICE}")
print("##################################################\n")


def main():
    logging.basicConfig(
        filename="log/vit_transfer.log",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ]
    )

    train_loader, validation_loader, test_loader = get_dataloader(
        BATCH_SIZE, train_transform, validation_transform
    )

    print(torch.__version__)

    model = torch.hub.load(
        "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True
    )

    print(model)
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)

    model.to(DEVICE)

    metric_fn = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    metric_fn = metric_fn.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    train_loss_list = []
    train_accuracy_list = []

    val_loss_list = []
    val_accuracy_list = []

    start_time = datetime.now()

    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        # ==============  model train  ================
        train_summary = train_one_epoch(
            train_loader,
            model,
            loss_fn,
            metric_fn,
            DEVICE,
            optimizer,
            scheduler,
        )

        train_loss_list.append(train_summary["loss"])
        train_accuracy_list.append(train_summary["accuracy"])

        # ============  model evaluation  ==============
        val_summary = eval_one_epoch(
            validation_loader, model, loss_fn, metric_fn, DEVICE
        )

        val_loss_list.append(val_summary["loss"])
        val_accuracy_list.append(val_summary["accuracy"])

        log = (
            f"epoch {epoch+1}, "
            + f'train_loss: {train_summary["loss"]:.4f}, '
            + f'train_accuracy: {train_summary["accuracy"]:.4f}, '
            + f'val_loss: {val_summary["loss"]:.4f}, '
            + f'val_accuracy: {val_summary["accuracy"]:.4f}'
        )
        logging.info(log)

    end_time = datetime.now()
    logging.info(f"elapsed time => {end_time - start_time}")

    # test dataset 으로 정확도 및 오차 테스트
    test_summary = eval_one_epoch(test_loader, model, loss_fn, metric_fn, DEVICE)

    log = (
        f'test_loss: {test_summary["loss"]:.4f}, '
        + f'test_accuracy: {test_summary["accuracy"]:.4f}'
    )
    logging.info(log)

    # 학습 이력 그래프
    show_train_history(
        train_loss_list,
        train_accuracy_list,
        val_loss_list,
        val_accuracy_list,
        "log/vit_learning_history.png",
    )


if __name__ == "__main__":
    main()
