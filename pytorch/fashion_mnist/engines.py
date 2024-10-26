import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchmetrics.aggregation import MeanMetric
import tqdm
from matplotlib import pyplot as plt


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    metric_fn: Accuracy,
    device: str,
    optimizer: torch.optim.Optimizer,
    scheduler,
):

    model.train()

    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()

    for images, labels in tqdm.tqdm(
        dataloader
    ):  # images에는 이미지, labels에는 0-9 숫자

        x_train = images.to(device)
        y_train = labels.to(device)

        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)
        accuracy = metric_fn(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_epoch.update(loss.to("cpu"))
        accuracy_epoch.update(accuracy.to("cpu"))

    summary = {
        "loss": loss_epoch.compute(),
        "accuracy": accuracy_epoch.compute(),
    }

    return summary


def eval_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    metric_fn: Accuracy,
    device: str = "cpu",
):

    model.eval()

    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()

    # images에는 이미지, labels에는 0-9 숫자
    for images, labels in tqdm.tqdm(dataloader):

        x_val = images.to(device)
        y_val = labels.to(device)

        with torch.no_grad():  # 미분하지 않겠다는 것
            outputs = model(x_val)

        loss = loss_fn(outputs, y_val)
        accuracy = metric_fn(outputs, y_val)

        loss_epoch.update(loss.to("cpu"))
        accuracy_epoch.update(accuracy.to("cpu"))

    summary = {
        "loss": loss_epoch.compute(),
        "accuracy": accuracy_epoch.compute(),
    }

    return summary


def show_train_history(
    train_loss_list: list[float],
    train_accuracy_list: list[float],
    val_loss_list: list[float],
    val_accuracy_list: list[float],
):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Loss Trend")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid()
    plt.plot(train_loss_list, label="train")
    plt.plot(val_loss_list, label="validation")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy Trend")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.grid()
    plt.plot(train_accuracy_list, label="train")
    plt.plot(val_accuracy_list, label="validation")
    plt.legend()

    plt.show()
