import os
from datetime import datetime
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm

EPOCHS = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


print(f"using PyTorch version: {torch.__version__}, Device: {DEVICE}")


def get_dataloader():
    # 데이터 준비
    data_path = os.path.join("/root", "pg_source", "data", "MNIST_data")
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,  # 학습 데이터
        transform=transforms.ToTensor(),  # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,  # 테스트 데이터
        transform=transforms.ToTensor(),  # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
        download=True,
    )

    # train, validation 분리
    train_size = int(len(train_dataset) * 0.8)
    validation_size = len(train_dataset) - train_size

    train_dataset, validation_dataset = random_split(
        train_dataset, [train_size, validation_size]
    )

    print(len(train_dataset), len(validation_dataset), len(test_dataset))

    train_dataset_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_dataset_loader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataset_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return train_dataset_loader, validation_dataset_loader, test_dataset_loader


# 모델 구현
class MyCNNModel(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p=0.5)

    def forward(self, data):

        data = self.conv1(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = self.conv2(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = data.view(-1, 7 * 7 * 64)

        data = self.fc1(data)
        data = torch.relu(data)
        data = self.dropout50(data)

        logits = self.fc2(data)

        return logits


def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    total_train_batch = len(dataloader)

    for images, labels in tqdm(dataloader):  # images에는 이미지, labels에는 0-9 숫자

        x_train = images.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_total += y_train.size(0)
        train_correct += (
            ((torch.argmax(outputs, 1) == y_train)).sum().item()
        )  # 예측한 값과 일치한 값의 합

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100 * train_correct / train_total

    return (train_avg_loss, train_avg_accuracy)


def model_evaluate(dataloader, model, loss_function):

    model.eval()

    with torch.no_grad():  # 미분하지 않겠다는 것

        val_loss_sum = 0
        val_correct = 0
        val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in tqdm(
            dataloader
        ):  # images에는 이미지, labels에는 0-9 숫자

            x_val = images.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)  # label 열 사이즈 같음
            val_correct += (
                ((torch.argmax(outputs, 1) == y_val)).sum().item()
            )  # 예측한 값과 일치한 값의 합

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = 100 * val_correct / val_total

    return (val_avg_loss, val_avg_accuracy)


def model_test(dataloader, model, loss_function):

    model.eval()

    with torch.no_grad():  # test set으로 데이터를 다룰 때에는 gradient를 주면 안된다.

        test_loss_sum = 0
        test_correct = 0
        test_total = 0

        total_test_batch = len(dataloader)

        for images, labels in tqdm(
            dataloader
        ):  # images에는 이미지, labels에는 0-9 숫자

            x_test = images.to(DEVICE)
            y_test = labels.to(DEVICE)

            outputs = model(x_test)
            loss = loss_function(outputs, y_test)

            test_loss_sum += loss.item()

            test_total += y_test.size(0)  # label 열 사이즈 같음
            test_correct += (
                ((torch.argmax(outputs, 1) == y_test)).sum().item()
            )  # 예측한 값과 일치한 값의 합

        test_avg_loss = test_loss_sum / total_test_batch
        test_avg_accuracy = 100 * test_correct / test_total

        print("accuracy:", test_avg_accuracy)
        print("loss:", test_avg_loss)


def main():
    train_dataset_loader, validation_dataset_loader, test_dataset_loader = (
        get_dataloader()
    )

    model = MyCNNModel().to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss_list = []
    train_accuracy_list = []

    val_loss_list = []
    val_accuracy_list = []

    start_time = datetime.now()

    for epoch in range(EPOCHS):
        # ==============  model train  ================
        train_avg_loss, train_avg_accuracy = model_train(
            train_dataset_loader, model, loss_function, optimizer
        )  # training

        train_loss_list.append(train_avg_loss)
        train_accuracy_list.append(train_avg_accuracy)

        # ============  model evaluation  ==============
        val_avg_loss, val_avg_accuracy = model_evaluate(
            validation_dataset_loader, model, loss_function
        )  # evaluation

        val_loss_list.append(val_avg_loss)
        val_accuracy_list.append(val_avg_accuracy)

        print(
            "epoch:",
            "%02d" % (epoch + 1),
            "train loss =",
            "{:.4f}".format(train_avg_loss),
            "train accuracy =",
            "{:.4f}".format(train_avg_accuracy),
            "validation loss =",
            "{:.4f}".format(val_avg_loss),
            "validation accuracy =",
            "{:.4f}".format(val_avg_accuracy),
        )

    end_time = datetime.now()

    print("elapsed time => ", end_time - start_time)

    # test dataset 으로 정확도 및 오차 테스트
    model_test(test_dataset_loader, model, loss_function)


if __name__ == "__main__":
    main()
