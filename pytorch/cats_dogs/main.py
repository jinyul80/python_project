import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
import tqdm

# 전역 변수 설정
ROOT_DIR = os.path.join("/root", "pg_source", "data", "cats_and_dogs_filtered")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
VALIDATION_DIR = os.path.join(ROOT_DIR, "validation")
BATCH_SIZE = 32

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"using PyTorch version: {torch.__version__}, Device: {DEVICE}")


def get_dataloader():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    validation_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, train_transform)
    validation_dataset = datasets.ImageFolder(VALIDATION_DIR, validation_transform)

    train_dataset_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_dataset_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # 이미지 확인
    test_datasets(train_dataset, train_dataset_loader)

    return train_dataset_loader, validation_dataset_loader


def test_datasets(
    train_dataset: datasets.ImageFolder, train_dataset_loader: DataLoader
):
    images, labels = next(iter(train_dataset_loader))

    labels_map = {v: k for k, v in train_dataset.class_to_idx.items()}

    figure = plt.figure(figsize=(6, 7))

    cols, rows = 4, 4

    # 이미지 출력
    for i in range(1, cols * rows + 1):

        sample_idx = torch.randint(len(images), size=(1,)).item()
        img, label = images[sample_idx], labels[sample_idx].item()

        figure.add_subplot(rows, cols, i)

        plt.title(labels_map[label])
        plt.axis("off")

        # 본래 이미지의 shape은 (3, 224, 224) 인데,
        # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (224, 224, 3)으로 shape 변경을 한 후 시각화
        plt.imshow(torch.permute(img, (1, 2, 0)))

    plt.show()


class MyTransferLearningModel(torch.nn.Module):

    def __init__(self, pretrained_model, feature_extractor):

        super().__init__()

        if feature_extractor:
            for param in pretrained_model.parameters():
                param.require_grad = False

        # vision transformer 에서의 classifier 부분은 heads 로 지정
        pretrained_model.heads = torch.nn.Sequential(
            torch.nn.Linear(pretrained_model.heads[0].in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 2),
        )

        self.model = pretrained_model

    def forward(self, data):

        logits = self.model(data)

        return logits


def model_train(dataloader, model, loss_function, optimizer):

    model.train()

    train_loss_sum = train_correct = train_total = 0

    total_train_batch = len(dataloader)

    for images, labels in tqdm.tqdm(dataloader):

        x_train = images.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        train_total += y_train.size(0)
        train_correct += ((torch.argmax(outputs, 1) == y_train)).sum().item()

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100 * train_correct / train_total

    return (train_avg_loss, train_avg_accuracy)


def model_evaluate(dataloader, model, loss_function, optimizer):

    model.eval()

    with torch.no_grad():

        val_loss_sum = val_correct = val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in tqdm.tqdm(dataloader):

            x_val = images.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, 1) == y_val)).sum().item()

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = 100 * val_correct / val_total

    return (val_avg_loss, val_avg_accuracy)


def main():

    # 데이터로더 객체 생성
    train_dataset_loader, validation_dataset_loader = get_dataloader()

    # 사전 학습 모델 다운로드
    pretrained_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    print(pretrained_model)

    feature_extractor = False  # True: Feature Extractor,  False: Fine Tuning

    # 학습 모델 생성
    model = MyTransferLearningModel(pretrained_model, feature_extractor).to(DEVICE)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    from datetime import datetime

    train_loss_list = []
    train_accuracy_list = []

    val_loss_list = []
    val_accuracy_list = []

    start_time = datetime.now()

    EPOCHS = 10

    for epoch in range(EPOCHS):

        # ==============  model train  ================
        train_avg_loss, train_avg_accuracy = model_train(
            train_dataset_loader, model, loss_function, optimizer
        )

        train_loss_list.append(train_avg_loss)
        train_accuracy_list.append(train_avg_accuracy)
        # =============================================

        # ============  model evaluation  ==============
        val_avg_loss, val_avg_accuracy = model_evaluate(
            validation_dataset_loader, model, loss_function, optimizer
        )

        val_loss_list.append(val_avg_loss)
        val_accuracy_list.append(val_avg_accuracy)
        # ============  model evaluation  ==============

        print(
            "epoch:",
            "%02d" % (epoch + 1),
            "train loss =",
            "{:.3f}".format(train_avg_loss),
            "train acc =",
            "{:.3f}".format(train_avg_accuracy),
            "val loss =",
            "{:.3f}".format(val_avg_loss),
            "val acc =",
            "{:.3f}".format(val_avg_accuracy),
        )

    end_time = datetime.now()

    print("elapsed time => ", end_time - start_time)

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


if __name__ == "__main__":
    main()
