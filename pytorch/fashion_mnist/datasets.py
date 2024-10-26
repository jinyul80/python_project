import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloader(batch_size: int, train_transform=None, validation_transform=None):

    if train_transform == None:
        train_transform = transforms.ToTensor()

    if validation_transform == None:
        validation_transform = transforms.ToTensor()

    # 데이터 준비
    data_path = os.path.join("/root", "pg_source", "data", "FashionMNIST_data")
    train_dataset = datasets.FashionMNIST(
        root=data_path,
        train=True,  # 학습 데이터
        transform=train_transform,  # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
        download=True,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_path,
        train=False,  # 테스트 데이터
        transform=validation_transform,  # 0~255까지의 값을 0~1 사이의 값으로 변환시켜줌
        download=True,
    )

    # train, validation 분리
    train_size = int(len(train_dataset) * 0.8)
    validation_size = len(train_dataset) - train_size

    train_dataset, validation_dataset = random_split(
        train_dataset, [train_size, validation_size]
    )

    # train_dataset의 shape을 출력
    print("train_dataset shape:", train_dataset.dataset.data.shape)
    print(len(train_dataset), len(validation_dataset), len(test_dataset))

    train_dataset_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_dataset_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_dataset_loader, validation_dataset_loader, test_dataset_loader
