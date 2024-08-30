# %%

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm


# %%
def get_random_sampler(dataset, subset_size=1000, random_seed=42):
    # データセットのインデックス配列を作成する
    indices = list(range(len(dataset)))
    # インデックスをシャッフルする
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    # シャッフルしたインデックスからsubset_size分だけ取得する
    subset_indices = indices[:subset_size]
    # SubsetRandomSamplerにインデックスを渡すことで、そのインデックスのデータをサンプリングする
    return SubsetRandomSampler(subset_indices)


# %%
def train_epoch(model, train_loader, criterion, optimizer, device):
    # モデルをtrainモードにする
    model.train()
    # 損失を記録する変数を定義
    running_loss = 0.0

    # ミニバッチごとにループを回す
    for images, labels in tqdm(train_loader, total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        # 勾配を初期化する
        optimizer.zero_grad()
        # 準伝搬
        outputs = model(images)
        # 損失関数を計算
        loss = criterion(outputs, labels)
        # 逆伝搬
        loss.backward()
        # パラメータ更新
        optimizer.step()

        # ミニバッチの損失を計算し記録する
        running_loss += loss.item()

    # 1エポックあたりの平均損失を計算する
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate_epoch(model, val_loader, criterion, device):
    # モデルをevalモードにする
    model.eval()
    # 損失を記録する変数を定義
    running_loss = 0.0

    all_output = []
    all_labels = []
    # 勾配計算をしないようにする(推論なので)
    with torch.no_grad():
        # ミニバッチごとにループを回す
        for images, labels in tqdm(val_loader, total=len(val_loader)):
            # デバイスの指定
            images, labels = images.to(device), labels.to(device)
            # 準伝搬
            outputs = model(images)
            all_output.append(outputs)
            all_labels.append(labels)
            # 損失計算
            loss = criterion(outputs, labels)
            # 損失を記録する
            running_loss += loss.item()

    # 1エポックあたりの平均損失を計算する
    avg_loss = running_loss / len(val_loader)
    # テストデータの予測結果を取得する
    all_output = torch.cat(all_output, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()
    return avg_loss, all_output, all_labels


def get_cifar10_train_test_loader(
    train_samples: int = 1000,
    test_samples: int = 1000,
    resize: tuple[int, int] = (256, 256),
    batch_size: int = 32,
):
    # 画像を256x156にリサイズして、テンソルに変換する
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

    # 学習データセットの作成
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # データセットからランダムにデータを取得する
    train_sampler = get_random_sampler(train_dataset, train_samples)
    # 学習DataLoaderの作成
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    # 検証データセットの作成
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    # データセットからランダムにデータを取得する
    test_sampler = get_random_sampler(test_dataset, test_samples)
    # 検証DataLoaderの作成
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader


# %%
# 学習コードの作成
