import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import FashionMnistDataset
from src.model import NeuralNetwork
from src.utils import split_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()

def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        preds = model(images)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.flatten(targets)

            preds = model(images)

            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).float().sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

def train(device):
    image_dir = "data/fashion-mnist/images"
    csv_path = 'data/fashion-mnist/answers.csv'
    train_csv_path = 'data/fashion-mnist/train_answer.csv'
    test_csv_path = 'data/fashion-mnist/test_answer.csv'

    num_classes = 10
    batch_size = 32
    epochs = 5
    lr = 1e-3

    split_dataset(csv_path)

    trianing_data = FashionMnistDataset(
        image_dir,
        train_csv_path,
    )
    test_data = FashionMnistDataset(
        image_dir,
        test_csv_path,
    )

    train_dataloader = DataLoader(trianing_data, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NeuralNetwork(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for t in range(epochs):
        print(f'Epoch {t+1} \n-------------------------')
        train_one_epoch(train_dataloader, device, model, loss_fn, optimizer)
        val_one_epoch(test_dataloader, device, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), 'fashion-mnist-net.pth')
    print('Saved PyTorch Model State to fashion-mnist-net.pth')

if __name__ == "__main__":
    train(args.device)