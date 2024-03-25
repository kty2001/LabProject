import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import ResNet152Classifier
from src.model import ResNet50Classifier
from src.model import ResNet18Classifier
from src.dataset import ExerciseDataset


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="학습에 사용되는 장치")
args = parser.parse_args()


def train_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module, optimizer) -> None:
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
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    return loss


def valid_one_epoch(dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module) -> None:
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

    return [100*correct, test_loss]


def train(device: str):
    num_classes = 5
    batch_size = 32
    epochs = 30
    lr = 1e-3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ExerciseDataset("./images/train", transform=transform)
    testset = ExerciseDataset("./images/test", transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("------trianing by", device, "------")

    model = ResNet152Classifier(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_results = []
    valid_results = []

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_result = train_one_epoch(train_loader, device, model, loss_fn, optimizer)
        valid_result = valid_one_epoch(test_loader, device, model, loss_fn)
        train_results.append(train_result)
        valid_results.append(valid_result)
    print('Done!')

    torch.save(model.state_dict(), 'exercise-net.pth')
    print('Saved PyTorch Model State to exercise-net.pth')

    for i in range(len(train_results)):
        print(f"Epoch {i+1} - train_loss: {train_results[i]:>5f} / valid_loss: {valid_results[i][1]:>5f} / accuracy: {valid_results[i][0]:>0.1f}%")


if __name__ == "__main__":
    train(device=args.device)