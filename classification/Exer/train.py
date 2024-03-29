import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.model import EfficientNetB7Classifier
from src.model import EfficientNetB5Classifier
from src.model import EfficientNetB3Classifier
from src.model import EfficientNetB1Classifier
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
    current = 0
    for batch, (images, targets) in enumerate(dataloader):
        current += len(targets)

        images = images.to(device)
        targets = targets.to(device)
        targets = torch.flatten(targets)

        preds = model(images)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if current % 320 == 0:
            print(f'loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]')
        elif current == size:
            print(f'loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]')

    return loss.item()

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

    return test_loss, correct

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def visualization(train_results, valid_results, corrects, epochs):

    train_loss = np.array(train_results)
    valid_loss = np.array(valid_results)
    correct = np.array(corrects)
    smoothed_train_loss = smooth_curve(train_loss, factor=0.9)
    smoothed_valid_loss = smooth_curve(valid_loss, factor=0.9)
    smoothed_correct = smooth_curve(correct, factor=0.8)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss, label='train_loss', color="b", alpha=0.2)
    plt.plot(range(1, epochs + 1), smoothed_train_loss, label='Smoothed train_loss', color="b")
    plt.plot(range(1, epochs + 1), valid_loss, label='valid_loss', color="r", alpha=0.2)
    plt.plot(range(1, epochs + 1), smoothed_valid_loss, label='Smoothed valid_loss', color="r")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Graph')
    plt.legend()
    
    # Correct 시각화
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), correct, label='correct')
    plt.plot(range(1, epochs + 1), smoothed_correct, label='Smoothed correct')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Smoothed Correct')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train(device: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ExerciseDataset("./images/train", transform=transform)
    testset = ExerciseDataset("./images/test", transform=transform)

    num_classes = len(trainset.classes_dic)
    batch_size = 16
    epochs = 100
    lr = 1e-4

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    if device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n<<< trianing by", device, ">>>\n")

    model = EfficientNetB7Classifier(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_results = []
    valid_results = []
    corrects = []
    best_corrects = 0.0
    best_epoch = 0

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_results.append(train_one_epoch(train_loader, device, model, loss_fn, optimizer))
        valid_result, correct = valid_one_epoch(test_loader, device, model, loss_fn)
        valid_results.append(valid_result)
        corrects.append(correct)

        if corrects[-1] >= best_corrects:
            best_corrects = corrects[-1]
            best_epoch = t+1
            torch.save(model.state_dict(), 'best-exernet.pth')
    print('Done!')

    for i in range(epochs):
        print(f"Epoch {i+1:>2d} - train_loss: {train_results[i]:>5f} / valid_loss: {valid_results[i]:>5f} / accuracy: {100*corrects[i]:>0.1f}%")
    print()
    
    print(f'{best_epoch} Epoch: Saved Best Model State to best-exernet.pth')
    
    torch.save(model.state_dict(), 'exernet.pth')
    print('Saved PyTorch Model State to exernet.pth')
    
    visualization(train_results, valid_results, corrects, epochs)


if __name__ == "__main__":
    train(device=args.device)