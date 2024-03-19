import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Classifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet18(x)

"""# 예시 데이터셋을 불러옵니다. 여기서는 CIFAR-10을 사용합니다.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)"""

# 모델을 초기화하고 손실 함수 및 옵티마이저를 정의합니다.
net = ResNet18Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""# 모델을 훈련합니다.
for epoch in range(2):  # 데이터셋을 여러 번 훈련합니다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 입력 데이터를 받아옵니다.
        inputs, labels = data

        # 변화도를 0으로 초기화합니다.
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 수행합니다.
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # 매 2000 미니배치마다 출력합니다.
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0"""

print('Finished Training')