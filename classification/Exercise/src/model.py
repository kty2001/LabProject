from torch import nn, Tensor
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Classifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet18(x)