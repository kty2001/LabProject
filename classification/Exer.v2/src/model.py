from torch import nn, Tensor
import torchvision.models as models


class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet152Classifier, self).__init__()
        self.resnet152 = models.resnet152(pretrained=False)
        num_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet152(x)