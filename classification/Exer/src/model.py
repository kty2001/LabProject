from torch import nn, Tensor
import torchvision.models as models
import warnings

warnings.filterwarnings("ignore")

class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet152Classifier, self).__init__()
        self.resnet152 = models.resnet152(pretrained=True)
        num_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet152(x)
    
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Classifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet18(x)
    
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Classifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet50(x)