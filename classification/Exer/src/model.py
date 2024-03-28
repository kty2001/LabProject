from torch import nn, Tensor
import torchvision.models as models
import warnings

warnings.filterwarnings("ignore")


class EfficientNetB1Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, lastconv_output_channels=1280):
        super(EfficientNetB1Classifier, self).__init__()
        self.efficientnet_b1 = models.efficientnet_b1(pretrained=True)
        self.efficientnet_b1.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
            )

    def forward(self, x):
        return self.efficientnet_b1(x)

class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, lastconv_output_channels=1536):
        super(EfficientNetB3Classifier, self).__init__()
        self.efficientnet_b3 = models.efficientnet_b3(pretrained=True)
        self.efficientnet_b3.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
            )

    def forward(self, x):
        return self.efficientnet_b3(x)
    
class EfficientNetB5Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, lastconv_output_channels=2048):
        super(EfficientNetB5Classifier, self).__init__()
        self.efficientnet_b5 = models.efficientnet_b5(pretrained=True)
        self.efficientnet_b5.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
            )

    def forward(self, x):
        return self.efficientnet_b5(x)

class EfficientNetB7Classifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, lastconv_output_channels=2560):
        super(EfficientNetB7Classifier, self).__init__()
        self.efficientnet_b7 = models.efficientnet_b7(pretrained=True)
        self.efficientnet_b7.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
            )

    def forward(self, x):
        return self.efficientnet_b7(x)

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