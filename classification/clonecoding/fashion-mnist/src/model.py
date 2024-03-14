from torch import nn, Tensor


class NeuralNetwork(nn.Module):
    """FashionMNIST 데이터를 훈련할 모델을 정의"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # 모델 초기화
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """순전파 진행하는 함수
        
        param x: 입력 이미지
        type x: Tensor
        return: 입력 이미지에 대한 예측값
        rtype: Tensor
        """
        
        # 모델 진행 후 결과 반환
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits