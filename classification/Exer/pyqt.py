import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

import test
from src.dataset import ExerciseDataset
import torch

import random


def on_button1_clicked():
    global img_idx
    img_idx = random.randint(0, len(test_data))

    img_tensor = test_data[img_idx][0]
    image_np = img_tensor
    height, width, channel = image_np.shape
    bytesPerLine = channel * width
    q_img = QImage(image_np.data, width, height, bytesPerLine, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img).scaled(448, 448, Qt.KeepAspectRatio)
    image_label.setPixmap(pixmap)
    
    text_label.setText(f'Loaded Image Index: {img_idx}\n') 

def on_button2_clicked():
    img_pred, img_actual = test.test(device='cuda', idx=img_idx)
    
    text_label.setText(f"Predict: {test_data.classes[img_pred]}  /  Actual: {test_data.classes[img_actual]}\n")
    print("\nDone")


def main():
    app = QApplication(sys.argv)                # QApplication 인스턴스 생성

    # 윈도우 및 메인 레이아웃
    window = QWidget()                          # 메인 윈도우 생성
    layout = QVBoxLayout()                      # 메인 수직 레이아웃 생성

    # 수직 레이아웃
    vertical_layout = QVBoxLayout()             # 수직 레이아웃 생성

    # 이미지 표시를 위한 QLabel 위젯
    global image_label  # 이미지 표시용 QLabel 위젯을 전역 변수로 선언합니다.
    image_label = QLabel()  # QLabel 위젯 생성
    image_label.setAlignment(Qt.AlignCenter)  # 이미지를 중앙에 정렬합니다.
    image_label.setFixedSize(800, 500)  # 이미지 표시용 QLabel 위젯의 크기를 설정합니다.
    layout.addWidget(image_label)  # 이미지 표시용 QLabel 위젯을 레이아웃에 추가


    # 수직 레이아웃을 메인 수직 레이아웃에 추가
    layout.addLayout(vertical_layout)           # 메인 레이아웃에 수직 레이아웃 추가

    # 이미지 데이터
    global test_data
    test_data = ExerciseDataset("./images/test")

    # 텍스트
    global text_label
    text_label = QLabel()
    text_label.setAlignment(Qt.AlignCenter)  # 텍스트를 중앙에 정렬합니다.
    text_label.setFont(QFont("Times New Roman", 20))  # 폰트 크기 조정
    layout.addWidget(text_label)

    layout.addSpacing(50)

    # 버튼 1
    button = QPushButton('Random Image Load')           # 버튼 위젯 생성
    button.clicked.connect(on_button1_clicked)  # 버튼 클릭 시 이벤트 처리
    layout.addWidget(button)                   # 메인 레이아웃에 버튼 2 추가

    # 버튼 2
    button = QPushButton('Image Predict')           # 버튼 위젯 생성
    button.clicked.connect(on_button2_clicked)  # 버튼 클릭 시 이벤트 처리
    layout.addWidget(button)                   # 메인 레이아웃에 버튼 2 추가


    # 윈도우 설정
    window.setLayout(layout)                    # 레이아웃 설정
    window.setGeometry(1200, 400, 800, 600)  # 윈도우 위치와 크기 설정
    window.setWindowTitle("Image Viewer")  # 윈도우 제목 설정
    
    window.show()                               # 메인 윈도우 표시
    
    sys.exit(app.exec_())                       # 이벤트 루프 시작


if __name__ == '__main__':
    main()