# =================================================================================================
# [초상세 주석 버전] ResNet50 기반 지도 학습 딥페이크 탐지 모델 (단일 이미지 예측)
# 이 스크립트는 사전 학습된 ResNet50 모델을 불러와 특정 이미지의 딥페이크 여부를 판단합니다.
# =================================================================================================

# ==============================================================================
# 섹션 1: 라이브러리 임포트 및 기본 설정
# ==============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch # PyTorch의 핵심 라이브러리
import torch.nn as nn # 신경망 모듈
from torchvision import transforms, models # 이미지 변환 도구 및 사전 학습된 모델
import numpy as np # 수치 연산
from PIL import Image # 이미지 파일 처리 (Pillow 라이브러리)

# --- 하이퍼파라미터 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # GPU 또는 CPU 설정
IMG_SIZE = 224 # ResNet 모델의 표준 입력 이미지 크기 (224x224). 모델 학습 시와 동일해야 합니다.
NUM_CLASSES = 2 # 분류할 클래스 수 (0: 진짜, 1: 가짜). 모델 학습 시와 동일해야 합니다.

# 불러올 모델 파일의 경로
MODEL_PATH = "./models/deepfake_resnet50_classifier_backup_cnn.pth"

# 예측할 이미지 파일의 경로 (사용자님의 'img' 파일)
# INPUT_IMAGE_PATH = "D:/PythonProject/diffusion_Model/data/09.jpg"
# INPUT_IMAGE_PATH = "D:/PythonProject/diffusion_Model/data/IMG_bad.jpeg"
# INPUT_IMAGE_PATH = "D:/PythonProject/GAN/MVTec/mvtec_anomaly_detection/bottle/train/good/real_132838_005_frame_005_tracked.jpg"
# INPUT_IMAGE_PATH = "D:\\PythonProject\\GAN\\new_fakes_from_real_reconstruction\\generated_fake_0019.png"
# INPUT_IMAGE_PATH = "D:\\PythonProject\\diffusion_Model\\0e23d546a5f952542a00_021.mp4_0_11.jpg"
# INPUT_IMAGE_PATH = "D:\\PythonProject\\Dlib_Preprocess\\fake_ai.jpg"

# ==============================================================================
# 섹션 2: 모델 정의 및 수정 (학습 시와 동일한 모델 구조)
# ==============================================================================

def get_model(num_classes=NUM_CLASSES):
    """사전 학습된 ResNet50 모델을 로드하고, 마지막 분류 레이어를 수정합니다."""
    # ImageNet으로 사전 학습된 ResNet50 모델을 불러옵니다.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # ResNet50의 마지막 완전 연결(Fully Connected) 레이어는 `fc`입니다.
    # 우리는 2개 클래스(진짜/가짜) 분류를 할 것이므로, 이 레이어를 새로 정의해야 합니다.
    num_ftrs = model.fc.in_features # `fc` 레이어의 입력 특징(feature) 개수를 가져옵니다.
    # model.fc = nn.Linear(num_ftrs, num_classes) # 새로운 `fc` 레이어를 2개 클래스 출력으로 정의합니다.
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    # 모델을 지정된 장치(GPU 또는 CPU)로 이동시킵니다.
    model = model.to(DEVICE)
    return model

# ==============================================================================
# 섹션 3: 이미지 전처리 파이프라인 정의 (학습 시와 동일)
# ==============================================================================

def get_transform():
    """모델 입력에 맞게 이미지를 전처리하는 파이프라인을 반환합니다."""
    # ImageNet으로 사전 학습된 모델을 사용하므로, ImageNet의 평균과 표준편차로 정규화합니다.
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # 이미지 크기 조절
        transforms.ToTensor(), # 이미지를 텐서로 변환 (0-1 범위)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 정규화
    ])
    return transform

# ==============================================================================
# 섹션 4: 메인 예측 함수
# ==============================================================================

def predict_deepfake(image_path):
    """주어진 이미지 경로의 파일이 딥페이크인지 아닌지 예측합니다."""
    # 1. 모델 로드
    model = get_model(NUM_CLASSES) # 모델 구조를 가져옵니다.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # 저장된 모델 파일이 존재하는지 확인합니다.
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
        print("'CNN_Detection.py'를 먼저 실행하여 모델을 학습하고 저장해주세요.")
        return None, None
    '''
    # 저장된 모델의 가중치를 불러옵니다.
    # map_location=DEVICE는 GPU에서 학습된 모델을 CPU 환경에서도 불러올 수 있게 해줍니다.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    '''

    if list(checkpoint.keys())[0].startswith('module.'):
        # 'module.' 접두사를 제거한 새로운 state_dict를 만듭니다.
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        # model.load_state_dict(new_state_dict, strict=False)
        model.load_state_dict(new_state_dict, strict=False)
        print(f"'{MODEL_PATH}'에서 DataParallel 모델을 성공적으로 불러왔습니다.")
    else:
        # 'module.' 접두사가 없으면 일반 모델처럼 로드합니다.
        # model.load_state_dict(checkpoint,strict=False)
        model.load_state_dict(checkpoint,strict=False)
        print(f"'{MODEL_PATH}'에서 일반 모델을 성공적으로 불러왔습니다.")

    #strict=False
    """
    작동 원리: strict=False는 이름표가 정확히 일치하는 상자(레이어)의 가중치만 불러오고, 이름이 맞지
  않는(fc.weight vs fc.1.weight) 부분은 조용히 무시합니다. 따라서 ResNet의 대부분의 가중치는 재사용되고,
  model.fc 부분만 새로 초기화된 상태로 학습을 시작하게 됩니다.
    """
    model.eval() # 모델을 평가 모드로 설정합니다. (Dropout, BatchNorm 등이 추론 모드로 작동)
    print(f"'{MODEL_PATH}'에서 모델을 성공적으로 불러왔습니다.")

    # 2. 이미지 로드 및 전처리
    try:
        # PIL을 사용하여 이미지 파일을 엽니다. RGB 모드로 변환하여 3채널을 보장합니다.
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 입력 이미지 파일 '{image_path}'을 찾을 수 없습니다.")
        return None, None
    except Exception as e:
        print(f"오류: 이미지 로드 중 문제 발생: {e}")
        return None, None

    transform = get_transform() # 전처리 파이프라인을 가져옵니다.
    # 이미지에 전처리를 적용하고, 모델 입력에 필요한 배치 차원(unsqueeze(0))을 추가합니다.
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    print(f"입력 이미지 '{os.path.basename(image_path)}' 전처리 완료. 형태: {input_tensor.shape}")

    # 3. 딥페이크 예측
    with torch.no_grad(): # 예측 시에는 기울기 계산이 필요 없으므로 비활성화합니다.
        output = model(input_tensor) # 모델에 전처리된 이미지를 입력하여 예측 결과를 얻습니다.

    # 4. 예측 결과 해석
    # output은 로짓(logit) 값입니다. 이를 확률로 변환하기 위해 softmax를 사용합니다.
    probabilities = torch.softmax(output, dim=1)[0] # 배치 차원을 제거하고 확률만 가져옵니다.

    # 가장 높은 확률을 가진 클래스의 인덱스를 가져옵니다.
    _, predicted_class = torch.max(output.data, 1)

    # 예측 결과 출력
    class_labels = {0: "진짜 (Real)", 1: "가짜 (Deepfake)"}
    predicted_label = class_labels[predicted_class.item()]

    print(f"\n--- 예측 결과 ---")
    print(f"입력 이미지: '{os.path.basename(image_path)}'")
    print(f"진짜일 확률: {probabilities[0].item():.4f}")
    print(f"가짜일 확률: {probabilities[1].item():.4f}")
    print(f"최종 예측: {predicted_label}")

    if(predicted_label=="진짜 (Real)"):
        return predicted_label,probabilities[0].item()
    else:
        return predicted_label,probabilities[1].item()
    # return predicted_label, probabilities


# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    # 예측 함수 호출
    INPUT_IMAGE_PATH = "C:\\Users\\AI-00\\Desktop\\Portfolio_Web\\Portfolio_Server\\fake_ai.jfif"
    # INPUT_IMAGE_PATH = "C:\\Users\\AI-00\\Desktop\\Portfolio_Web\\Portfolio_Server\\leo_messi.jpg"
    predict_deepfake(INPUT_IMAGE_PATH)
