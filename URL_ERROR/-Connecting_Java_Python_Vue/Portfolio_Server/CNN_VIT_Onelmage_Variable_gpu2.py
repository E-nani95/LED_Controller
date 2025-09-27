# =================================================================================================
# [단일 이미지 예측] 학습된 ResNet50+ViT 앙상블 모델로 이미지 판별
# 이 스크립트는 사전에 학습된 ResNet50과 ViT 모델의 가중치를 불러온 후,
# 코드 내에 지정된 단일 이미지에 대해 딥페이크 여부를 예측하고 결과를 출력합니다.
#
# [사용법]
# 1. 아래 IMAGE_TO_TEST 변수에 판별하고 싶은 이미지의 전체 경로를 입력합니다.
# 2. 터미널에서 python CNN_VIT_OneImage.py 명령어로 스크립트를 실행합니다.
# =================================================================================================

# ==============================================================================
# 섹션 1: 라이브러리 임포트 및 기본 설정
# ==============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict # OrderedDict 임포트 추가

# --- 기본 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 2

# --- 불러올 학습된 모델 경로 ---
# CNN_VIT_01.py 또는 CNN_VIT_02.py 에서 저장한 모델 경로와 동일해야 합니다.
RESNET_MODEL_PATH = "./models/deepfake_resnet50_classifier_backup_cnnvit.pth"
VIT_MODEL_PATH = "./models/deepfake_vit_b_16_classifier_backup_cnnvit.pth"
# deepfake_resnet50_classifier_backup_cnn.pth 이랑 deepfake_vit_b_16_classifier_backup_cnnvit.pth 가 잘맞음
# ==============================================================================
# 섹션 2: 모델 정의 (학습 스크립트와 동일한 구조)
# ==============================================================================

def get_resnet_model(num_classes=NUM_CLASSES, dropout_rate=0.5):
    """사전 학습된 ResNet50 모델의 구조를 정의합니다."""
    model = models.resnet50(weights=None) # 가중치는 파일에서 불러오므로 None으로 설정
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(DEVICE)
    return model

def get_vit_model(num_classes=NUM_CLASSES, dropout_rate=0.5):
    """사전 학습된 ViT 모델의 구조를 정의합니다."""
    model = models.vit_b_16(weights=None) # 가중치는 파일에서 불러오므로 None으로 설정
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(DEVICE)
    return model

# ==============================================================================
# 섹션 3: 앙상블 예측 함수
# ==============================================================================

def ensemble_predict_single_image(resnet_model, vit_model, image_path, transform):
    """두 모델의 예측을 앙상블하여 단일 이미지를 예측합니다."""
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 입력 이미지 파일 '{image_path}'을 찾을 수 없습니다. 경로를 확인하세요.")
        return
    except Exception as e:
        print(f"오류: 이미지를 여는 중 문제가 발생했습니다: {e}")
        return

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    class_labels = {0: "진짜 (Real)", 1: "가짜 (Deepfake)"}

    # 예측 시에는 기울기 계산이 필요 없습니다.
    with torch.no_grad():
        # --- 각 모델의 예측 확률 계산 ---
        resnet_output = resnet_model(input_tensor)
        resnet_probs_batch = torch.softmax(resnet_output, dim=1)

        vit_output = vit_model(input_tensor)
        vit_probs_batch = torch.softmax(vit_output, dim=1)

        # --- 두 모델의 예측 확률을 평균내어 앙상블 ---
        ensemble_probs_batch = (resnet_probs_batch + vit_probs_batch) / 2
        confidence, predicted_class = torch.max(ensemble_probs_batch, 1)
        predicted_label = class_labels[predicted_class.item()]
        ensemble_probs = ensemble_probs_batch[0]

    print(f"\n====== 예측 결과 ======")
    print(f"입력 이미지: '{os.path.basename(image_path)}'")
    print("--------------------------------------------------")
    print(f"  - ResNet50 예측: 진짜 확률 {resnet_probs_batch[0][0]:.4f} / 가짜 확률 {resnet_probs_batch[0][1]: .4f}")
    print(f"  - ViT      예측: 진짜 확률 {vit_probs_batch[0][0]:.4f} / 가짜 확률 {vit_probs_batch[0][1]: .4f}")
    print("--------------------------------------------------")
    print(f"  => 최종 앙상블 예측: {predicted_label}")
    print(f"     (신뢰도: {confidence.item():.2%}, 진짜 확률 {ensemble_probs[0]:.4f} / 가짜 확률 {ensemble_probs[1]:.4f})")
    print(f"=======================")

# ==============================================================================
# 섹션 4: 메인 실행 함수
# ==============================================================================

def main():
    # --- 1. 예측할 이미지 경로 설정 ---
    # 여기에 판별하고 싶은 이미지의 전체 경로를 큰따옴표 안에 입력하세요.
    # Windows 예시: IMAGE_TO_TEST = "C:\\Users\\사용자명\\Desktop\\내사진.jpg"
    # 중요: 경로의 역슬래시(\)는 두 번 (\\) 써주거나, 슬래시(/)로 바꿔주세요.
    # IMAGE_TO_TEST = "D:\PythonProject\Dlib_Preprocess\cropped_face_1.jpg"
    # IMAGE_TO_TEST = "D:\PythonProject\Dlib_Preprocess\IMG_5568.jpeg"
    IMAGE_TO_TEST = "D:\PythonProject\Dlib_Preprocess\images_ai_01.jpg"
    # IMAGE_TO_TEST = "D:\\PythonProject\\diffusion_Model\\0e23d546a5f952542a00_021.mp4_0_11.jpg"

    # --- 2. 모델 불러오기 ---
    print("학습된 모델 가중치를 불러옵니다...")
    
    try:
        resnet_model = get_resnet_model()
        # DataParallel로 저장된 모델의 state_dict 처리
        resnet_state_dict = torch.load(RESNET_MODEL_PATH, map_location=DEVICE)
        new_resnet_state_dict = OrderedDict()
        for k, v in resnet_state_dict.items():
            name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
            new_resnet_state_dict[name] = v
        resnet_model.load_state_dict(new_resnet_state_dict, strict=False)
        resnet_model.eval()
        print(f" - ResNet50 모델 로드 완료: '{RESNET_MODEL_PATH}'")
    except FileNotFoundError:
        print(f"오류: ResNet 모델 파일('{RESNET_MODEL_PATH}')을 찾을 수 없습니다. 먼저 CNN_VIT_01.py 또는 CNN_VIT_02.py를 실행하여 모델을 학습시키세요.")
        return

    try:
        vit_model = get_vit_model()
        # DataParallel로 저장된 모델의 state_dict 처리
        vit_state_dict = torch.load(VIT_MODEL_PATH, map_location=DEVICE)
        new_vit_state_dict = OrderedDict()
        for k, v in vit_state_dict.items():
            name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
            new_vit_state_dict[name] = v
        vit_model.load_state_dict(new_vit_state_dict, strict=False)
        vit_model.eval()
        print(f" - ViT 모델 로드 완료: '{VIT_MODEL_PATH}'")
    except FileNotFoundError:
        print(f"오류: ViT 모델 파일('{VIT_MODEL_PATH}')을 찾을 수 없습니다. 먼저 CNN_VIT_01.py 또는 CNN_VIT_02.py를 실행하여 모델을 학습시키세요.")
        return

    # --- 3. 이미지 전처리 설정 ---
    predict_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 4. 예측 실행 ---
    if IMAGE_TO_TEST == "여기에_이미지_전체_경로를_입력하세요.jpg":
        print("\n[오류] 예측할 이미지 경로를 설정해주세요.")
        print("스크립트의 IMAGE_TO_TEST 변수에 이미지 파일의 전체 경로를 입력해야 합니다.")
        return

    ensemble_predict_single_image(resnet_model, vit_model, IMAGE_TO_TEST, predict_transform)

# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    # 스크립트를 실행하면 바로 main 함수가 호출됩니다.
    main()
