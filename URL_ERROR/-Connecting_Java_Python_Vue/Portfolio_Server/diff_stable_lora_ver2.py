# =================================================================================================
# [초상세 주석 버전] Stable Diffusion + LoRA 모델 결합 및 이미지 생성 스크립트
# 이 스크립트는 사전 학습된 Stable Diffusion 모델에 LoRA 어댑터를 로드하여
# 특정 스타일이나 개념이 적용된 이미지를 생성하는 방법을 보여줍니다.
# =================================================================================================
import base64
import datetime
# ==============================================================================
# 섹션 1: 라이브러리 임포트 및 기본 설정
# ==============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch # PyTorch의 핵심 라이브러리
from PIL import Image # 이미지 파일 처리 (Pillow 라이브러리)
import matplotlib.pyplot as plt # 이미지 시각화

# diffusers 라이브러리에서 필요한 모듈 임포트
from diffusers import StableDiffusionPipeline # Stable Diffusion 파이프라인

# --- 하이퍼파라미터 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # GPU 또는 CPU 설정

# Stable Diffusion 기본 모델 경로 (Hugging Face Hub ID)
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
# BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# 로드할 LoRA 어댑터의 경로 (Hugging Face Hub ID 또는 로컬 경로)
# 이 예시에서는 Hugging Face Hub에 공개된 특정 LoRA 어댑터를 사용합니다.
# 실제 사용 시에는 학습된 자신의 LoRA 파일 경로로 변경해야 합니다.
LORA_MODEL_ID = "ostris/wan21_i2v_dolly_zoom_lora" # 예시 LoRA 모델 ID (특정 스타일 학습)
# LORA_MODEL_ID = "ostris/Flex.1-alpha-Redux" # 예시 LoRA 모델 ID (특정 스타일 학습)

# LORA_MODEL_ID = "./models/my_trained_lora.safetensors" # 로컬에 저장된 LoRA 파일 경로 예시

# ==============================================================================
# 섹션 2: Stable Diffusion 파이프라인 로드 및 LoRA 결합
# ==============================================================================

def load_and_combine_lora_model():
    """사전 학습된 Stable Diffusion 모델과 LoRA 어댑터를 로드하고 결합합니다."""
    print(f"\n1. 기본 Stable Diffusion 모델({BASE_MODEL_ID})을 로드합니다...")
    # StableDiffusionPipeline을 사용하여 사전 학습된 기본 모델을 로드합니다.
    # torch_dtype=torch.float16은 메모리 사용량을 줄여 GPU 메모리가 부족할 때 유용합니다.
    pipeline = StableDiffusionPipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
    
    # 파이프라인을 지정된 장치(GPU 또는 CPU)로 이동시킵니다.
    pipeline.to(DEVICE)
    print("기본 모델 로드 완료.")

    print(f"\n2. LoRA 어댑터({LORA_MODEL_ID})를 로드하고 결합합니다...")
    # LoRA 어댑터를 파이프라인에 로드하고 결합합니다.
    # load_lora_weights() 메서드는 LoRA 가중치를 불러와 파이프라인의 U-Net과 Text Encoder에 적용합니다.
    # 이 메서드는 Hugging Face Hub ID 또는 로컬 파일 경로를 모두 지원합니다.
    pipeline.load_lora_weights(LORA_MODEL_ID)
    print("LoRA 어댑터 로드 및 결합 완료.")

    return pipeline # LoRA가 결합된 파이프라인을 반환합니다.

# ==============================================================================
# 섹션 3: 이미지 생성 및 시각화
# ==============================================================================

def generate_and_display_image(pipeline,prompt,account):
    """결합된 모델을 사용하여 이미지를 생성하고 저장 및 표시합니다."""
    # 이미지 생성에 사용할 텍스트 프롬프트
    # 이 프롬프트는 LoRA 어댑터가 학습된 개념이나 스타일을 포함해야 합니다.
    # 'ostris/super-cereal-sd-v1-5' LoRA는 특정 스타일을 학습했으므로, 일반적인 프롬프트를 사용합니다.
    # prompt = "a photo of a cat, high quality, detailed, in super cereal style"
    # prompt = "A mysterious girl standing in a foggy forest, camera zooms in dramatically, her eyes glowing faintly"
    # prompt = "A beautiful girl standing in a forest"
    # account ="TempAccount"

    print(f"\n3. 이미지 생성을 시작합니다. 프롬프트: \"{prompt}\" ")
    
    # 이미지 생성
    # num_inference_steps: 노이즈 제거 단계 수. 높을수록 품질 좋지만 느림. (기본 50)
    # guidance_scale: 텍스트 프롬프트에 얼마나 충실할지 조절. (높을수록 프롬프트 따름, 기본 7.5)
    # with torch.no_grad() 블록 안에서 실행하여 메모리 사용량을 줄이고 속도를 높입니다.
    with torch.no_grad():
        generated_image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    print("이미지 생성이 완료되었습니다.")

    # 생성된 이미지 저장
    # output_dir = "./results_diffusion/stable_diffusion_lora"
    output_dir = f"D:\\PythonProject\\Portfolio_Server\\results\\{account}"
    # output_dir = f"C:\\Users\\AI-00\\Desktop\\Portfolio_Web\\portf\\public\\assets\\{account}"
    os.makedirs(output_dir, exist_ok=True)
    # timestamp
    timestamp=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp=str(timestamp)
    print(timestamp)
    output_path = os.path.join(output_dir, f"generated_image_{timestamp}_now.png")
    generated_image.save(output_path)
    try:
        with open(output_path,'rb') as imagePath:
            base64_image=imagePath.read()
            base64_image_encoding=base64.b64encode(base64_image)
            base64_image_encoding_tostring=base64_image_encoding.decode('utf-8')
    except FileNotFoundError:
        print(f"오류: '{output_path}' 파일을 찾을 수 없습니다. 파일 이름과 경로를 확인해주세요.")
    # vue보안피하기
    # output_path=f"/assets/{account}/generated_image_{timestamp}_now.png"
    print(f"생성된 이미지가 '{output_path}'에 저장되었습니다.")

    # print(base64_image_encoding_tostring)
    # return output_path
    return base64_image_encoding_tostring
    # matplotlib을 사용하여 이미지 표시
    # plt.figure(figsize=(8, 8)) # 그림 크기 설정
    # plt.imshow(generated_image) # 이미지 표시
    # plt.title("Generated Image with LoRA") # 제목 설정
    # plt.axis('off') # 축 숨기기
    # plt.show() # 화면에 그림 표시

# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    # 결과 및 모델 저장 폴더 생성
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    # 1. Stable Diffusion 파이프라인과 LoRA 어댑터 로드 및 결합
    sd_lora_pipeline = load_and_combine_lora_model()
    
    # 2. 결합된 모델을 사용하여 이미지 생성 및 표시
    aa=generate_and_display_image(sd_lora_pipeline)
    print(aa)
