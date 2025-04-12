import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import shutil

# 경로 설정
input_dir = "dataset_split/train"  # 원본 데이터셋 경로
train_output_dir = "augmented_dataset/train"
val_output_dir = "augmented_dataset/val"

# 파라미터 설정
val_split = 0.2  # validation 데이터 비율
augment_count = 5  # 각 이미지 당 증강 횟수

# 증강 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# 출력 디렉토리 생성
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# 클래스별 처리
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    os.makedirs(os.path.join(train_output_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_output_dir, class_name), exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # 증강 데이터 저장
    for file in tqdm(train_files, desc=f"Augmenting {class_name} (train)"):
        img_path = os.path.join(class_path, file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"이미지 열기 실패: {img_path}, 에러: {e}")
            continue

        for i in range(augment_count):
            augmented = transform(image)
            save_name = f"{os.path.splitext(file)[0]}_aug{i}.jpg"
            save_path = os.path.join(train_output_dir, class_name, save_name)
            augmented.save(save_path)

    # validation 데이터는 증강 없이 복사
    for file in tqdm(val_files, desc=f"Copying {class_name} (val)"):
        src_path = os.path.join(class_path, file)
        dst_path = os.path.join(val_output_dir, class_name, file)
        shutil.copy(src_path, dst_path)

print("✅ 증강 완료 및 데이터셋 저장 완료!")
