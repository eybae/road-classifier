import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

# 설정
model_path = 'models/best_model_91.67.pth'
test_dir = 'test'
result_dir = 'result'
class_names = ['city', 'country', 'highway']  # 클래스 순서 맞게 조정

# 결과 폴더 없으면 생성
os.makedirs(result_dir, exist_ok=True)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 모델 로드 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 폰트 설정 (Linux 환경)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font = ImageFont.load_default()

# 테스트 이미지 예측 및 저장
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_dir, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            predicted_label = class_names[pred.item()]
            confidence = conf.item() * 100  # 퍼센트 변환

        # 결과 이미지에 텍스트 삽입
        image_with_text = image.copy()
        draw = ImageDraw.Draw(image_with_text)
        text = f'Prediction: {predicted_label} ({confidence:.1f}%)'
        draw.text((10, 10), text, fill=(255, 0, 0), font=font)

        # 저장
        result_path = os.path.join(result_dir, filename)
        image_with_text.save(result_path)

        # 터미널 출력
        print(f'{filename}: {text}')
