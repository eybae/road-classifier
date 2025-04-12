# Road Image Classifier 🚗🛣️

이 프로젝트는 도시(city), 시골(country), 고속도로(highway) 이미지를 분류하는 PyTorch 기반 이미지 분류기입니다.

## 🔍 모델 정보
- 모델: ResNet34
- 입력 사이즈: 224x224
- 학습 배치 사이즈: 64
- 최고 정확도: 91.67%

### 📊 성능 (Test Set)
| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| City     | 0.83      | 0.83   | 0.83     | 6       |
| Country  | 0.86      | 1.00   | 0.92     | 6       |
| Highway  | 1.00      | 0.83   | 0.91     | 6       |

**Accuracy:** 89% (18개 이미지 기준)

---

## 🧪 예측 실행 방법

```bash
python classify.py



📁 폴더 설명
폴더/파일	설명
models/	학습된 모델 파일 (.pth) 저장 위치
test/	테스트할 이미지들 저장 위치
result/	예측 결과 이미지가 저장됨 (자동 생성됨)
classify.py	예측 실행 코드
train.py (선택)	모델 학습 코드