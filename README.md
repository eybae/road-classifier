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

## 구성
- `train.py`: 증강 데이터로 모델 학습
- `classify.py`: test 폴더 내 이미지 예측 후 결과 저장
- `models/`: 학습된 PyTorch 모델 (.pth)
- `test/`: 테스트용 이미지
- `result/`: 예측된 결과 이미지

---

## 🧪 예측 실행 방법

```bash
python classify.py

