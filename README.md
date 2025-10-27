# XAI-Driven Deep Learning Model for Herbal Medicine

한약재 분류를 위한 설명 가능한 딥러닝 모델 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 한약재 이미지를 분류하기 위한 다양한 딥러닝 모델들을 구현하고 비교 분석합니다.

## 모델 구조

### 1. CNN (VGG16)
- 전통적인 Convolutional Neural Network
- VGG16 아키텍처 기반
- 3번 반복 실험을 통한 통계적 분석

### 2. ViT (Vision Transformer)
- Transformer 기반 이미지 분류 모델
- timm 라이브러리 활용
- 어텐션 메커니즘 분석

### 3. TransFG (Transformer with Fine-Grained)
- Global + Local Branch 구조
- Fine-grained 특징 추출
- 패치 중요도 기반 분석

## 데이터셋

- 한약재 이미지 데이터셋
- 증강된 데이터 포함
- Train/Validation/Test 분할

## 사용법

### CNN 모델 실행
```bash
python CNN.py --train_dir /path/to/train --val_dir /path/to/val --num_classes 3 --batch_size 16 --epochs 30
```

### ViT 모델 실행
```bash
python ViT.py --train_dir /path/to/train --val_dir /path/to/val --num_classes 3 --batch_size 16 --epochs 30
```

### TransFG 모델 실행
```bash
python TransFG.py --train_dir /path/to/train --val_dir /path/to/val --num_classes 3 --batch_size 16 --epochs 30
```

## 결과

- 표준편차 및 신뢰구간 포함한 Error Bar
- CSV 파일로 결과 저장
- 시각화 그래프 생성

## 요구사항

- Python 3.8+
- PyTorch
- torchvision
- timm
- matplotlib
- pandas
- numpy
- tqdm

## 라이선스

MIT License
