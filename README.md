# Topic Classification in Data-Centric
[Wrap-UP Report](https://basalt-viscount-15f.notion.site/Topic-Classification-in-Data-1a8cec41a1a28022b928f843df7da24a?pvs=4)
한국어 신문기사 헤드라인을 기반으로 카테고리를 분류하는 Data-Centric 프로젝트입니다.

## 프로젝트 개요

자연어 독해 및 분석을 위해 텍스트의 주제에 대한 이해는 필수적입니다. 이 프로젝트는 한국어 신문기사 헤드라인을 입력으로 받아 해당 기사의 카테고리를 예측하는 분류 문제를 다룹니다.

Data-Centric의 취지에 맞게, 주어진 데이터에는 무작위적으로 노이즈가 포함되어 있으며, 베이스라인 모델의 수정 없이 오직 데이터 품질 개선을 통해 최대의 성능 향상을 목표로 합니다.

## 데이터셋

KLUE-TC(YNAT) 데이터셋과 동일한 형식을 가진 데이터로 구성되어 있습니다:

- **Train dataset**: 총 2,800개
    - **text noise data**: 1,600개 (텍스트에 20%~80% 노이즈 추가)
    - **label noise data**: 1,000개 (임의로 재지정된 라벨)
    - **normal data**: 200개 (노이즈 없음)
- **Test dataset**: 총 30,000개 (인위적 노이즈 없음)

각 데이터는 다음 필드로 구성됩니다:

- **ID**: 각 데이터 샘플의 고유 번호
- **text**: 분류 대상이 되는 자연어 텍스트 (한국어 텍스트에 일부 영어, 한자, 특수문자 등이 무작위로 삽입됨)
- **label**: 정수로 인코딩된 라벨 (생활문화, 스포츠, 세계, 정치, 경제, IT과학, 사회 중 하나)

## 방법론

### 1. 노이즈 제거 (Denoising)

### Rule-based 접근법

- ASCII 코드 비율을 통해 text noise data 선별 (threshold = 0.3)
- 추출된 text noise data에서 특수문자 제거
- 한국어, 영어, 숫자, 한자를 제외한 모든 문자를 특수문자로 간주하고 제거
- 서로 다른 언어가 붙어있는 경우 공백 추가

### LLM 프롬프트

Rule-based로 식별된 Noise text에 대해 LLM을 이용한 정제 작업 수행:

- 특수문자, 영어, 한자를 공백으로 치환한 후 비어있는 부분 유추
- 특수문자만 제거한 후 맥락과 가장 유사한 헤드라인 생성

### 2. 라벨 재지정 (Relabeling)

### Clustering

K-Means Clustering을 통해 각 label의 중심을 계산하고, 각 중심에서 가장 가까운 label을 찾아 label noise data에 해당 label 부여

### Self-train

BERT 기반의 분류 모델을 통해 text noise data를 이용해 학습 후, label noise data에 대한 모델의 예측값을 이용해 Relabeling 진행

### Cleanlab

Self-train으로 얻은 확률값을 활용하여 LightGBM 분류기를 통해 기존 label의 불일치를 분석하고 잠재적인 라벨 오류 탐지

### 3. 데이터 증강 (Augmentation)

- **Back-Translation**: LLM을 사용하여 한국어→영어→한국어 번역을 통한 데이터 증강
- **Paraphrasing**: LLM을 활용해 문장마다 유사한 의미의 변형 문장 5개 생성
- **EDA (Easy Data Augmentation)**: 랜덤으로 텍스트에 노이즈 삽입
- **BM25 기반 필터링**: 증강된 데이터와 기존 데이터 간 유사도 측정, 유사도가 낮은 새 데이터만 선별 추가

## 결과

단계적 접근 방식을 통해 최종적으로 약 8,000개의 데이터셋을 구성하여 학습한 결과:

- 베이스라인: Accuracy 0.61, F1 Score 0.60
- 최종 결과: Accuracy 0.85, F1 Score 0.85

## 적용 절차

1. Rule-based 방식으로 text noise data 식별 및 LLM을 활용한 노이즈 정제
2. 정제된 데이터셋으로 RoBERTa 기반 분류 모델 학습
3. 11,000개의 증강 데이터 생성 및 BM25 알고리즘으로 5,000개 선별
4. 원본 데이터 2,760개에 선별된 데이터를 1,500개씩 단계적 추가
5. 각 단계마다 Self-train 진행하여 relabeling 과정 3회 반복

## 사용한 오픈소스 LLM 모델

- [Linkbricks-Horizon-AI-Korean-Pro-8B](https://huggingface.co/Saxo/Linkbricks-Horizon-AI-Korean-Pro-8B)
- [Linkbricks-Llama3.2-Korean-cpt-3b](https://huggingface.co/Saxo/Linkbricks-Llama3.2-Korean-cpt-3b)
- [llama-3.2-Korean-Bllossom-3B](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B)
- [polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)
- [EEVE-Korean-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0)

## 향후 개선 방향

- **Cleanlab**: label_quality가 낮은 데이터에 대한 처리 방법 개선
- **Clustering**: 더 효과적인 임베딩 벡터 계산 방법 적용
- **Contrastive Learning**: 문법적/문맥적 정보와 함께 label별 주제 의미를 구별할 수 있는 fine-tuning 방법 개발

## 참고 자료

- [Cleanlab 공식 문서](https://docs.cleanlab.ai/stable/index.html)
