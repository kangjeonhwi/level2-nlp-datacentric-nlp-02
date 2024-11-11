"""
BM25를 이용한 데이터 증강 및 필터링 스크립트

이 스크립트는 원본 데이터와 증강된 데이터를 병합하고, BM25 알고리즘을 사용하여
유사도가 낮은 문장만을 선별하여 최종 데이터셋을 생성합니다.

주요 기능:
1. 원본 데이터와 여러 증강 데이터 파일을 로드합니다.
2. BM25 알고리즘을 사용하여 증강된 문장과 원본 문장 간의 유사도를 계산합니다.
3. 유사도가 특정 임계값 미만인 문장만을 선별합니다.
4. 선별된 문장을 원본 데이터와 병합하여 최종 데이터셋을 생성합니다.
"""

from rank_bm25 import BM25Okapi
from utils import merge_dataframes
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import numpy as np
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
TOKENIZER_MODEL_NAME = "klue/roberta-large"
THRESHOLD = 30
OUTPUT_NAME = "augmented_data.csv"

# 데이터 및 토크나이저 로드
origin_df = pd.read_csv(os.path.join(DATA_DIR, 'origin_data.csv'))
augmented_data_paths = ['/augmented1.csv', '/augmented2.csv', '/augmented3.csv']
augmented_df_list = [pd.read_csv(os.path.join(DATA_DIR, path)) for path in augmented_data_paths]
augmented_df = merge_dataframes(augmented_df_list)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

# 코퍼스 생성
corpus = origin_df['text'].values.tolist()
corpus = [sentence for sentence in corpus if len(sentence) > 5]
tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]

augmented_sentences = augmented_df['text'].values.tolist()
augmented_sentences = [sentence for sentence in augmented_sentences if len(sentence) > 5]

# BM25 모델 생성
bm25 = BM25Okapi(tokenized_corpus)

# 유사도 계산 및 선택
selected_sentences = []

for aug_sentence in tqdm(augmented_sentences, desc="Filtering sentences"):
    tokenized_query = tokenizer.tokenize(aug_sentence)
    scores = bm25.get_scores(tokenized_query)
    max_score = max(scores)

    if max_score < THRESHOLD:
        selected_sentences.append(aug_sentence)

# 선택된 문장들 추가
final_sentences = corpus + selected_sentences

# 문장 리스트를 DataFrame으로 변환
data = {
    'ID': [f'ynat-v1_train_{i:05d}' for i in range(len(final_sentences))],
    'text': final_sentences,
    'target': [0] * len(final_sentences)
}

augmented_df = pd.DataFrame(data)
augmented_df.to_csv(os.path.join(DATA_DIR, OUTPUT_NAME), index=False)