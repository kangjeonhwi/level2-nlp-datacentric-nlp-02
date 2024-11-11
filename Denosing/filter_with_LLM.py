"""
LLM을 이용한 뉴스 헤드라인 정제 스크립트

이 스크립트는 노이즈가 포함된 뉴스 헤드라인 데이터를 LLM(Language Model)을 사용하여 정제합니다.
주요 기능:
1. 입력된 CSV 파일에서 노이즈가 포함된 헤드라인을 읽어옵니다.
2. LLM을 사용하여 각 헤드라인의 노이즈를 제거하고 자연스러운 형태로 복원합니다.
3. 정제된 헤드라인을 새로운 CSV 파일로 저장합니다.

사용된 주요 함수:
- get_llm_result: LLM을 이용한 텍스트 처리
- remove_outer_quotes: 문자열의 앞뒤 따옴표 제거

"""

from utils import get_llm_result, remove_outer_quotes
import pandas as pd
import os
import torch
from tqdm import tqdm
import re


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')

    
if __name__ == '__main__':
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    data_path = os.path.join(DATA_DIR, "remove_symbols.csv")
    output_path = os.path.join(DATA_DIR, "cleaned_with_llm.csv")

    df = pd.read_csv(data_path)
    prompt = """당신은 전문적인 한국어 텍스트 정제 AI입니다. 다음은 노이즈가 포함된 한국어 뉴스 기사의 제목입니다. 이 제목에서 노이즈를 제거하고 원래의 자연스러운 뉴스 제목으로 복원해주세요.

                노이즈가 포함된 제목: '{input_text}'

                복원 시 다음 지침을 따라주세요:
                1. 무작위로 삽입된 영문자와 숫자를 제거하세요.
                2. 특수문자를 적절히 처리하세요.
                3. 줄임말이나 약어는 가능한 원래 형태로 복원하세요.
                4. 문맥을 고려하여 누락된 단어나 조사를 추가하세요.
                5. 제목의 전체적인 의미를 유지하면서 자연스러운 한국어 문장으로 만드세요.

                복원된 제목:"""
    
    cleaned_df = get_llm_result(df, prompt, output_path, "복원된 제목:")
    cleaned_df.loc[:, 'text'] = cleaned_df['text'].apply(remove_outer_quotes)
    cleaned_df.to_csv(output_path)