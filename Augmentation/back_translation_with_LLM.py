"""
LLM을 이용한 Back Translation 구현 스크립트

이 스크립트는 한국어 텍스트를 영어로 번역한 후, 다시 한국어로 번역하는 back translation을 수행합니다.

주요 기능:
1. 한국어 텍스트를 영어로 번역 (Korean to English)
2. 번역된 영어 텍스트를 다시 한국어로 번역 (English to Korean)
3. 각 단계별 번역 결과를 CSV 파일로 저장

사용된 주요 함수:
- get_llm_result: LLM을 이용한 텍스트 번역
- remove_outer_quotes: 번역된 텍스트의 불필요한 따옴표 제거
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

def translate_text(df, prompt, output_path, slice_word):
    """
    LLM을 사용하여 텍스트를 번역하고 결과를 저장합니다.

    :param df: 번역할 텍스트가 포함된 DataFrame
    :param prompt: LLM에 전달할 번역 프롬프트
    :param output_path: 번역 결과를 저장할 파일 경로
    :param slice_word: LLM 출력을 슬라이싱할 기준 단어
    :return: 번역된 텍스트가 포함된 DataFrame
    """
    translated_df = get_llm_result(df, prompt, output_path, slice_word)
    translated_df.loc[:, 'text'] = translated_df['text'].apply(remove_outer_quotes)
    translated_df.to_csv(output_path, index=False)
    return translated_df

if __name__ == '__main__':
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    data_path = os.path.join(DATA_DIR, "cleaned_data.csv")
    df = pd.read_csv(data_path)

    # Korean to English translation
    kor2en_prompt = """You are an AI translator proficient in both Korean and English. Please translate the following Korean sentence into English.
                Korean sentence: '{input_text}'
                
                Please follow these guidelines when translating:
                1. Translate the Korean text into English. While doing so, maintain the original meaning of the sentence and consider the context to provide a natural translation.
                2. If there are any awkward parts in the text, please revise them to make them more natural. 
                3. If there are any incomplete or insufficient parts, please fill them in or expand on them as necessary.
                4. Adjust the translated result to fit the format of a news article headline, making it sound natural.

                Translated headline:"""
    
    kor2en_output_path = os.path.join(DATA_DIR, "kor2en_data.csv")
    kor2en_df = translate_text(df, kor2en_prompt, kor2en_output_path, "Translated headline:")

    # English to Korean translation
    en2kor_prompt = """당신은 한국어와 영어에 능통한 AI 번역가입니다. 다음 영어 문장을 한국어로 번역해주세요.
                영어 문장: '{input_text}'
                
                번역 시 다음 지침을 따라주세요:
                1. 영어 텍스트를 한국어로 번역하세요. 이 과정에서 문장의 원래 의미를 유지하고 맥락을 고려하여 자연스러운 번역을 제공하세요.
                2. 텍스트에 어색한 부분이 있다면 더 자연스럽게 수정해주세요.
                3. 불완전하거나 부족한 부분이 있다면 필요에 따라 채우거나 확장해주세요.
                4. 번역된 결과를 뉴스 기사 헤드라인 형식에 맞게 조정하여 자연스러운 뉴스기사 제목이 되도록 만드세요.

                번역된 헤드라인:"""
    
    en2kor_output_path = os.path.join(DATA_DIR, "en2kor_data.csv")
    en2kor_df = translate_text(kor2en_df, en2kor_prompt, en2kor_output_path, "번역된 헤드라인:")