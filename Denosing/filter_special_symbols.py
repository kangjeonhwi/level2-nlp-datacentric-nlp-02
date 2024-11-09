import pandas as pd
import re
import os
from typing import Dict, Any

"""
이 스크립트는 텍스트 데이터에서 노이즈를 감지하고 제거하는 기능을 수행합니다.
특수 문자, 숫자, 한글의 비율을 계산하여 노이즈를 식별하고, 정제된 텍스트를 생성합니다.
"""

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')

def calculate_ratio(text: str, pattern: str) -> float:
    """
    텍스트 내 특정 패턴의 비율을 계산합니다.
    
    :param text: 분석할 텍스트 문자열
    :param pattern: 찾을 정규 표현식 패턴
    :return: 패턴의 비율 (0~1 사이의 float)
    """
    return len(re.findall(pattern, text)) / len(text) if text else 0

def special_char_ratio(text: str) -> float:
    """텍스트 내 특수 문자의 비율을 계산합니다."""
    return calculate_ratio(text, r'[^\w\s]')

def number_ratio(text: str) -> float:
    """텍스트 내 숫자의 비율을 계산합니다."""
    return calculate_ratio(text, r'\d')

def hangul_ratio(text: str) -> float:
    """텍스트 내 한글의 비율을 계산합니다."""
    return calculate_ratio(text, r'[가-힣]')

def is_noisy(row: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    """
    주어진 임계값을 기준으로 텍스트가 노이즈인지 판단합니다.
    
    :param row: 데이터프레임의 행
    :param thresholds: 각 비율의 임계값을 담은 딕셔너리
    :return: 노이즈 여부 (Boolean)
    """
    return (row['special_char_ratio'] > thresholds['special'] or 
            row['number_ratio'] > thresholds['number'] or 
            row['hangul_ratio'] < thresholds['hangul'])

def clean_text(text: str) -> str:
    """
    텍스트를 정제합니다. 특수 문자를 제거하고 불필요한 공백을 제거합니다.
    
    :param text: 정제할 텍스트 문자열
    :return: 정제된 텍스트 문자열
    """
    text = re.sub(r'[^\w\s가-힣]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def filter_by_ratio(df: pd.DataFrame, output_name: str, thresholds: Dict[str, float]):
    """
    데이터프레임의 텍스트를 분석하여 노이즈를 식별하고 제거한 후 결과를 저장합니다.
    
    :param df: 처리할 데이터프레임
    :param output_name: 출력 파일 이름
    :param thresholds: 각 비율의 임계값을 담은 딕셔너리
    """
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    for ratio_func in [special_char_ratio, number_ratio, hangul_ratio]:
        df[f'{ratio_func.__name__}'] = df['text'].apply(ratio_func)
    
    df['is_noisy'] = df.apply(lambda row: is_noisy(row, thresholds), axis=1)
    noisy_df = df[df['is_noisy']].copy()
    noisy_df.loc[:, 'text'] = noisy_df['text'].apply(clean_text)
    noisy_df = noisy_df.drop(['is_noisy', 'special_char_ratio', 'number_ratio', 'hangul_ratio'], axis=1)

    noisy_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == '__main__':
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    data_path = os.path.join(DATA_DIR, "train.csv")
    df = pd.read_csv(data_path)
    
    thresholds = {
        'special': 0.16,
        'number': 0.2,
        'hangul': 0.55
    }
    
    filter_by_ratio(df, 'remove_symbols.csv', thresholds)