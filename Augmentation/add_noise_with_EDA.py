"""
EDA(Easy Data Augmentation)를 이용한 텍스트 데이터 증강 스크립트

이 스크립트는 주어진 CSV 파일의 텍스트 데이터에 EDA 기법을 적용하여 데이터를 증강합니다.
증강된 데이터는 새로운 CSV 파일로 저장됩니다.

주요 기능:
1. 입력된 CSV 파일에서 텍스트 데이터를 읽어옵니다.
2. EDA 기법을 사용하여 각 텍스트에 대해 여러 개의 증강된 버전을 생성합니다.
3. 원본 데이터와 증강된 데이터를 포함한 새로운 DataFrame을 생성합니다.
4. 결과를 CSV 파일로 저장합니다.

사용된 EDA 기법:
- 동의어 교체 (Synonym Replacement)
- 무작위 삽입 (Random Insertion)
- 무작위 교환 (Random Swap)
- 무작위 삭제 (Random Deletion)
"""

import pandas as pd
import os
from eda import eda

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')

def add_noise(row, num_aug, alpha_sr, alpha_ri, alpha_rs, alpha_rd):
    """
    주어진 행의 텍스트에 EDA를 적용하여 증강된 텍스트를 생성합니다.

    :param row: DataFrame의 한 행
    :param num_aug: 생성할 증강 데이터의 수
    :param alpha_sr: 동의어 교체 비율
    :param alpha_ri: 무작위 삽입 비율
    :param alpha_rs: 무작위 교환 비율
    :param alpha_rd: 무작위 삭제 비율
    :return: 증강된 데이터를 포함한 DataFrame
    """
    text = row['text']
    augmented_texts = eda.eda(text, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    return pd.DataFrame({
        'id': [row['id']] * (num_aug + 1),
        'text': [text] + augmented_texts,
        'label': [row['label']] * (num_aug + 1)
    })

def augment_data(input_path, output_path, num_aug=4, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1):
    """
    CSV 파일의 텍스트 데이터를 증강하고 결과를 새 CSV 파일로 저장합니다.

    :param input_path: 입력 CSV 파일 경로
    :param output_path: 출력 CSV 파일 경로
    :param num_aug: 각 문장당 생성할 증강 데이터 수
    :param alpha_sr: 동의어 교체 비율
    :param alpha_ri: 무작위 삽입 비율
    :param alpha_rs: 무작위 교환 비율
    :param alpha_rd: 무작위 삭제 비율
    """
    df = pd.read_csv(input_path)
    
    augmented_df = df.apply(lambda row: add_noise(row, num_aug, alpha_sr, alpha_ri, alpha_rs, alpha_rd), axis=1)
    augmented_df = pd.concat(augmented_df.tolist(), ignore_index=True)

    augmented_df.to_csv(output_path, index=False)
    print(f'증강된 데이터가 {output_path}에 저장되었습니다.')

if __name__ == '__main__':
    input_file = "cleaned_data.csv"
    output_file = "augmented_with_eda.csv"
    
    input_path = os.path.join(DATA_DIR, input_file)
    output_path = os.path.join(OUTPUT_DIR, output_file)

    augment_data(input_path, output_path)