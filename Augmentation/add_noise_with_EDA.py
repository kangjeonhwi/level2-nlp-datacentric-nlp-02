import pandas as pd
import os
from eda import eda

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')

# 데이터 로드
data_path = os.path.join(DATA_DIR, "cleaned_data.csv")
df = pd.read_csv(data_path)

# EDA 파라미터 설정
num_aug = 4  # 각 문장당 생성할 증강 데이터 수
alpha_sr = 0.1  # 동의어 교체 비율
alpha_ri = 0.1  # 무작위 삽입 비율
alpha_rs = 0.1  # 무작위 교환 비율
alpha_rd = 0.1  # 무작위 삭제 비율

# 노이즈 추가 함수
def add_noise(row):
    text = row['text']
    augmented_texts = eda.eda(text, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    return pd.DataFrame({'id': [row['id']] * (num_aug + 1),
                         'text': [text] + augmented_texts,
                         'label': [row['label']] * (num_aug + 1)})

# 데이터 증강 적용
augmented_df = df.apply(add_noise, axis=1)
augmented_df = pd.concat(augmented_df.tolist(), ignore_index=True)

# 결과 확인
print(augmented_df.head(10))

# CSV 파일로 저장
output_path = 'augmented_with_eda.csv'
output_path = os.path.join(OUTPUT_DIR,output_path)
augmented_df.to_csv(output_path, index=False)