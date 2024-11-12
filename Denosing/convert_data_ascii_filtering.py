import os
import pandas as pd

# 데이터 불러오기
data = pd.read_csv("./data/train.csv")


# ASCII 비율 계산 함수
def calculate_ascii_ratio(text):
    ascii_chars = sum(1 for char in text if ord(char) < 128 and char != " ")  # ASCII 범위의 문자 개수
    return ascii_chars / len(text) if len(text) > 0 else 0  # ASCII 비율 계산


# ASCII 비율을 계산하여 새로운 열 추가
data["ascii_ratio"] = data["text"].apply(calculate_ascii_ratio)
print(data)


# ASCII 코드 문자를 공백으로 치환하는 함수
def replace_ascii_with_space(text):
    return "".join(char if ord(char) >= 128 or char == " " else "^" for char in text)


threshold = 0.2

# ASCII 비율이 threshold 이상인 경우 ASCII 코드 문자 치환
data.loc[data["ascii_ratio"] >= threshold, "text"] = data.loc[data["ascii_ratio"] >= threshold, "text"].apply(replace_ascii_with_space)

# 불필요한 열 제거 (ascii_ratio)
# data = data.drop(columns=["ascii_ratio"])
data["is_noisy"] = data["ascii_ratio"].apply(lambda x: 1 if x > threshold else 0)
# 결과를 새로운 CSV 파일로 저장
data.to_csv(f"./data/train_modified_t{threshold}_ver2.csv", index=False)
