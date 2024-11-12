import os
import pandas as pd
import matplotlib.pyplot as plt
import platform
import numpy as np

# 데이터 불러오기
data = pd.read_csv(os.path.join("./data", "train.csv"))
data = pd.read_csv("./data/train.csv")


# ASCII 비율 계산 함수
def calculate_ascii_ratio(text):
    ascii_chars = sum(1 for char in text if ord(char) < 128 and char != " ")  # ASCII 범위의 문자 개수
    return ascii_chars / len(text) if len(text) > 0 else 0  # ASCII 비율 계산


# 각 텍스트의 ASCII 비율을 계산하여 새로운 열로 추가
data["ascii_ratio"] = data["text"].apply(calculate_ascii_ratio)


# OS에 따라 한글 폰트 설정
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"  # MacOS
else:
    plt.rcParams["font.family"] = "NanumGothic"  # Linux (Ubuntu에 폰트 설치 필요)

# 마이너스 폰트 설정 (한글 폰트 사용 시 깨짐 방지)
plt.rcParams["axes.unicode_minus"] = False
# ASCII 비율의 분포 시각화
plt.hist(data["ascii_ratio"], bins=20, edgecolor="black")
plt.xlabel("ASCII 비율")
plt.ylabel("빈도수")
plt.title("텍스트 내 ASCII 비율 분포")
plt.show()
# 1600개의 데이터에 해당하는 상위/하위 임계값 계산
# 총 데이터 개수 중 상위 1600개에 해당하는 비율을 찾기 위해 탐색
values = np.linspace(0.0, 1.0, 101)
for i in values:
    threshold = data["ascii_ratio"].quantile(i)
    # 노이즈가 포함된 데이터 필터링
    noisy_data = data[data["ascii_ratio"] >= threshold]

    print(f"설정된 ASCII 비율 임계값: {threshold}")
    print(f"탐지된 노이즈 데이터 개수: {len(noisy_data)}")

    if len(noisy_data) < 1600:
        break
